import json
import re
import pathlib
from typing import Any, Dict, List, Set, Callable

from cardiology_gen_ai import EmbeddingConfig, IndexingConfig
from langchain_core.documents.base import Document
from pydantic import BaseModel, ConfigDict

from src.agentic_rag.managers.search_manager import SearchManager
from src.agentic_rag.config.manager import SearchConfig
from tests.accuracy_tests.eval import EvalTestConfig, EvalTest
from tests.accuracy_tests.nodes_config import NodeOptimizationConfig


class RetrieverTestConfig(EvalTestConfig):
    index_config: IndexingConfig
    search_config: SearchConfig
    embeddings_config: EmbeddingConfig

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "RetrieverTestConfig":
        embedding_dict = config_dict["embeddings"]
        embedding_config = EmbeddingConfig.from_config(embedding_dict)
        indexing_dict = config_dict["indexing"]
        indexing_config = IndexingConfig.from_config(indexing_dict)
        search_dict = config_dict["search"]
        search_config = SearchConfig.from_config(search_dict)
        base_kwargs = {
            k: v for k, v in config_dict.items()
            if k in EvalTestConfig.model_fields
        }
        return cls(
            **base_kwargs, index_config=indexing_config, search_config=search_config, embeddings_config=embedding_config
        )


# TODO: maybe this is useful also for the graph
class SearchResult(BaseModel):
    chunks: List[Document]
    normalize: bool = True

    def extract_unique_chunks(self) -> List[Document]:
        unique_chunks, unique_sources = set(), []
        for chunk in self.chunks:
            doc_info = chunk.metadata
            if (doc_info["filename"], doc_info["chunk_id"]) not in unique_chunks:
                unique_sources.append(chunk)
                unique_chunks.add((doc_info["filename"], doc_info["chunk_id"]))
        return unique_sources

    def normalize_filename(self, filename: str) -> str:
        return filename.split("/")[-1] if self.normalize else filename

    def normalize_headers(self, headers: Dict[str, str]) -> Dict[str, str] | List[str]:
        return list(headers.values()) if self.normalize else headers

    def extract_unique_filenames(self) -> List[str]:
        return list(set([chunk.metadata["filename"] for chunk in self.chunks]))

    def group_by_filename(self) -> Dict[str, Set[str]]:
        sections_by_filename = dict()
        for chunk in self.chunks:
            filename = self.normalize_filename(chunk.metadata["filename"])
            chunk_sections = self.normalize_headers(chunk.metadata["headers"])
            if filename not in list(sections_by_filename.keys()):
                sections_by_filename[filename] = set()
            sections_by_filename[filename].update(chunk_sections)
        return sections_by_filename


class Source(BaseModel):
    document: str
    sections: List[str]


class RetrieverData(BaseModel):
    question: str
    sources: List[Source]
    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_search_result(cls, question: str, search_result: SearchResult) -> "RetrieverData":
        grouped_sources = search_result.group_by_filename()
        sources = [Source(document=doc, sections=list(sections)) for doc, sections in grouped_sources.items()]
        return cls(question=question, sources=sources)

    def extract_unique_filenames(self) -> List[str]:
        return list(set([source.document for source in self.sources]))

    def group_sources_by_filename(self) -> Dict[str, Set[str]]:
        sections_by_filename = dict()
        for source in self.sources:
            filename, file_sections  = source.document, source.sections
            if filename not in list(sections_by_filename.keys()):
                sections_by_filename[filename] = set()
            sections_by_filename[filename].update(file_sections)
        return sections_by_filename


class EvalRetriever(EvalTest):
    def __init__(self, test_config_path: pathlib.Path, node_config: NodeOptimizationConfig):
        super().__init__(test_config_path=test_config_path, node_config=node_config)
        search_manager = SearchManager(
            index_config=self.test_config.index_config,
            search_config=self.test_config.search_config,
            embeddings=self.test_config.embeddings_config
        )
        self.retriever = search_manager.vectorstore.retriever

    def get_config(self):
        config_dict = self._load_config()
        return RetrieverTestConfig.from_config(config_dict)

    @staticmethod
    def _normalize_text(s: str) -> str:
        """
        Helper to normalize section/header strings:
        - lowercase
        - remove common punctuation
        - collapse whitespace
        """
        s = s.lower()
        s = re.sub(r"[/#*()]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def get_data(self, **kwargs) -> List[Dict]:
        with open(self.test_config.data_folder / self.test_config.data_name, "r", encoding="utf-8") as f:
            items = json.load(f)
        retriever_data_list = [RetrieverData.model_validate(item) for item in items]
        invoke_dict_list = []
        for retriever_data in retriever_data_list:
            retriever_data_dict = retriever_data.model_dump()
            if "document" in self.node_config.output_keys:
                invoke_dict = {
                    "inputs": {"_input": retriever_data_dict.get("question")},
                    "expectations": {"_output": retriever_data.extract_unique_filenames()},
                }
                invoke_dict_list.append(invoke_dict)
            elif "sections" in self.node_config.output_keys:
                grouped_sources = retriever_data.group_sources_by_filename()
                for source, source_headers in grouped_sources.items():
                    invoke_dict = {
                        "inputs": {"_input": retriever_data_dict.get("question"), "_context": source},
                        "expectations": {"_output": [self._normalize_text(header) for header in  list(source_headers)]},
                    }
                    invoke_dict_list.append(invoke_dict)
            else:
                invoke_dict = {
                    "inputs": {"question": retriever_data_dict.get("question")},
                    "expectations": {"sources": retriever_data_dict.get("sources")},
                }
                invoke_dict_list.append(invoke_dict)
        return invoke_dict_list

    def post_process(self, search_results: SearchResult, **kwargs) -> List[str]:
        if "document" in self.node_config.output_keys:
            return search_results.extract_unique_filenames()
        elif "sections" in self.node_config.output_keys:
            grouped_sources = search_results.group_by_filename()
            for filename, file_sections in grouped_sources.items():
                grouped_sources[filename] = [self._normalize_text(section) for section in list(file_sections)]
            return grouped_sources.get(kwargs.get("_context"), [])
        return []

    def get_predict_fn(self, **kwargs) -> Callable:
        def predict_fn(**inputs):
            results = SearchResult(chunks=self.retriever.invoke(inputs.get("_input")))
            if "document" in self.node_config.output_keys or "sections" in self.node_config.output_keys:
                results = self.post_process(results, _context=inputs["_context"]) if "_context" in inputs.keys() else self.post_process(results)
            return {"_output": results}
        return predict_fn

    def run_eval(self, save: bool = True):
        results = self.get_eval_results()
        if save:
            _ = results["index_config"].pop("folder")
            results["index_config"]["type"] = results["index_config"].get("type").value
            results["index_config"]["distance"] = results["index_config"].get("distance").value
            results["index_config"]["retrieval_mode"] = results["index_config"].get("retrieval_mode").value
            results["search_config"]["type"] = results["search_config"].get("type").value
            _ = results["embeddings_config"].pop("model")
            _ = results["embeddings_config"].pop("kwargs")
            results_file = (
                    self.test_config.results_folder/ f"run_{self.test_config.test_id}" / f"{self.node_config.name}.json")
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Eval results saved in: {results_file}")