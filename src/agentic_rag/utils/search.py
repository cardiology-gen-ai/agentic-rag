from typing import List, Callable, Dict, Set

from langchain_core.documents import Document
from pydantic import BaseModel

from agentic_rag.config.manager import FusionStrategy, SearchConfig


class SearchResult(BaseModel):
    chunks: List[Document] = []
    normalize: bool = True

    def extract_unique_chunks(self) -> List[Document]:
        unique_chunks, unique_sources = set(), []
        for chunk in self.chunks:
            doc_info = chunk.metadata
            if (doc_info["filename"], doc_info["chunk_idx"]) not in unique_chunks:
                unique_sources.append(chunk)
                unique_chunks.add((doc_info["filename"], doc_info["chunk_idx"]))
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


def reciprocal_rank_fusion(list_of_search_results: List[SearchResult]) -> SearchResult:
    rank_dict, doc_dict = {}, {}
    for idx, results in enumerate(list_of_search_results):
        for rank, doc in enumerate(results.chunks):
            doc_id = doc.metadata["filename"] + "_" + str(doc.metadata["chunk_idx"])
            if doc_id not in rank_dict:
                doc_dict[doc_id] = doc
                rank_dict[doc_id] = 0
            rank_dict[doc_id] += (1 / (rank + 1))  # reciprocal ranking
    ranked_id = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
    return SearchResult(chunks=[doc_dict[doc_item[0]] for doc_item in ranked_id])


class FusionStrategyFactory:
    fusion_strategy_mapping = {
        FusionStrategy.rrf.value: reciprocal_rank_fusion
    }

    @classmethod
    def get_fusion_strategy(cls, search_config: SearchConfig) -> Callable:
        return cls.fusion_strategy_mapping[str(search_config.fusion.value)]
