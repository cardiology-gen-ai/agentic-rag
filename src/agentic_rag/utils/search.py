from itertools import chain
from typing import List, Callable, Dict, Set, Tuple

import torch
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank, CrossEncoderReranker
from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from sentence_transformers.cross_encoder import CrossEncoder

from agentic_rag.agent.prompts.name_title_mapping import filename_title_mapping
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

    def normalize_headers(self, headers: Dict[str, str | List]) -> Dict[str, str | List] | List[str]:
        header_values = list(headers.values())
        if isinstance(header_values[0], list):
            header_values = list(chain.from_iterable(header_values))
        return header_values if self.normalize else headers

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

    def to_sources_payload(self) -> List[Dict]:
        sources = []
        unique = self.extract_unique_chunks()
        for document in unique:
            sources.append({
                "filename": document.metadata.get("filename", "unknown"),
                "chunk_idx": document.metadata.get("chunk_idx", "unknown"),
                "headers": document.metadata.get("headers", []),
            })
        return sources

    def format_sources(self) -> str:
        # unique = self.extract_unique_chunks()
        unique = self.group_by_filename()
        if not unique:
            return ""
        lines = []
        for idx, (doc_name, doc_headers) in enumerate(unique.items(), start=1):
            formatted_filename = doc_name.split(".")[0]
            file_title = filename_title_mapping.get(formatted_filename, formatted_filename)
            # headers = doc.metadata.get("headers", [])
            # flat_headers = [item for sublist in headers.values() for item in sublist] if headers else []
            # header_str = "; ".join(list(doc_headers)) if doc_headers else "N/A"
            lines.append(f"{idx}. {file_title}")
            for sec in doc_headers:
                lines.append(f"   - {sec};")
        return "\n".join(lines)


def reciprocal_rank_fusion(list_of_search_results: List[SearchResult], top_k: int, **kwargs) -> SearchResult:
    rank_dict, doc_dict = {}, {}
    for idx, results in enumerate(list_of_search_results):
        for rank, doc in enumerate(results.chunks):
            doc_id = doc.metadata["filename"] + "_" + str(doc.metadata["chunk_idx"])
            if doc_id not in rank_dict:
                doc_dict[doc_id] = doc
                rank_dict[doc_id] = 0
            rank_dict[doc_id] += (1 / (rank + 1))  # reciprocal ranking
    ranked_id = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
    return SearchResult(chunks=[doc_dict[doc_item[0]] for doc_item in ranked_id[:top_k]])


def rerank_fusion(list_of_search_results: List[SearchResult], top_k: int, reranker: BaseChatModel, query: str) -> SearchResult:
    listwise_reranker = LLMListwiseRerank.from_llm(llm=reranker, prompt=None, top_n=top_k)
    all_search_results = list(chain.from_iterable([[chunk for chunk in results.chunks] for results in list_of_search_results]))
    sorted_results = listwise_reranker.compress_documents(documents=all_search_results, query=query)
    return SearchResult(chunks=[doc for doc in sorted_results])


class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    def __init__(self, cross_encoder_name: str):
        self.cross_encoder_model = CrossEncoder(cross_encoder_name, trust_remote_code=True, device="cuda")
        self.cross_encoder_model.model = self.cross_encoder_model.model.to(torch.bfloat16)
        tokenizer = self.cross_encoder_model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.cross_encoder_model.model.config.pad_token_id = \
            self.cross_encoder_model.tokenizer.pad_token_id
        self.cross_encoder_model.model.eval()

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        batch_size = 8
        scores = []
        for i in range(0, len(text_pairs), batch_size):
            batch = text_pairs[i:i + batch_size]
            scores.extend(self.cross_encoder_model.predict(batch).tolist())
        return scores


def build_cross_encoder(cross_encoder_name: str) -> BaseCrossEncoder:
    return SentenceTransformerCrossEncoder(cross_encoder_name=cross_encoder_name)


def cross_encoder_fusion(list_of_search_results: List[SearchResult], top_k: int, cross_encoder: BaseCrossEncoder, query: str) -> SearchResult:
    cross_encoder_reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
    docs: List[Document] = list(chain.from_iterable([[result for result in list_of_results.chunks] for list_of_results in list_of_search_results]))
    sorted_results = cross_encoder_reranker.compress_documents(documents=docs, query=query)
    return SearchResult(chunks=[doc for doc in sorted_results])


def bm25_fusion(list_of_search_results: List[SearchResult], top_k: int, query: str) -> SearchResult:
    docs: List[Document] = list(chain.from_iterable([[
        result for result in list_of_results.chunks] for list_of_results in list_of_search_results])
    )
    retriever = BM25Retriever.from_documents(documents=docs, k=top_k)
    return SearchResult(chunks=retriever.invoke(query))


class FusionStrategyFactory:
    fusion_strategy_mapping = {
        FusionStrategy.rrf.value: reciprocal_rank_fusion,
        FusionStrategy.reranking.value: rerank_fusion,
        FusionStrategy.cross_encoder.value: cross_encoder_fusion,
        FusionStrategy.bm25.value: bm25_fusion,
    }

    @classmethod
    def get_fusion_strategy(cls, search_config: SearchConfig) -> Callable:
        assert search_config.fusion is not None
        return cls.fusion_strategy_mapping[str(search_config.fusion.value)]
