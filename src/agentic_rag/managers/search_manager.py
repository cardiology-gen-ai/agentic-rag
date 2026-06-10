import json
import logging
import os
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents.base import Document
from qdrant_client.http import models

from agentic_rag.config.manager import FusionStrategy
from agentic_rag.managers.llm_manager import LLMManager
from agentic_rag.utils.search import build_cross_encoder
from agentic_rag.utils.search import FusionStrategyFactory, SearchResult
from agentic_rag.config.manager import SearchConfig
from cardiology_gen_ai import IndexingConfig, IndexTypeNames, Vectorstore, QdrantVectorstore, FaissVectorstore, \
    BM25Vectorstore, BM25Dict
from cardiology_gen_ai.utils.logger import get_logger


class SearchableVectorstore(Vectorstore, ABC):
    """Abstract class that adds search utilities to a :class:`cardiology_gen_ai.models.Vectorstore`."""
    search_config: SearchConfig #: :class:`~src.agentic_rag.config.manager.SearchConfig` : Configuration controlling the retrieval strategy (search type, kwargs, metadata filters, hybrid fusion).
    retriever: Optional[VectorStoreRetriever | Dict] = None #: :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>` : Retriever used by :meth:`~src.agentic_rag.managers.search_manager.SearchableVectorstore.search`.

    @abstractmethod
    def get_retriever(self, **kwargs) -> VectorStoreRetriever:
        """Instantiate and return the underlying retriever.

        Returns
        -------
        :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>`
            The configured retriever instance.
        """
        pass

    def search(self, query: str) -> List[Document]:
        """Execute a single retrieval over the vector store.

        Parameters
        ----------
        query : :class:`str`
            Free-text query.

        Returns
        -------
        list of :langchain:`Document <core/documents/langchain_core.documents.base.Document.html>`
            Retrieved documents.
        """
        return self.retriever.invoke(query)


class SearchableQdrantVectorstore(SearchableVectorstore, QdrantVectorstore):
    """:langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`-backed searchable vector store.

    Adds optional metadata filtering via :qdrant_client:`Qdrant Filtering <filtering>` and hybrid
    fusion (RRF) when enabled in :class:`~src.agentic_rag.config.manager.SearchConfig`.
    """

    def get_retriever(self) -> VectorStoreRetriever:
        """Build a :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>` for :langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`.

        When :attr:`~src.agentic_rag.config.manager.SearchConfig.metadata_filter` is provided, this method converts it into a :qdrant_client:`Filter <filtering/?q=Filter>`
        (with a :qdrant_client:`FieldCondition <filtering/?q=FieldCondition>` / :qdrant_client:`MatchValue <filtering/?q=MatchValue>`)
        If :attr:`~src.agentic_rag.config.manager.SearchConfig.fusion` is enabled, it sets ``hybrid_fusion`` to :qdrant_client:`FusionQuery <hybrid-queries>` with RRF.

        Returns
        -------
        :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>`
            The configured retriever.
        """
        if self.search_config.metadata_filter is not None:
            metadata_filter = self.search_config.metadata_filter
            # TODO: refine when needed [possibly the one in FAISS should work]
            metadata_search_filter = models.Filter(
                must=[models.FieldCondition(
                    key="metadata.filename", match=models.MatchValue(value=metadata_filter.get("filename")))]
            )
            self.serach_config.kwargs["filter"] = metadata_search_filter
        if self.search_config.fusion:
            self.serach_config.kwargs["hybrid_fusion"] = models.FusionQuery(fusion=models.Fusion.RRF)
        self.retriever = self.vectorstore.as_retriever(
            search_type=self.search_config.type.value,
            search_kwargs=self.search_config.kwargs,
        )
        return self.retriever


class SearchableFaissVectorstore(SearchableVectorstore, FaissVectorstore):
    """:langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`-backed searchable vector store.

    Adds optional metadata filtering by turning :attr:`~src.agentic_rag.config.manager.SearchConfig.metadata_filter`
    into a dictionary-based filter understood by the :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>` retriever wrapper.
    """

    def get_retriever(self, **kwargs) -> VectorStoreRetriever:
        """Build a :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>` for :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`.

        Returns
        -------
        :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>`
            The configured retriever.
        """
        if self.search_config.metadata_filter is not None:
            metadata_filter = self.search_config.metadata_filter
            # TODO: refine when needed
            metadata_search_filter = {
                "filename": {"$in": metadata_filter.get("filename")},  # "$eq"
            }
            self.serach_config.kwargs["filter"] = metadata_search_filter
        self.retriever = self.vectorstore.as_retriever(
            search_type=self.search_config.type.value,
            search_kwargs=self.search_config.kwargs,
        )
        return self.retriever


class SearchableBM25Vectorstore(SearchableVectorstore, BM25Vectorstore):

    def get_retriever(self, **kwargs) -> BM25Dict:
        self.retriever = self.vectorstore
        return self.retriever

    def search(self, query: str) -> List[Document]:
        scores = self.vectorstore.bm25.get_scores(BM25Vectorstore.tokenize(query))
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.search_config.kwargs["k"]]
        return [self.vectorstore.documents[i] for i in top_k]


class SearchableVectorstoreFactory:
    searchable_vectorstore_mapping = {
        IndexTypeNames.qdrant: SearchableQdrantVectorstore,
        IndexTypeNames.faiss: SearchableFaissVectorstore,
        IndexTypeNames.bm25: SearchableBM25Vectorstore,
    }

    @classmethod
    def get_searchable_vectorstore(cls, index_config: IndexingConfig, search_config: SearchConfig) -> SearchableVectorstore:
        return cls.searchable_vectorstore_mapping[IndexTypeNames(index_config.type)](config=index_config, search_config=search_config)


class SearchManager: # (metaclass=Singleton):
    """High-level orchestrator that selects and instantiates a searchable vector store.

    The manager chooses between a :langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`-backed or :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>-backed implementation
    based on :class:`cardiology_gen_ai.models.IndexingConfig`, ensures the index is loaded, creates the retriever, and exposes a single ``search()`` entry point.

    Parameters
    ----------
    index_config : :class:`cardiology_gen_ai.models.IndexingConfig`
        Configuration for index name, type, and retrieval mode.
    search_config : :class:`~src.agentic_rag.config.manager.SearchConfig`
        Configuration for search type, kwargs, filters, and hybrid fusion.
    """
    logger: logging.Logger #: :class:`logging.Logger` : Logger for lifecycle and diagnostics.
    index_config: List[IndexingConfig] #: :class:`cardiology_gen_ai.models.IndexingConfig` : Indexing configuration.
    search_config: SearchConfig #: :class:`~src.agentic_rag.config.manager.SearchConfig` : Search configuration.
    vectorstores: SearchableVectorstore #: :class:`SearchableVectorstore` : Concrete searchable vector store.
    def __init__(self, index_config: IndexingConfig | List[IndexingConfig], search_config: SearchConfig):
        self.logger = get_logger("Searching based on LangChain VectorStores")
        self.index_config = [index_config] if isinstance(index_config, IndexingConfig) else index_config
        self.search_config = search_config
        self.vectorstores: List[SearchableVectorstore] = self._init_vectorstores()
        self.make_searchable()
        self.reranker = LLMManager(config=self.search_config.reranker) if (self.search_config.fusion.value == FusionStrategy.reranking.value and self.search_config.reranker is not None) else None
        self.cross_encoder = build_cross_encoder(self.search_config.cross_encoder) if (self.search_config.fusion.value == FusionStrategy.cross_encoder.value and self.search_config.cross_encoder is not None) else None

    def _init_vectorstores(self) -> List[SearchableVectorstore]:
        vectorstores = []
        for index_config in self.index_config:
            current_vectorstore = SearchableVectorstoreFactory.get_searchable_vectorstore(
                index_config=index_config, search_config=self.search_config,
            )
            vectorstores.append(current_vectorstore)
        return vectorstores

    def get_n_documents_in_vectorstores(self) -> List[int]:
        """Return the number of documents currently stored in the vector index.

        Returns
        -------
        int
            Document count (as reported by the underlying vector store).
        """
        n_docs_in_vectorstores = []
        for vectorstore in self.vectorstores:
            n_docs_in_vectorstores.append(vectorstore.get_n_documents_in_vectorstore())
        return n_docs_in_vectorstores

    def load_indexes(self):
        """Load an existing index from disk/remote using the configured embeddings.

        .. rubric:: Notes

        If the index does not exist yet, the method logs and returns without error.
        """
        for idx, index_config in zip(range(len(self.vectorstores)), self.index_config):
            if not self.vectorstores[idx].vectorstore_exists():
                self.logger.info(f"Index {index_config.name} (of type {index_config.type.value}) "
                                 f"does not exist yet. Will be created when documents are added.")
                return
            try:
                self.vectorstores[idx].load_vectorstore(
                    embeddings_model=index_config.embeddings, retrieval_mode=index_config.retrieval_mode.value
                )
                self.logger.info(f"Index {index_config.name} (of type {index_config.type.value}) loaded successfully.")
            except Exception as e:
                self.logger.info(f"Error loading index {index_config.name} (of type {index_config.type.value}): {str(e)}")
                raise

    def get_retrievers(self):
        """Instantiate the retriever on the current vector store.

        .. rubric:: Notes

        If the index does not exist yet, logs an informational message and returns.
        """
        for idx, vectorstore in enumerate(self.vectorstores):
            if not self.vectorstores[idx].vectorstore_exists():
                self.logger.info("No vectorstore available yet. Retriever will be created when documents are added.")
                return
            try:
                self.vectorstores[idx].get_retriever()
                self.logger.info("Retriever instantiated successfully.")
            except Exception as e:
                self.logger.info(f"Error getting retriever: {str(e)}")
                raise

    def make_searchable(self):
        """Ensure the index is loaded and the retriever is ready."""
        self.load_indexes()
        self.get_retrievers()

    def get_from_vectorstore(self, query: str) -> List[SearchResult]:
        if len(self.vectorstores) == 1:
            self.logger.info("Retrieving documents from a single vectorstore..")
            results = self.vectorstores[0].search(query)
            results = SearchResult(chunks=results)
            return [results]
        else:
            assert len(self.vectorstores) > 1
            self.logger.info(f"Retrieving documents from {len(self.vectorstores)} vectorstores..")
            return [SearchResult(chunks=vectorstore.search(query)) for vectorstore in self.vectorstores]

    def rerank(self, query: str, list_of_search_results: List[SearchResult]) -> SearchResult:
        assert self.search_config.fusion is not None
        self.logger.info(f"Fusing search results with {self.search_config.fusion.value} function..")
        kwargs: Dict[str, Any] = {"query": query}
        if self.search_config.fusion.value == FusionStrategy.reranking.value:
            assert self.reranker is not None
            kwargs["reranker"] = self.reranker.llm
        elif self.search_config.fusion.value == FusionStrategy.cross_encoder.value:
            assert self.cross_encoder is not None
            kwargs["cross_encoder"] = self.cross_encoder
        elif self.search_config.fusion.value == FusionStrategy.rrf.value and len(self.vectorstores) <= 1:
            self.logger.warning("RRF makes sense only in presence on multiple vectorstores!")
        fusion_fn = FusionStrategyFactory.get_fusion_strategy(self.search_config)
        fused_results: SearchResult = fusion_fn(list_of_search_results, top_k=self.search_config.top_k, **kwargs)
        return fused_results

    def search(self, query: str) -> SearchResult:
        """Run a search query against the selected vector store.

        Parameters
        ----------
        query : :class:`str`
            Free-text query.

        Returns
        -------
        list of :langchain:`Document <core/documents/langchain_core.documents.base.Document.html>`
            Retrieved documents.
        """
        list_of_search_results = self.get_from_vectorstore(query)
        if self.search_config.fusion is not None:
            return self.rerank(query, list_of_search_results)
        return list_of_search_results[0] if len(self.vectorstores) == 1 \
            else SearchResult(chunks=list(chain.from_iterable(
            [[result for result in list_of_results.chunks] for list_of_results in list_of_search_results])
        ))


if __name__ == "__main__":
    print(os.getcwd())
    with open(Path("tests/data") / "synthetic_data.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        print(item["question"])
    # config = AgentConfigManager().config
    # search_manager = SearchManager(
    #     index_config=config.indexing,
    #     search_config=config.search,
    #     embeddings=config.embeddings
    # )
    # print(type(search_manager.vectorstore.retriever))
