import logging
from abc import ABC, abstractmethod
from typing import List

from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents.base import Document
from qdrant_client.http import models

from src.config.manager import SearchConfig, AgentConfigManager

from cardiology_gen_ai.utils.singleton import Singleton
from cardiology_gen_ai import IndexingConfig, IndexTypeNames, Vectorstore, QdrantVectorstore, FaissVectorstore, \
    EmbeddingConfig
from cardiology_gen_ai.utils.logger import get_logger


class SearchableVectorstore(Vectorstore, ABC):
    """Abstract class that adds search utilities to a :class:`cardiology_gen_ai.models.Vectorstore`."""
    search_config: SearchConfig #: :class:`~src.config.manager.SearchConfig` : Configuration controlling the retrieval strategy (search type, kwargs, metadata filters, hybrid fusion).
    retriever: VectorStoreRetriever = None #: :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>` : Retriever used by :meth:`~src.managers.search_manager.SearchableVectorstore.search`.

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
        return self.retriver.invoke(query)


class SearchableQdrantVectorstore(SearchableVectorstore, QdrantVectorstore):
    """:langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`-backed searchable vector store.

    Adds optional metadata filtering via :mod:`qdrant_client.models` and hybrid
    fusion (RRF) when enabled in :class:`~src.config.manager.SearchConfig`.
    """

    def get_retriever(self) -> VectorStoreRetriever:
        """Build a :langchain_core:`VectorStoreRetriever <vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html>` for :langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`.

        When :attr:`~src.config.manager.SearchConfig.metadata_filter` is provided, this method converts it into a :class:`~qdrant_client.models.Filter`
        (with a :class:`~qdrant_client.models.FieldCondition` / :class:`~qdrant_client.models.MatchValue`)
        If :attr:`~src.config.manager.SearchConfig.fusion` is enabled, it sets ``hybrid_fusion`` to :class:`~qdrant_client.models.FusionQuery` with RRF.

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

    Adds optional metadata filtering by turning :attr:`~src.config.manager.SearchConfig.metadata_filter`
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


class SearchManager(metaclass=Singleton):
    """High-level orchestrator that selects and instantiates a searchable vector store.

    The manager chooses between a :langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`-backed or :class:`FAISS`-backed implementation
    based on :class:`cardiology_gen_ai.models.IndexingConfig`, ensures the index is loaded, creates the retriever, and exposes a single ``search()`` entry point.

    Parameters
    ----------
    index_config : :class:`cardiology_gen_ai.models.IndexingConfig`
        Configuration for index name, type, and retrieval mode.
    search_config : :class:`~src.config.manager.SearchConfig`
        Configuration for search type, kwargs, filters, and hybrid fusion.
    embeddings : :class:`cardiology_gen_ai.models.EmbeddingConfig`
        Embedding model/configuration used when loading the index.
    """
    logger: logging.Logger #: :class:`logging.Logger` : Logger for lifecycle and diagnostics.
    indexing_config: IndexingConfig #: :class:`cardiology_gen_ai.models.IndexingConfig` : Indexing configuration.
    search_config: SearchConfig #: :class:`~src.config.manager.SearchConfig` : Search configuration.
    embeddings: EmbeddingConfig #: :class:`cardiology_gen_ai.models.EmbeddingConfig` : Embedding configuration.
    vectorstore: SearchableVectorstore #: :class:`SearchableVectorstore` : Concrete searchable vector store.
    def __init__(self, index_config: IndexingConfig, search_config: SearchConfig, embeddings: EmbeddingConfig):
        self.logger = get_logger("Searching based on LangChain VectorStores")
        self.index_config = index_config
        self.search_config = search_config
        self.embeddings = embeddings
        self.vectorstore: SearchableVectorstore = (
            SearchableQdrantVectorstore(config=self.index_config, search_config=self.search_config)) \
            if IndexTypeNames(self.index_config.type) == IndexTypeNames.qdrant \
            else SearchableFaissVectorstore(config=self.index_config, search_config=self.search_config)
        self.make_searchable()

    def get_n_documents_in_vectorstore(self):
        """Return the number of documents currently stored in the vector index.

        Returns
        -------
        int
            Document count (as reported by the underlying vector store).
        """
        return self.vectorstore.get_n_documents_in_vectorstore()

    def load_index(self):
        """Load an existing index from disk/remote using the configured embeddings.

        .. rubric:: Notes

        If the index does not exist yet, the method logs and returns without error.
        """
        if not self.vectorstore.vectorstore_exists():
            self.logger.info(f"Index {self.index_config.name} does not exist yet. Will be created when documents are added.")
            return
        try:
            self.vectorstore.load_vectorstore(embeddings_model=self.embeddings,
                                              retrieval_mode=self.index_config.retrieval_mode.value)
            self.logger.info(f"Index {self.index_config.name} loaded successfully.")
        except Exception as e:
            self.logger.info(f"Error loading {self.index_config.name} index: {str(e)}")
            raise

    def get_retriever(self):
        """Instantiate the retriever on the current vector store.

        .. rubric:: Notes

        If the index does not exist yet, logs an informational message and returns.
        """
        if not self.vectorstore.vectorstore_exists():
            self.logger.info("No vectorstore available yet. Retriever will be created when documents are added.")
            return
        try:
            self.vectorstore.get_retriever()
            self.logger.info("Retriever instantiated successfully.")
        except Exception as e:
            self.logger.info(f"Error getting retriever: {str(e)}")
            raise

    def make_searchable(self):
        """Ensure the index is loaded and the retriever is ready."""
        self.load_index()
        self.get_retriever()

    def search(self, query: str) -> List[Document]:
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
        return self.vectorstore.search(query)


if __name__ == "__main__":
    config = AgentConfigManager().config
    search_manager = SearchManager(
        index_config=config.indexing,
        search_config=config.search,
        embeddings=config.embeddings
    )
    print(type(search_manager.vectorstore.retriever))
