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
    search_config: SearchConfig
    retriever: VectorStoreRetriever = None

    @abstractmethod
    def get_retriever(self, **kwargs) -> VectorStoreRetriever:
        pass

    def search(self, query: str) -> List[Document]:
        return self.retriver.invoke(query)


class SearchableQdrantVectorstore(SearchableVectorstore, QdrantVectorstore):

    def get_retriever(self) -> VectorStoreRetriever:
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

    def get_retriever(self, **kwargs) -> VectorStoreRetriever:
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
        return self.vectorstore.get_n_documents_in_vectorstore()

    def load_index(self):
        try:
            self.vectorstore.load_vectorstore(embeddings_model=self.embeddings,
                                              retrieval_mode=self.index_config.retrieval_mode.value)
            self.logger.info(f"Index {self.index_config.name} loaded successfully.")
        except Exception as e:
            self.logger.info(f"Error loading {self.index_config.name} index: {str(e)}")
            raise

    def get_retriever(self):
        try:
            self.vectorstore.get_retriever()
            self.logger.info("Retriever instantiated successfully.")
        except Exception as e:
            self.logger.info(f"Error getting retriever: {str(e)}")
            raise

    def make_searchable(self):
        self.load_index()
        self.get_retriever()

    def search(self, query: str) -> List[Document]:
        return self.vectorstore.search(query)


if __name__ == "__main__":
    config = AgentConfigManager().config
    search_manager = SearchManager(
        index_config=config.indexing,
        search_config=config.search,
        embeddings=config.embeddings
    )
    print(type(search_manager.vectorstore.retriever))
