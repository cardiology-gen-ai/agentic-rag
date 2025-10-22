Searchable Vector Stores and their Management
=============================================

This module defines abstract and concrete classes for searchable vector stores
and a high-level search manager that orchestrates query routing, index loading,
and retrieval across different vector store backends
(:langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`
and :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`).

.. automodule:: src.agentic_rag.managers.search_manager
   :members:
   :exclude-members: from_config, model_config
   :member-order: bysource