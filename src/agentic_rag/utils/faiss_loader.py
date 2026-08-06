from __future__ import annotations

import math
from typing import List

import faiss
from cardiology_gen_ai import DistanceTypeNames, EmbeddingConfig, IndexingConfig
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings


class L2NormalizedEmbeddings(Embeddings):
    """Normalize document and query embeddings for cosine FAISS search."""

    def __init__(self, base: Embeddings):
        self.base = base

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError("Cannot L2-normalize a zero embedding vector")
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            self._normalize(vector)
            for vector in self.base.embed_documents(texts)
        ]

    def embed_query(self, text: str) -> List[float]:
        return self._normalize(self.base.embed_query(text))


def load_faiss_vectorstore(
    config: IndexingConfig,
    embeddings_model: EmbeddingConfig,
) -> FAISS:
    """Load FAISS with the same metric semantics used by ``data-etl``.

    Cosine indexes are persisted as inner-product indexes whose document
    embeddings are already L2-normalized. The query must therefore be
    normalized as well and LangChain must be told that larger inner-product
    scores are better. Euclidean indexes keep the original embedding model and
    Euclidean distance strategy.
    """

    if config.distance == DistanceTypeNames.cosine:
        embedding_function: Embeddings = L2NormalizedEmbeddings(
            embeddings_model.model
        )
        distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        expected_metric_type = faiss.METRIC_INNER_PRODUCT
    elif config.distance == DistanceTypeNames.euclidean:
        embedding_function = embeddings_model.model
        distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
        expected_metric_type = faiss.METRIC_L2
    else:
        raise ValueError(
            "FAISS loading requires distance='cosine' or distance='euclidean'; "
            f"received {config.distance!r}"
        )

    vectorstore = FAISS.load_local(
        folder_path=config.folder.as_posix(),
        index_name=config.name,
        embeddings=embedding_function,
        allow_dangerous_deserialization=True,
        normalize_L2=False,
        distance_strategy=distance_strategy,
    )

    actual_metric_type = getattr(vectorstore.index, "metric_type", None)
    if actual_metric_type != expected_metric_type:
        raise ValueError(
            "FAISS metric/config mismatch: "
            f"index={config.name!r}, distance={config.distance.value!r}, "
            f"expected_metric_type={expected_metric_type!r}, "
            f"actual_metric_type={actual_metric_type!r}"
        )

    return vectorstore
