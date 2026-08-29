from __future__ import annotations

import json

import numpy as np

from agentic_rag.kg.frozen_embeddings import (
    load_concept_embedding_map,
    load_query_term_embedding_map,
)


def test_concept_cache_preserves_exact_case_identities(tmp_path):
    path = tmp_path / "concepts.npz"
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metadata = {
        "concept_names": ["DMD", "dmd"],
        "catalogue_hash": "x",
    }
    np.savez_compressed(
        path,
        embeddings=matrix,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )

    concepts, _ = load_concept_embedding_map(path)

    assert set(concepts) == {"DMD", "dmd"}
    assert np.array_equal(concepts["DMD"], matrix[0])
    assert np.array_equal(concepts["dmd"], matrix[1])


def test_query_term_cache_is_normalized_on_read(tmp_path):
    metadata = {
        "schema_version": "query_term_embedding_cache_v1",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": None,
        "term": "RASopathy",
    }
    np.savez_compressed(
        tmp_path / "term.npz",
        embedding=np.asarray([[3.0, 4.0]], dtype=np.float32),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )

    terms = load_query_term_embedding_map(
        tmp_path,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=None,
    )

    assert np.allclose(terms["rasopathy"], np.asarray([0.6, 0.8], dtype=np.float32))
