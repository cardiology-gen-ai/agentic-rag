# Modular KG retrieval baselines

Modes:

- `mentions_only`: MENTIONS candidate generation, `concept_match`, no expansion,
  no reranking.
- `mentions_weighted`: same MENTIONS candidates with the existing heuristic
  lexical weights and title bonus.
- `mentions_descendants`: pure MENTIONS seeds, expansion through `HAS_CHILD`,
  deterministic seed-by-seed ordering.

The retrieval unit is always a Neo4j `Section` node. Both high-level sections
and nested subsections are eligible (`unit_scope=all_levels`). `NEXT` is not
used because it represents reading order rather than hierarchy.

The existing `KGParameterizedRetriever` and `kg_retrieval_router.yaml` remain
unchanged and available as the advanced `planned_role_aware` configuration.
