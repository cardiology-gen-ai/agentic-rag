# Ablation Sintesi

## Cosa confronta

L’ablation valuta tre modi di collegare i termini prodotti dal `KGMentionsPlan` ai nodi `Concept` del knowledge graph:

* `mentions_only`: matching lessicale implicito direttamente nella query Neo4j;
* `mentions_lexical_seeded`: selezione esplicita dei top-m Concept tramite matching lessicale;
* `mentions_embedding_seeded`: selezione dei top-m Concept tramite similarità semantica con `text-embedding-3-small`.

Tutte le modalità:

* usano le stesse 20 domande di `tests/data/CM.json`;
* riutilizzano lo stesso piano di termini per ogni domanda;
* restituiscono nodi `Section`;
* usano `candidate_k=30` e `top_k=10`;
* non attraversano relazioni UMLS, `SAME_AS`, gerarchie o `NEXT`.

Il confronto principale è tra `mentions_lexical_seeded` e `mentions_embedding_seeded`, perché entrambi selezionano lo stesso numero massimo di Concept per termine.

## Risultati

| Metodo                      |  m |    Hit@10 | Recall@10 | Complete Recall@10 |    MRR@10 |
| --------------------------- | -: | --------: | --------: | -----------------: | --------: |
| `mentions_only`             |  — |     0.550 |     0.323 |              0.200 |     0.185 |
| `mentions_lexical_seeded`   |  1 |     0.400 |     0.245 |              0.150 |     0.079 |
| `mentions_embedding_seeded` |  1 |     0.400 |     0.263 |              0.200 |     0.150 |
| `mentions_lexical_seeded`   |  3 |     0.450 |     0.258 |              0.150 |     0.172 |
| `mentions_embedding_seeded` |  3 |     0.500 |     0.288 |              0.200 |     0.218 |
| `mentions_lexical_seeded`   |  5 |     0.500 |     0.328 |              0.200 |     0.178 |
| `mentions_embedding_seeded` |  5 | **0.650** | **0.372** |          **0.250** | **0.250** |

I valori di `m=1` e `m=5` sono ricavati dai log per-query arrotondati; per i valori definitivi bisogna usare `aggregate_metrics.csv`.


## Esempio di esecuzione

```bash
cd /home/marta/projects/CardioAI/agentic-rag
export PYTHONPATH=src

python scripts/evaluate_kg_retrieval.py \
  --env-file .env \
  --dataset tests/data/CM.json \
  --coverage-artifact \
    artifacts/kg_retrieval/kg_gold_validation_20260709T213637Z/gold_resolution_enriched.json \
  --mentions-plans-file \
    artifacts/kg_retrieval/cm_seed_ablation_full_openai_m3/mentions_plans.jsonl \
  --model gpt-4.1-mini \
  --mode mentions_lexical_seeded \
  --mode mentions_embedding_seeded \
  --candidate-k 30 \
  --top-k 10 \
  --concepts-per-term 5 \
  --concept-embedding-model text-embedding-3-small \
  --concept-embedding-cache \
    artifacts/kg_retrieval/openai_concept_embedding_cache \
  --run-id cm_seed_ablation_full_openai_m5 \
  --fail-fast
```

## Parametri

* `--mentions-plans-file`: riutilizza esattamente gli stessi termini tra le run, evitando variabilità dovuta al router.
* `--model gpt-4.1-mini`: modello usato per generare i piani quando questi non vengono caricati da file.
* `--concepts-per-term`: numero massimo di Concept selezionati per ogni termine.
* `--candidate-k 30`: numero massimo di sezioni candidate mantenute internamente.
* `--top-k 10`: numero di sezioni restituite e valutate.
* `--concept-embedding-model`: modello OpenAI usato per la similarità termine–Concept.
* `--concept-embedding-cache`: evita di ricalcolare gli embedding dei Concept.
* `--run-id`: nome della cartella degli artifact.
* `--fail-fast`: interrompe la run in presenza di un errore.

Gli artifact vengono salvati in:

```text
artifacts/kg_retrieval/<run-id>/
```

I file principali sono:

* `aggregate_metrics.csv`;
* `per_query_metrics.csv`;
* `queries.jsonl`;
* `mentions_plans.jsonl`;
* `concept_seed_diagnostics.json`;
* `manifest.json`;
* `summary.json`.
