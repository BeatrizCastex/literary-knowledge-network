# Pipeline Summary

This project automates the creation of a Goodreads-derived literature graph in Neo4j.  
The workflow is split across seven scripted steps; each step reads the previous step’s output and produces enriched artefacts for the next stage.

## Processing Steps

| Step | Script | Purpose | Key Outputs |
|------|--------|---------|-------------|
| 1 | `data-extraction-v3.py` | Randomly samples works from the raw Goodreads dumps, gathers their books, associated authors, series, and crafts initial DataFrames. | `data/processed/extraction/*.parquet` |
| 2 | `data-normalization.py` | Cleans and deduplicates publishers, tags, and entity keys. Supports auto-merge or interactive review. | `data/processed/normalized/*.parquet` |
| 3 | `enritch-data.py` | Enriches normalized data (Wikidata lookups, inferred countries, keyword extraction, citations). | `data/processed/enriched/*.parquet` |
| 4 | `data-injection.py` | Streams enriched tables into Neo4j, MERGE-ing nodes/relationships and applying tag filters. | Neo4j database populated with core graph |
| 5 | `embedding.py` | Generates work embeddings (SBERT + optional TF-IDF). | `data/output/embeddings/work_embeddings_sbert.parquet` |
| 6 | `similarity.py` | Computes cosine similarity between works, writes weighted `SIMILAR_TO` edges, exports similarity parquet. | `data/output/similarity/work_similarity.parquet`, `data/output/edges/*.csv` + Neo4j edges |
| 7 | `community_detection.py` | Runs Leiden community detection over the similarity graph, writes cluster metadata back to Neo4j. | `data/output/communities/work_communities.parquet`, `data/output/communities/work_communities.csv`, `data/output/results_metrics.json` + `Work.cluster_*` properties |

A convenience runner, `run_pipeline.sh`, executes all seven steps from a single command (`./run_pipeline.sh <sample_size> [random_seed]`).

Need to grow an existing cohort? `data-extraction-v3.py` can now top up to a target post-filter size using `--existing-work-file`, `--target-after-filters`, and `--incremental-only`, exporting the incremental sample via `--incremental-export-dir` for a focused normalization/enrichment pass.

## Tooling & Libraries

- **Python 3.10+**
- Core data stack: `pandas`, `numpy`, `scikit-learn`, `scipy`
- Graph integrations: `neo4j` Python driver, `python-igraph`, `leidenalg`
- NLP / embeddings: `sentence-transformers`, `huggingface-hub`, `tokenizers`
- Utility packages: `tqdm`, `pyarrow` (optional for Parquet), `requests`
- External services: a running **Neo4j** database (defaults to `bolt://127.0.0.1:7687`, user `neo4j`/`seniorthesis`); optional Wikidata API access for enrichment

Ensure the venv contains the packages above before running the full pipeline.

## Environment Notes

- Raw Goodreads dumps live under `data/raw/` (filenames configured in `config.yaml` under `raw_files`).
- Neo4j credentials can be overridden via the environment variables `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.
- The embedding and community detection steps require network access to download pretrained models the first time they run.
- `run_pipeline.sh` assumes the scripts live alongside it. Paths resolve via `config.yaml`, so the defaults target `data/processed/` and `data/output/`.

For reproducibility, record the sample size and random seed used when running Step 1 (both accepted by the extraction script and the shell pipeline).
