# Literary Knowledge Network

Computational pipeline that extracts, normalizes, enriches, and analyzes Goodreads metadata to build a literary knowledge graph and similarity network. The project powers community detection and recommendation workflows backed by Neo4j and modern embedding models.

## Repository Layout

```
literary-knowledge-network/
├── data/
│   ├── raw/               # Source dumps (ignored by git; add your own)
│   ├── processed/         # Intermediate parquet/csv exports
│   └── output/            # Final embeddings, similarity edges, communities, metrics
├── docs/                  # Pipeline summary, data dictionary, reproducibility, thesis notes
├── logs/                  # Runtime logs and pipeline summaries
├── notebooks/             # Exploratory notebooks (ignored by git)
├── scripts/               # End-to-end pipeline scripts (Steps 1-7)
├── .env.example           # Neo4j credential template
├── .gitignore             # Ignore rules for data, envs, caches
├── CITATION.cff           # Citation metadata for the thesis project
├── config.yaml            # Central configuration (paths, Neo4j, defaults)
├── environment.yml        # Optional Conda environment specification
├── LICENSE                # MIT License
├── README.md              # This document
└── requirements.txt       # Python dependencies
```

## Getting Started

1. **Clone the repository** and create a Python 3.10+ environment.
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   or with Conda:
   ```bash
   conda env create -f environment.yml
   conda activate literary-knowledge-network
   ```
3. **Configure credentials**
   - Copy `.env.example` to `.env` and populate Neo4j URI, user, and password.
   - Export environment variables or load them via your shell/runner.
4. **Place raw Goodreads dumps**
   - Drop the gzipped JSON files inside `data/raw/`.
   - The filenames should match those referenced in `config.yaml`.

### Running the full 10k workflow

The thesis release is based on a run that targets **10,000 works**. Once Neo4j is populated and the raw dumps are available locally, execute:

```bash
cd scripts
./run_pipeline.sh 10000 42 <neo4j_password>
```

- The optional random seed (`42` above) guarantees reproducible sampling.
- Expect several hours of runtime for the enrichment step at this scale; watch `logs/pipeline_runs/` for progress.
- Final similarities and community assignments land in `data/output/similarity/` and `data/output/communities/` (these directories are git-ignored by default—export the artefacts you want to publish separately).

### Topping up an existing dataset

If you already extracted and enriched a smaller cohort (e.g., ~3.5k works) and want to reach a 10k-node dataset without rerunning everything:

```bash
cd scripts

# Sample additional works (excludes IDs in the existing enriched file)
python3 data-extraction-v3.py \
  --incremental-only \
  --existing-work-file data/processed/enriched/work_final.parquet \
  --target-after-filters 10000 \
  --incremental-export-dir data/processed/extraction_incremental \
  --random-seed 42

# Normalize and enrich only the new batch
python3 data-normalization.py \
  --input-dir data/processed/extraction_incremental \
  --output-dir data/processed/normalized_incremental \
  --export \
  --auto-merge

python3 enritch-data.py \
  --input-dir data/processed/normalized_incremental \
  --output-dir data/processed/enriched_incremental \
  --keyword-method hybrid \
  --no-wikidata \
  --export
```

You can then merge the incremental enriched tables with the original ones (e.g., via pandas concat) before running Steps 4–7 on the combined dataset. The Neo4j loader uses `MERGE`, so injecting the combined exports will not duplicate existing works.

## Pipeline Overview

| Step | Script | Purpose | Key Outputs |
|------|--------|---------|-------------|
| 1 | `scripts/data-extraction-v3.py` | Sample works with metadata and build entity tables. | `data/processed/extraction/*.parquet` |
| 2 | `scripts/data-normalization.py` | Harmonize identifiers, deduplicate, and prepare merge suggestions. | `data/processed/normalized/*.parquet` |
| 3 | `scripts/enritch-data.py` | Enrich with Wikidata, keyword extraction, and NER-derived citations. | `data/processed/enriched/*.parquet` |
| 4 | `scripts/data-injection.py` | Load enriched entities into Neo4j (idempotent). | `logs/data_injection_summary.json` |
| 5 | `scripts/embedding.py` | Generate SBERT & TF-IDF embeddings for works. | `data/output/embeddings/` |
| 6 | `scripts/similarity.py` | Build similarity graph, optionally write SIMILAR_TO edges in Neo4j. | `data/output/similarity/work_similarity.parquet`, `data/output/edges/*.csv` |
| 7 | `scripts/community_detection.py` | Detect graph communities and persist to Neo4j / parquet. | `data/output/communities/work_communities.parquet`, `data/output/communities/work_communities.csv` |

Run the entire pipeline with:

```bash
cd scripts
./run_pipeline.sh <sample_size> [random_seed] [neo4j_password]
```

Each script also provides granular CLI flags (see `--help`).

## Runtime & Logging

- `run_pipeline.sh` records wall-clock duration, per-step commands, and a snapshot of the Neo4j node/relationship counts in `logs/pipeline_runs/`.
- Estimate future runtimes with `python3 scripts/estimate_runtime.py --sample-size <n>` (see `docs/reproducibility.md` for advanced options).
- Commit the generated JSON logs whenever you need an auditable record of an experiment or thesis result.
- Step 6 now exports SBERT, TF-IDF, and hybrid edge lists as CSV under `data/output/edges/` for notebook analyses (git-ignored).
- Step 7 aggregates key metrics into `data/output/results_metrics.json` (also mirrored under `data/output/analysis/`) for downstream notebooks.
- A local Wikidata cache is written under `logs/wikidata-cache/` to reduce API calls; the directory is ignored by git and can be wiped if you need to reclaim disk space.

## Configuration

All paths, Neo4j defaults, and algorithm parameters live in `config.yaml`. Every script loads this file to resolve directories, export locations, and connection settings. Adjust the YAML once instead of editing individual scripts.

Key sections:

- `paths`: Base directories for raw, processed, output data, and logs.
- `extraction`, `normalization`, `enrichment`, `embedding`, `similarity`, `communities`: Pipeline defaults (sample size, thresholds, filenames).
- `neo4j`: Connection URI and environment variable keys.

## Documentation

- `docs/pipeline_summary.md` - Stage-by-stage pipeline narrative with tooling notes.
- `docs/neo4j_data_dictionary.md` - Graph schema, node/relationship properties, and provenance.
- `docs/reproducibility.md` - Runtime estimation, logging outputs, and reproducibility checklist.
- `data/output/results_metrics.json` - Aggregated embedding, similarity, and community metrics consumed by analysis notebooks.

## License & Citation

This project is released under the MIT License (see `LICENSE`). If you build on this work in academic publications, please cite using the metadata in `CITATION.cff`.

**Note:** The code and documentation for this project was made with the assistance of OpenAI's ChatGPT and Codex.
