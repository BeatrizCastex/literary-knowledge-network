# Reproducibility & Run Metadata

This project keeps lightweight metadata for every full pipeline execution so that analyses can be reproduced or audited later. 

## What gets logged

- `logs/pipeline_runs/pipeline_run_<timestamp>.json` – Created by `scripts/run_pipeline.sh` after a successful execution. Captures:
  - pipeline start/end timestamps and total wall-clock duration
  - command-line used for each step plus its duration
  - sample size / random seed provided to Step 1
  - SHA-256 hash of `config.yaml` at run time
- `logs/pipeline_runs/pipeline_run_<timestamp>_injection_summary.json` – Snapshot of the Neo4j node/relationship counts produced in Step 4 (copied from `logs/data_injection_summary.json`).
- `data/output/results_metrics.json` – Aggregated embedding, similarity, and community summaries emitted by Step 7 (also mirrored under `data/output/analysis/`). Not strictly a log, but a convenient bundle used by the analysis notebooks.

## Estimating runtime

Use `scripts/estimate_runtime.py` to project how long the full pipeline will take for a given sample size or node target.

```bash
cd scripts
python3 estimate_runtime.py --sample-size 2500 --show-steps
python3 estimate_runtime.py --node-count 120000
python3 estimate_runtime.py --list-runs
```

## Keeping runs reproducible

1. Before each run, commit any configuration changes (`config.yaml`, scripts, requirements).
2. Execute `./run_pipeline.sh ...` from `scripts/` (see `README.md` for full usage).
3. Inspect the generated `logs/pipeline_runs/` JSON files and commit alongside analysis outputs when preparing a publication or release.
4. Optional: store the exact `.env` (with secrets redacted) or environment export in your lab notebook; the runtime logs already record the date, sample size, and relevant command-line arguments.
5. Check `data/output/results_metrics.json` into your research archive whenever you rely on the consolidated metrics for figures or tables.

Following this checklist ensures that anyone can reconstruct which parameters and code produced a given set of results or figures.
