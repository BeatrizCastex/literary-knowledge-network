#!/usr/bin/env bash

set -euo pipefail

PIPELINE_START_EPOCH=$(date +%s)
PIPELINE_START_ISO=$(date -Iseconds)
RUN_ID=$(date -u +"%Y%m%dT%H%M%SZ")

declare -a STEP_TITLES=()
declare -a STEP_COMMANDS=()
declare -a STEP_DURATIONS=()

TOTAL_STEPS=7
CURRENT_STEP=0
SPINNER_ACTIVE=0

hash -r 2>/dev/null || true
PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python3 interpreter not found on PATH."
  exit 127
fi

trap '[[ $SPINNER_ACTIVE -eq 1 ]] && tput cnorm 2>/dev/null || true; exit 1' INT TERM

show_spinner() {
  local pid="$1"
  local message="$2"
  local spin='-\|/'
  local i=0
  SPINNER_ACTIVE=1
  if command -v tput >/dev/null 2>&1; then
    tput civis
  fi
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r%s %s" "$message" "${spin:i++%${#spin}:1}"
    sleep 0.2
  done
  printf "\r%-40s\r" ""
  if command -v tput >/dev/null 2>&1; then
    tput cnorm
  fi
  SPINNER_ACTIVE=0
}

run_step() {
  local title="$1"
  shift
  CURRENT_STEP=$((CURRENT_STEP + 1))
  local prefix="[$CURRENT_STEP/$TOTAL_STEPS]"
  printf "\n%s %s\n" "$prefix" "$title"

  local -a cmd=("$@")
  local command_str=""
  for arg in "${cmd[@]}"; do
    command_str+="$(printf '%q ' "$arg")"
  done
  command_str=${command_str%% }

  local step_start
  step_start=$(date +%s)

  "$@" &
  local cmd_pid=$!
  show_spinner "$cmd_pid" "   Working..." &
  local spinner_pid=$!

  set +e
  wait "$cmd_pid"
  local exit_code=$?
  set -e

  local step_end
  step_end=$(date +%s)
  local duration=$((step_end - step_start))

  STEP_TITLES+=("$title")
  STEP_COMMANDS+=("$command_str")
  STEP_DURATIONS+=("$duration")

  wait "$spinner_pid" 2>/dev/null || true
  printf "\r"
  if [ $exit_code -ne 0 ]; then
    echo "   ✖ ${title} (exit code $exit_code)"
    exit $exit_code
  else
    echo "   ✔ ${title}"
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <sample_size> [random_seed] [neo4j_password]"
  exit 1
fi

SAMPLE_SIZE="$1"
RANDOM_SEED="${2:-}"
if [[ $# -ge 3 ]]; then
  export NEO4J_PASSWORD="$3"
fi
if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
  echo "Warning: NEO4J_PASSWORD not set; scripts that require Neo4j auth may fail."
fi

RUNTIME_LOG_DIR="$SCRIPT_DIR/../logs/pipeline_runs"
if compgen -G "$RUNTIME_LOG_DIR/pipeline_run_*.json" > /dev/null; then
  echo -e "\n[info] Estimating pipeline runtime based on previous runs..."
  "$PYTHON_BIN" estimate_runtime.py --sample-size "$SAMPLE_SIZE" || true
  echo
fi

EXTRACTION_ARGS=(
  "--sample-size" "$SAMPLE_SIZE"
  "--export"
  "--export-dir" "data/processed/extraction"
)
if [[ -n "$RANDOM_SEED" ]]; then
  EXTRACTION_ARGS+=("--random-seed" "$RANDOM_SEED")
fi
run_step "Step 1: Data extraction (sample size ${SAMPLE_SIZE})" \
  "$PYTHON_BIN" data-extraction-v3.py "${EXTRACTION_ARGS[@]}"

run_step "Step 2: Data normalization" \
  "$PYTHON_BIN" data-normalization.py \
  --input-dir data/processed/extraction \
  --output-dir data/processed/normalized \
  --export \
  --auto-merge

run_step "Step 3: Data enrichment" \
  "$PYTHON_BIN" enritch-data.py \
  --input-dir data/processed/normalized \
  --output-dir data/processed/enriched \
  --keyword-method hybrid \
  --auto-wikidata \
  --export \
  --progress

run_step "Step 4: Data injection (Neo4j)" \
  "$PYTHON_BIN" data-injection.py \
  --input-dir data/processed/enriched \
  --verbose

run_step "Step 5: Embedding generation" \
  "$PYTHON_BIN" embedding.py \
  --input-dir data/processed/enriched \
  --output-dir data/output/embeddings \
  --method both \
  --progress \
  --verbose

run_step "Step 6: Similarity computation" \
  "$PYTHON_BIN" similarity.py \
  --embedding-file data/output/embeddings/work_embeddings_sbert.parquet \
  --output-file data/output/similarity/work_similarity.parquet \
  --top-k 20 \
  --threshold 0.50 \
  --tfidf-file data/output/embeddings/work_embeddings_tfidf.npz \
  --tfidf-weight 0.1 \
  --export-edge-csvs \
  --write-neo4j \
  --clear-existing \
  --verbose

run_step "Step 7: Community detection" \
  "$PYTHON_BIN" community_detection.py \
  --similarity-file data/output/similarity/work_similarity.parquet \
  --output-file data/output/communities/work_communities.parquet \
  --write-neo4j \
  --clear-existing \
  --verbose

PIPELINE_END_EPOCH=$(date +%s)
PIPELINE_END_ISO=$(date -Iseconds)
TOTAL_DURATION=$((PIPELINE_END_EPOCH - PIPELINE_START_EPOCH))

PIPELINE_LOG_DIR="$SCRIPT_DIR/../logs/pipeline_runs"
mkdir -p "$PIPELINE_LOG_DIR"
PIPELINE_LOG_PATH="$PIPELINE_LOG_DIR/pipeline_run_${RUN_ID}.json"
PIPELINE_LOG_REL="logs/pipeline_runs/pipeline_run_${RUN_ID}.json"

CONFIG_PATH="$SCRIPT_DIR/../config.yaml"
CONFIG_HASH=""
if [[ -f "$CONFIG_PATH" ]]; then
  CONFIG_HASH=$(sha256sum "$CONFIG_PATH" | awk '{print $1}')
fi

PIPELINE_STEP_TITLES_JSON=$(printf '%s\n' "${STEP_TITLES[@]}" | "$PYTHON_BIN" -c 'import sys,json; print(json.dumps([line.rstrip("\n") for line in sys.stdin]))')
PIPELINE_STEP_COMMANDS_JSON=$(printf '%s\n' "${STEP_COMMANDS[@]}" | "$PYTHON_BIN" -c 'import sys,json; print(json.dumps([line.rstrip("\n") for line in sys.stdin]))')
PIPELINE_STEP_DURATIONS_JSON=$(printf '%s\n' "${STEP_DURATIONS[@]}" | "$PYTHON_BIN" -c 'import sys,json; print(json.dumps([int(line.strip()) for line in sys.stdin if line.strip() != ""]))')

PIPELINE_SAMPLE_SIZE="$SAMPLE_SIZE"
PIPELINE_RANDOM_SEED="${RANDOM_SEED:-}"
if [[ -n "${NEO4J_PASSWORD:-}" ]]; then
  PIPELINE_NEO4J_PASSWORD_SET="true"
else
  PIPELINE_NEO4J_PASSWORD_SET="false"
fi

PIPELINE_DATA_INJECTION_REL=""
if [[ -f "$SCRIPT_DIR/../logs/data_injection_summary.json" ]]; then
  cp "$SCRIPT_DIR/../logs/data_injection_summary.json" "$PIPELINE_LOG_DIR/pipeline_run_${RUN_ID}_injection_summary.json"
  PIPELINE_DATA_INJECTION_REL="logs/pipeline_runs/pipeline_run_${RUN_ID}_injection_summary.json"
fi

export PIPELINE_LOG_PATH PIPELINE_LOG_REL PIPELINE_START_ISO PIPELINE_END_ISO TOTAL_DURATION RUN_ID
export PIPELINE_SAMPLE_SIZE PIPELINE_RANDOM_SEED PIPELINE_NEO4J_PASSWORD_SET CONFIG_HASH
export PIPELINE_STEP_TITLES_JSON PIPELINE_STEP_COMMANDS_JSON PIPELINE_STEP_DURATIONS_JSON
export PIPELINE_DATA_INJECTION_REL

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

log_path = Path(os.environ["PIPELINE_LOG_PATH"])
log_path.parent.mkdir(parents=True, exist_ok=True)

titles = json.loads(os.environ["PIPELINE_STEP_TITLES_JSON"])
commands = json.loads(os.environ["PIPELINE_STEP_COMMANDS_JSON"])
durations = json.loads(os.environ["PIPELINE_STEP_DURATIONS_JSON"])

steps = []
for idx, (title, command, duration) in enumerate(zip(titles, commands, durations), start=1):
    steps.append(
        {
            "index": idx,
            "title": title,
            "command": command,
            "duration_seconds": duration,
        }
    )

random_seed = os.environ.get("PIPELINE_RANDOM_SEED") or None
config_hash = os.environ.get("CONFIG_HASH") or None
data_injection = os.environ.get("PIPELINE_DATA_INJECTION_REL") or None

payload = {
    "pipeline_id": os.environ["RUN_ID"],
    "pipeline_start": os.environ["PIPELINE_START_ISO"],
    "pipeline_end": os.environ["PIPELINE_END_ISO"],
    "total_duration_seconds": int(os.environ["TOTAL_DURATION"]),
    "sample_size": int(os.environ["PIPELINE_SAMPLE_SIZE"]),
    "random_seed": random_seed,
    "neo4j_password_supplied": os.environ["PIPELINE_NEO4J_PASSWORD_SET"].lower() == "true",
    "config_sha256": config_hash,
    "steps": steps,
    "log_files": {
        "pipeline_run": os.environ["PIPELINE_LOG_REL"],
        "data_injection_summary": data_injection,
    },
}

log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

echo "Pipeline metadata logged to ${PIPELINE_LOG_REL}"

printf "\nPipeline complete.\n"
