#!/usr/bin/env bash
# Frozen-feature probes (F1-F3, plus F5-F7 on tasks with several named ROIs) and the
# radiomics baseline (R1) for one prepared task, confounder references, Cox probes
# for tasks with a binary label plus time/event (nnfb-cox), then the design grid.
# Meant to run on the host, after `nnfb-data prepare TASK`.
#
#   scripts/run_probes.sh TASK
#
# Settings (environment variables):
#   MODELS="cnn vit"            frozen encoders; model i runs on the i-th entry of GPUS
#   GPUS="0 1"                  one GPU per model, extractions run in parallel
#   RADIOMICS="original"        radiomics feature sets ("" skips radiomics, e.g. MRNet;
#                               "original filtered" adds the dropped filtered set, D25)
#   JOBS=32                     radiomics worker processes
#
# Safe to rerun: features already extracted are skipped; probes are cheap and refitted.
set -euo pipefail

TASK=${1:?usage: scripts/run_probes.sh TASK}
MODELS=${MODELS:-"cnn vit"}
GPUS=${GPUS:-"0 1"}
RADIOMICS=${RADIOMICS-"original"}
JOBS=${JOBS:-32}

cd "$(dirname "$0")/.."
LOG_DIR=outputs/logs/$TASK
mkdir -p "$LOG_DIR"
read -ra MODEL_LIST <<< "$MODELS"
read -ra GPU_LIST <<< "$GPUS"

echo "== Extract frozen features ($MODELS) in the background, logs in $LOG_DIR"
PIDS=()
for i in "${!MODEL_LIST[@]}"; do
  model=${MODEL_LIST[$i]}
  gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  uv run nnfb-features extract "$TASK" --model "$model" --device "cuda:$gpu" \
    >> "$LOG_DIR/features_$model.log" 2>&1 &
  PIDS+=($!)
done

for features in $RADIOMICS; do
  echo "== Radiomics: extract $features ($JOBS processes)"
  uv run nnfb-radiomics extract "$TASK" --features "$features" --jobs "$JOBS"
done

echo "== Waiting for feature extraction"
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "Feature extraction failed, see $LOG_DIR/features_*.log"; exit 1; }
done

echo "== Probes"
for model in $MODELS; do
  uv run nnfb-features probe "$TASK" --model "$model"
done
for features in $RADIOMICS; do
  uv run nnfb-radiomics probe "$TASK" --features "$features"
done

# Confounder references from labels.csv columns: group (e.g. centre) alone, age alone, and
# age with group (D9, D29).
has_column() {
  head -1 "${NNFB_DATA_DIR:-data}/processed/$TASK/labels.csv" | tr , '\n' | grep -qx "$1"
}
if has_column group; then
  uv run nnfb-predict group-baseline "$TASK"
fi
if has_column age; then
  uv run nnfb-predict covariate-baseline "$TASK" --column age
  if has_column group; then
    uv run nnfb-predict covariate-baseline "$TASK" --column age --column group
  fi
fi

# Tasks with a binary label plus time and event (UPENN): Cox probes via nnfb-cox, C-index
# (D32). Survival-only tasks (label = event, D27) got their Cox probes above (D33).
survival_only() {
  uv run python -c "import sys; from nnfoundation_bench.modeling.survival import is_survival; sys.exit(0 if is_survival('$TASK') else 1)"
}
if has_column time && has_column event && ! survival_only; then
  echo "== Cox probes (C-index)"
  for model in $MODELS; do
    uv run nnfb-cox probe "$TASK" --model "$model"
  done
  if [[ " $RADIOMICS " == *" original "* ]]; then
    uv run nnfb-cox probe "$TASK" --model radiomics
  fi
  if has_column age; then
    uv run nnfb-cox probe "$TASK" --model covariates --column age
  fi
fi

echo "== Design grid"
uv run nnfb-predict table "$TASK"
