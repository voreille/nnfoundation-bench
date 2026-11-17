#!/usr/bin/env bash
# Frozen-feature probes (F1-F3) and radiomics baselines (R1-R5) for one prepared task,
# then the design grid. Meant to run on the host, after `nnfb-data prepare TASK`.
#
#   scripts/run_probes.sh TASK
#
# Settings (environment variables):
#   MODELS="cnn vit"            frozen encoders; model i runs on the i-th entry of GPUS
#   GPUS="0 1"                  one GPU per model, extractions run in parallel
#   RADIOMICS="original filtered"  radiomics feature sets ("" skips radiomics, e.g. MRNet)
#   JOBS=32                     radiomics worker processes
#
# Safe to rerun: features already extracted are skipped; probes are cheap and refitted.
set -euo pipefail

TASK=${1:?usage: scripts/run_probes.sh TASK}
MODELS=${MODELS:-"cnn vit"}
GPUS=${GPUS:-"0 1"}
RADIOMICS=${RADIOMICS-"original filtered"}
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
  uv run nnfb-radiomics probe "$TASK" --features "$features" --corr-threshold 0.9
  uv run nnfb-radiomics probe "$TASK" --features "$features" --classifier enet
  uv run nnfb-radiomics probe "$TASK" --features "$features" --corr-threshold 0.9 \
    --classifier enet
done

# Confounder reference (e.g. centre alone) for tasks whose labels.csv has a group column
if head -1 "${NNFB_DATA_DIR:-data}/processed/$TASK/labels.csv" | grep -q group; then
  uv run nnfb-predict group-baseline "$TASK"
fi

echo "== Design grid"
uv run nnfb-predict table "$TASK"
