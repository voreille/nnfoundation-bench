#!/usr/bin/env bash
# Run the whole classification benchmark for one task: download, prepare, then
# fine-tune + evaluate every (model, fold), and summarise. Meant to run on the host.
#
#   scripts/run_benchmark.sh [TASK]
#
# Settings (environment variables):
#   GPUS="0 1"            one worker per GPU; runs are spread across them
#   MODELS="cnn vit"      pretrained models to fine-tune
#   SCRATCH=0             1 = also train each architecture from random init (baseline)
#   FOLDS="0 1 2 3 4"     split indices in splits.json
#   NUM_WORKERS=8         DataLoader workers (needs a large /dev/shm)
#   EXTRA_TRAIN_ARGS=""   passed through to nnfb-train, e.g. "--epochs 20"
#
# Safe to rerun: download/prepare skip finished files, and a run whose
# reports/results/<task>/<run>/fold<k>/metrics.json exists is skipped. An
# interrupted training run restarts from scratch.
set -euo pipefail

TASK=${1:-kits_liver}
GPUS=${GPUS:-"0 1"}
MODELS=${MODELS:-"cnn vit"}
SCRATCH=${SCRATCH:-0}
FOLDS=${FOLDS:-"0 1 2 3 4"}
NUM_WORKERS=${NUM_WORKERS:-8}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}

cd "$(dirname "$0")/.."
LOG_DIR=reports/logs/$TASK
mkdir -p "$LOG_DIR"
read -ra GPU_LIST <<< "$GPUS"

echo "== Download and prepare $TASK"
uv run nnfb-data download "$TASK"
uv run nnfb-data prepare "$TASK" --device "gpu:${GPU_LIST[0]}"

# Job list: "model scratch_flag fold"
JOBS=()
for fold in $FOLDS; do
  for model in $MODELS; do
    JOBS+=("$model 0 $fold")
    [[ $SCRATCH == 1 ]] && JOBS+=("$model 1 $fold")
  done
done

run_job() {  # gpu model scratch fold
  local gpu=$1 model=$2 scratch=$3 fold=$4 flag="" run
  [[ $scratch == 1 ]] && flag="--scratch"
  run=$(uv run python -c "from nnfoundation_bench.modeling.runs import run_name; \
print(run_name('$model', scratch=bool($scratch)))")
  if [[ -f reports/results/$TASK/$run/fold$fold/metrics.json ]]; then
    echo "[gpu $gpu] skip $run fold $fold (done)"
    return
  fi
  echo "[gpu $gpu] train $run fold $fold -> $LOG_DIR/${run}_fold$fold.log"
  # shellcheck disable=SC2086
  uv run nnfb-train "$TASK" --model "$model" --fold "$fold" --gpus "$gpu" $flag \
    --num-workers "$NUM_WORKERS" $EXTRA_TRAIN_ARGS >> "$LOG_DIR/${run}_fold$fold.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu uv run nnfb-predict predict "$TASK" --model "$model" \
    --fold "$fold" $flag --num-workers "$NUM_WORKERS" >> "$LOG_DIR/${run}_fold$fold.log" 2>&1
  echo "[gpu $gpu] done $run fold $fold"
}

worker() {  # worker_index gpu: takes every n-th job
  local i=$1 gpu=$2 j
  for ((j = i; j < ${#JOBS[@]}; j += ${#GPU_LIST[@]})); do
    # shellcheck disable=SC2086
    run_job "$gpu" ${JOBS[$j]} || echo "[gpu $gpu] FAILED ${JOBS[$j]}, see $LOG_DIR"
  done
}

echo "== ${#JOBS[@]} runs on GPU(s) $GPUS"
for i in "${!GPU_LIST[@]}"; do
  worker "$i" "${GPU_LIST[$i]}" &
done
wait

echo "== Summary"
uv run nnfb-predict summarize "$TASK"
