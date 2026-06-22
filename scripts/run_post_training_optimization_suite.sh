#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export BABYLLM_DIR="${BABYLLM_DIR:-/mnt/proj/babyllm}"
export PIPELINE_DIR="${PIPELINE_DIR:-/mnt/proj/chinese-babylm-eval-pipeline-official}"
export PYTHON="${PYTHON:-/mnt/proj/babyllm/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

pid_file="$BABYLLM_DIR/chinese_cache_clean/logs/zh-d12-clean-2gpu.pid"
if [[ -f "$pid_file" ]]; then
  train_pid="$(cat "$pid_file")"
  while kill -0 "$train_pid" 2>/dev/null; do
    echo "Waiting for d12 clean training PID $train_pid ..."
    sleep 120
  done
fi

echo "Running d22 checkpoint and cognitive layer sweeps..."
WAIT_FOR_TRAIN=0 bash "$BABYLLM_DIR/scripts/run_current_model_optimization_eval.sh"

echo "Exporting and evaluating d12 clean final checkpoint..."
WAIT_FOR_TRAIN=0 bash "$BABYLLM_DIR/scripts/run_d12_clean_posttrain_eval.sh"

echo "Post-training optimization suite finished."
