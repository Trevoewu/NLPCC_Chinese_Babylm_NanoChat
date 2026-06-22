#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${PWD}/chinese_cache}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS="${NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS:-1}"

.venv/bin/python -m scripts.base_train \
    --depth "${DEPTH:-12}" \
    --head-dim "${HEAD_DIM:-64}" \
    --window-pattern L \
    --max-seq-len "${MAX_SEQ_LEN:-1024}" \
    --device-batch-size "${DEVICE_BATCH_SIZE:-8}" \
    --total-batch-size "${TOTAL_BATCH_SIZE:-65536}" \
    --eval-every "${EVAL_EVERY:-100}" \
    --eval-tokens "${EVAL_TOKENS:-524288}" \
    --core-metric-every -1 \
    --sample-every "${SAMPLE_EVERY:-100}" \
    --num-iterations "${NUM_ITERATIONS:-5000}" \
    --run "${WANDB_RUN:-dummy}" \
    --model-tag "${MODEL_TAG:-zh-d12}"
