#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${PWD}/chinese_cache_clean}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS="${NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS:-1}"
export NANOCHAT_DISABLE_COMPILE="${NANOCHAT_DISABLE_COMPILE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 -m scripts.base_train \
  --depth 12 \
  --aspect-ratio 64 \
  --head-dim 64 \
  --max-seq-len 1024 \
  --window-pattern L \
  --num-iterations 10000 \
  --device-batch-size 8 \
  --total-batch-size 65536 \
  --eval-every -1 \
  --eval-tokens 4194304 \
  --core-metric-every -1 \
  --sample-every -1 \
  --save-every 1000 \
  --model-tag zh-d12-clean-2gpu
