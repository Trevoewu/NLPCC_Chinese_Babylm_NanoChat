#!/bin/bash
set -euo pipefail

# End-to-end Chinese BabyLM pretraining run for nanochat.
#
# Example:
#   bash runs/chinese_babylm.sh
#
# Useful overrides:
#   WANDB_RUN=zh-d12 DEPTH=12 NUM_ITERATIONS=5000 bash runs/chinese_babylm.sh
#   RUN_BASE_TRAIN=0 bash runs/chinese_babylm.sh   # prepare data + tokenizer only

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS="${NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/chinese_cache}"

DATA_DIR="${CHINESE_BABYLM_DATA_DIR:-data/babylm-zho-100M}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

VOCAB_SIZE="${VOCAB_SIZE:-32768}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-200000000}"
TOKENIZER_DOC_CAP="${TOKENIZER_DOC_CAP:-10000}"
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-16}"
VAL_FRACTION="${VAL_FRACTION:-0.005}"

NUM_GPUS="${NUM_GPUS:-2}"
DEPTH="${DEPTH:-12}"
HEAD_DIM="${HEAD_DIM:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
SAMPLE_EVERY="${SAMPLE_EVERY:-100}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5000}"
DEVICE_TYPE="${DEVICE_TYPE:-}"
WANDB_RUN="${WANDB_RUN:-dummy}"
RUN_BASE_TRAIN="${RUN_BASE_TRAIN:-1}"
RUN_TOKENIZER_EVAL="${RUN_TOKENIZER_EVAL:-1}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found at $PYTHON_BIN. Create the repo venv first, or set PYTHON_BIN." >&2
    exit 1
fi

mkdir -p "$NANOCHAT_BASE_DIR"

"$PYTHON_BIN" -m nanochat.report reset

"$PYTHON_BIN" scripts/prepare_chinese_babylm.py \
    --data-dir "$DATA_DIR" \
    --base-dir "$NANOCHAT_BASE_DIR" \
    --num-train-shards "$NUM_TRAIN_SHARDS" \
    --val-fraction "$VAL_FRACTION" \
    --overwrite

"$PYTHON_BIN" -m scripts.tok_train \
    --vocab-size "$VOCAB_SIZE" \
    --max-chars "$TOKENIZER_MAX_CHARS" \
    --doc-cap "$TOKENIZER_DOC_CAP"

if [ "$RUN_TOKENIZER_EVAL" = "1" ]; then
    "$PYTHON_BIN" -m scripts.tok_eval
fi

if [ "$RUN_BASE_TRAIN" = "1" ]; then
    DEVICE_ARGS=()
    if [ -n "$DEVICE_TYPE" ]; then
        DEVICE_ARGS=(--device-type "$DEVICE_TYPE")
    fi

    TRAIN_ARGS=(
        "${DEVICE_ARGS[@]}"
        --depth "$DEPTH"
        --head-dim "$HEAD_DIM"
        --window-pattern L
        --max-seq-len "$MAX_SEQ_LEN"
        --device-batch-size "$DEVICE_BATCH_SIZE"
        --total-batch-size "$TOTAL_BATCH_SIZE"
        --eval-every "$EVAL_EVERY"
        --eval-tokens "$EVAL_TOKENS"
        --core-metric-every -1
        --sample-every "$SAMPLE_EVERY"
        --num-iterations "$NUM_ITERATIONS"
        --run "$WANDB_RUN"
        --model-tag "zh-d${DEPTH}"
    )

    if [ "$NUM_GPUS" -gt 1 ]; then
        "$PYTHON_BIN" -m torch.distributed.run \
            --standalone \
            --nproc_per_node "$NUM_GPUS" \
            -m scripts.base_train \
            -- "${TRAIN_ARGS[@]}"
    else
        "$PYTHON_BIN" -m scripts.base_train "${TRAIN_ARGS[@]}"
    fi
fi

"$PYTHON_BIN" -m nanochat.report generate
