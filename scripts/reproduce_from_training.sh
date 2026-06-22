#!/usr/bin/env bash
set -euo pipefail

# Reproduce the submitted Chinese BabyLM NanoChat model from training data,
# then export the final checkpoint and run the official final evaluation.
#
# Expected runtime on the original machine: about 6-7 hours for training, plus
# final evaluation time. The original final training run used one RTX 5090.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BASE_DIR="${BASE_DIR:-${PROJECT_DIR}/chinese_cache_reproduce}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/babylm-zho-100M}"
MODEL_TAG="${MODEL_TAG:-zh-d22-2gpu}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${BASE_DIR}/hf_models/zh-d22-step10000-layer08-selected}"
PIPELINE_DIR="${PIPELINE_DIR:-/mnt/proj/chinese-babylm-pipeline-final}"
FINAL_RESULTS_DIR="${FINAL_RESULTS_DIR:-${BASE_DIR}/final_eval_results}"
FINAL_RESULT_JSON="${FINAL_RESULT_JSON:-${BASE_DIR}/chinesebabylm_2026_final_results.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NANOCHAT_BASE_DIR="${BASE_DIR}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS="${NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS:-1}"
export NANOCHAT_DISABLE_COMPILE="${NANOCHAT_DISABLE_COMPILE:-1}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-disabled}"

cd "${PROJECT_DIR}"
mkdir -p "${BASE_DIR}" "${PROJECT_DIR}/runs/logs"

if [[ ! -x .venv/bin/python ]]; then
  bash runs/setup_recovered_env.sh
fi
PYTHON="${PYTHON:-${PROJECT_DIR}/.venv/bin/python}"

if [[ ! -e "${DATA_DIR}" ]]; then
  echo "Downloading official Chinese BabyLM corpus to ${DATA_DIR}"
  "${PYTHON}" - <<PY
from pathlib import Path
from datasets import load_dataset
out = Path("${DATA_DIR}")
out.parent.mkdir(parents=True, exist_ok=True)
ds = load_dataset("chinese-babylm-org/babylm-zho-100M")
ds.save_to_disk(str(out))
PY
fi

if [[ ! -f "${BASE_DIR}/base_data_climbmix/shard_99999.parquet" ]]; then
  "${PYTHON}" scripts/prepare_chinese_babylm.py \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --num-train-shards 16 \
    --val-fraction 0.005 \
    --seed 42 \
    --overwrite
fi

if [[ ! -f "${BASE_DIR}/tokenizer/tokenizer.pkl" ]]; then
  "${PYTHON}" -m scripts.tok_train \
    --vocab-size 32768 \
    --max-chars 200000000 \
    --doc-cap 10000
fi

checkpoint="${BASE_DIR}/base_checkpoints/${MODEL_TAG}/model_$(printf "%06d" "${TRAIN_STEPS}").pt"
if [[ ! -f "${checkpoint}" ]]; then
  log="${PROJECT_DIR}/runs/logs/reproduce_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).log"
  echo "Training ${MODEL_TAG} to step ${TRAIN_STEPS}. Log: ${log}"
  "${PYTHON}" -m scripts.base_train \
    --depth 22 \
    --aspect-ratio 64 \
    --head-dim 64 \
    --window-pattern L \
    --max-seq-len 1024 \
    --device-batch-size 4 \
    --total-batch-size 65536 \
    --eval-every -1 \
    --core-metric-every -1 \
    --sample-every -1 \
    --save-every 1000 \
    --num-iterations "${TRAIN_STEPS}" \
    --run dummy \
    --model-tag "${MODEL_TAG}" \
    2>&1 | tee "${log}"
fi

"${PYTHON}" -m scripts.export_nanochat_hf \
  --checkpoint-dir "${BASE_DIR}/base_checkpoints/${MODEL_TAG}" \
  --tokenizer-dir "${BASE_DIR}/tokenizer" \
  --step "${TRAIN_STEPS}" \
  --representation-layer 8 \
  --representation-hidden-states-mode selected_only \
  --output-dir "${HF_MODEL_DIR}"

PROJECT_DIR="${PROJECT_DIR}" \
PIPELINE_DIR="${PIPELINE_DIR}" \
MODEL_DIR="${HF_MODEL_DIR}" \
RESULTS_DIR="${FINAL_RESULTS_DIR}" \
RESULT_JSON="${FINAL_RESULT_JSON}" \
PARALLEL_EVAL="${PARALLEL_EVAL:-1}" \
bash scripts/run_final_eval.sh

echo "Reproduced final result JSON: ${FINAL_RESULT_JSON}"
