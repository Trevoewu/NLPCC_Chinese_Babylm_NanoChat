#!/usr/bin/env bash
set -euo pipefail

BABYLLM_DIR="${BABYLLM_DIR:-/mnt/proj/babyllm}"
PIPELINE_DIR="${PIPELINE_DIR:-/mnt/proj/chinese-babylm-eval-pipeline-official}"
PYTHON="${PYTHON:-/mnt/proj/babyllm/.venv/bin/python}"
STEP="${STEP:-10000}"
MODEL_TAG="${MODEL_TAG:-zh-d12-clean-2gpu}"
RESULTS_DIR="${RESULTS_DIR:-/mnt/proj/babyllm/chinese_cache_clean/eval_results_official}"
CONFIG_DIR="${CONFIG_DIR:-/mnt/proj/babyllm/chinese_cache_clean/eval_configs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

pid_file="$BABYLLM_DIR/chinese_cache_clean/logs/zh-d12-clean-2gpu.pid"
if [[ "${WAIT_FOR_TRAIN:-1}" == "1" && -f "$pid_file" ]]; then
  train_pid="$(cat "$pid_file")"
  while kill -0 "$train_pid" 2>/dev/null; do
    echo "Waiting for d12 clean training PID $train_pid ..."
    sleep 120
  done
fi

step_padded="$(printf "%06d" "$STEP")"
checkpoint="$BABYLLM_DIR/chinese_cache_clean/base_checkpoints/${MODEL_TAG}/model_${step_padded}.pt"
if [[ ! -f "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 1
fi

hf_dir="$BABYLLM_DIR/chinese_cache_clean/hf_models/zh-d12-clean-step$(printf "%05d" "$STEP")"
mkdir -p "$RESULTS_DIR" "$CONFIG_DIR"

cd "$BABYLLM_DIR"
NANOCHAT_BASE_DIR="$BABYLLM_DIR/chinese_cache_clean" "$PYTHON" -m scripts.export_nanochat_hf \
  --checkpoint-dir "$BABYLLM_DIR/chinese_cache_clean/base_checkpoints/${MODEL_TAG}" \
  --tokenizer-dir "$BABYLLM_DIR/chinese_cache_clean/tokenizer" \
  --step "$STEP" \
  --representation-layer -1 \
  --representation-hidden-states-mode all \
  --output-dir "$hf_dir"

config="$CONFIG_DIR/zh-d12-clean-step$(printf "%05d" "$STEP").yaml"
cat > "$config" <<EOF
models:
  - path: ${hf_dir}
    backend: causal

eval_dir: evaluation_data
results_dir: ${RESULTS_DIR}

tasks:
  zero_shot:
    - zhoblimp
    - hanzi_structure
    - hanzi_pinyin
  cogbench:
    - word_fmri
    - fmri
  finetune: []

finetune_hparams:
  lr: 3e-5
  batch_size: 32
  max_epochs: 10
  wsc_epochs: 30
  sequence_length: 128
  seed: 42
EOF

cd "$PIPELINE_DIR"
"$PYTHON" pipeline.py eval --config "$config" --tasks zhoblimp hanzi_structure hanzi_pinyin word_fmri fmri --force-redo
"$PYTHON" pipeline.py gather --config "$CONFIG_DIR" > "$RESULTS_DIR/gather.txt"
echo "Wrote gathered results to $RESULTS_DIR/gather.txt"
