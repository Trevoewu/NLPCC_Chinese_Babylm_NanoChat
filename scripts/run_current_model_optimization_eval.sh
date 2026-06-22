#!/usr/bin/env bash
set -euo pipefail

BABYLLM_DIR="${BABYLLM_DIR:-/mnt/proj/babyllm}"
PIPELINE_DIR="${PIPELINE_DIR:-/mnt/proj/chinese-babylm-eval-pipeline-official}"
RESULTS_DIR="${RESULTS_DIR:-/mnt/proj/babyllm/chinese_cache/eval_results_optimization}"
CONFIG_DIR="${CONFIG_DIR:-/mnt/proj/babyllm/chinese_cache/optimization_eval_configs}"
PYTHON="${PYTHON:-/mnt/proj/babyllm/.venv/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

if [[ "${WAIT_FOR_TRAIN:-0}" == "1" ]]; then
  pid_file="$BABYLLM_DIR/chinese_cache_clean/logs/zh-d12-clean-2gpu.pid"
  if [[ -f "$pid_file" ]]; then
    train_pid="$(cat "$pid_file")"
    while kill -0 "$train_pid" 2>/dev/null; do
      echo "Waiting for d12 clean training PID $train_pid ..."
      sleep 120
    done
  fi
fi

mkdir -p "$RESULTS_DIR" "$CONFIG_DIR"
cd "$PIPELINE_DIR"

write_config() {
  local name="$1"
  local model_path="$2"
  local config_path="$CONFIG_DIR/${name}.yaml"
  cat > "$config_path" <<EOF
models:
  - path: ${model_path}
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
  echo "$config_path"
}

run_tasks() {
  local config_path="$1"
  shift
  "$PYTHON" pipeline.py eval --config "$config_path" --tasks "$@" --force-redo
}

for step in 01000 05000 10000; do
  cfg="$(write_config "zh-d22-step${step}" "$BABYLLM_DIR/chinese_cache/hf_models/zh-d22-step${step}")"
  run_tasks "$cfg" zhoblimp hanzi_structure hanzi_pinyin word_fmri fmri
done

for suffix in layer04 layer08 layer12 layer16 layer20 final; do
  cfg="$(write_config "zh-d22-step10000-${suffix}-selected" "$BABYLLM_DIR/chinese_cache/hf_models/zh-d22-step10000-${suffix}-selected")"
  run_tasks "$cfg" word_fmri fmri
done

"$PYTHON" pipeline.py gather --config "$CONFIG_DIR" > "$RESULTS_DIR/gather.txt"
echo "Wrote gathered results to $RESULTS_DIR/gather.txt"
