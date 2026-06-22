#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/proj/babyllm/chinese_cache}"

for step in 1000 5000 10000; do
  out="/mnt/proj/babyllm/chinese_cache/hf_models/zh-d22-step$(printf "%05d" "$step")"
  .venv/bin/python -m scripts.export_nanochat_hf \
    --step "$step" \
    --representation-layer -1 \
    --representation-hidden-states-mode all \
    --output-dir "$out"
done

for layer in 4 8 12 16 20 -1; do
  if [[ "$layer" == "-1" ]]; then
    suffix="final"
  else
    suffix="layer$(printf "%02d" "$layer")"
  fi
  out="/mnt/proj/babyllm/chinese_cache/hf_models/zh-d22-step10000-${suffix}-selected"
  .venv/bin/python -m scripts.export_nanochat_hf \
    --step 10000 \
    --representation-layer "$layer" \
    --representation-hidden-states-mode selected_only \
    --output-dir "$out"
done
