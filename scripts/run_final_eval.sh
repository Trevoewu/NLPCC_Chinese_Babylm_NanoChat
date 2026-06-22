#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
cd "${PROJECT_DIR}"
PROJECT_DIR_ABS="$(pwd)"

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${PROJECT_DIR_ABS}/${1#./}" ;;
  esac
}

display_path() {
  case "$1" in
    "${PROJECT_DIR_ABS}") printf '.\n' ;;
    "${PROJECT_DIR_ABS}"/*) printf '%s\n' "${1#"${PROJECT_DIR_ABS}/"}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

PIPELINE_DIR="${PIPELINE_DIR:-.external/chinese-babylm-pipeline-final}"
VENV_DIR="${VENV_DIR:-.venv-final-eval}"
MODEL_DIR="${MODEL_DIR:-chinese_cache_reproduce/hf_models/zh-d22-step10000-layer08-selected}"
MODEL_REPO="l0ulan/chinese-babylm-nanochat-d22-step10000"
MODEL_REVISION="7945c2a48ea4b4555509616b6942116782ef6295"
CONFIG_SRC="${CONFIG_SRC:-eval_configs/config_final.yaml}"
CONFIG_DST="configs/babyllm_final.yaml"
RESULT_JSON="${RESULT_JSON:-chinese_cache_reproduce/chinesebabylm_2026_final_results.json}"
PARALLEL_EVAL="${PARALLEL_EVAL:-1}"
RESULTS_DIR="${RESULTS_DIR:-chinese_cache_reproduce/final_eval_results}"

PIPELINE_DIR_ABS="$(resolve_path "${PIPELINE_DIR}")"
VENV_DIR_ABS="$(resolve_path "${VENV_DIR}")"
MODEL_DIR_ABS="$(resolve_path "${MODEL_DIR}")"
CONFIG_SRC_ABS="$(resolve_path "${CONFIG_SRC}")"
CONFIG_DST_ABS="${PIPELINE_DIR_ABS}/${CONFIG_DST}"
RESULT_JSON_ABS="$(resolve_path "${RESULT_JSON}")"
RESULTS_DIR_ABS="$(resolve_path "${RESULTS_DIR}")"
REQUIREMENTS_STAMP="${VENV_DIR_ABS}/.final_pipeline_requirements_installed"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${BOOTSTRAP_PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(command -v python)"
  else
    echo "No bootstrap Python found. Set BOOTSTRAP_PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

if [[ ! -f "${PIPELINE_DIR_ABS}/pipeline.py" ]]; then
  archive="$(mktemp --suffix=.zip)"
  mkdir -p "$(dirname "${PIPELINE_DIR_ABS}")"
  wget -q -O "${archive}" \
    https://github.com/chinese-babylm/chinese-babylm-pipeline-final/archive/refs/heads/main.zip
  rm -rf "${PIPELINE_DIR_ABS}" "${PIPELINE_DIR_ABS}-main"
  unzip -q "${archive}" -d "$(dirname "${PIPELINE_DIR_ABS}")"
  mv "${PIPELINE_DIR_ABS}-main" "${PIPELINE_DIR_ABS}"
  rm -f "${archive}"
fi

if [[ ! -x "${VENV_DIR_ABS}/bin/python" ]]; then
  rm -rf "${VENV_DIR_ABS}"
  "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR_ABS}"
fi

if [[ ! -f "${REQUIREMENTS_STAMP}" ]]; then
  "${VENV_DIR_ABS}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR_ABS}/bin/python" -m pip install -r "${PIPELINE_DIR_ABS}/requirements.txt"
  touch "${REQUIREMENTS_STAMP}"
fi
# The exported NanoChat tokenizer stores a tiktoken.Encoding in tokenizer.pkl.
# The official pipeline requirements do not currently include this dependency.
"${VENV_DIR_ABS}/bin/python" -m pip install tiktoken

# The default torch 2.7.0 wheel is built for CUDA 12.6 and stops at sm_90.
# Keep the official torch version while selecting its CUDA 12.8 build on
# Blackwell GPUs such as the RTX 5090 (sm_120).
if ! "${VENV_DIR_ABS}/bin/python" - <<'PY'
import sys
import torch

needs_sm120 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12
sys.exit(0 if not needs_sm120 or "sm_120" in torch.cuda.get_arch_list() else 1)
PY
then
  "${VENV_DIR_ABS}/bin/python" -m pip install --upgrade \
    "torch==2.7.0+cu128" --index-url https://download.pytorch.org/whl/cu128
fi

mkdir -p "${MODEL_DIR_ABS}" "$(dirname "${RESULT_JSON_ABS}")" "${RESULTS_DIR_ABS}"
if [[ ! -s "${MODEL_DIR_ABS}/pytorch_model.bin" ]]; then
  MODEL_REPO="${MODEL_REPO}" MODEL_REVISION="${MODEL_REVISION}" MODEL_DIR="${MODEL_DIR_ABS}" \
    "${VENV_DIR_ABS}/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_DIR"],
)
PY
fi

cp "${CONFIG_SRC_ABS}" "${CONFIG_DST_ABS}"
CONFIG_DST="${CONFIG_DST_ABS}" MODEL_DIR="${MODEL_DIR_ABS}" RESULTS_DIR="${RESULTS_DIR_ABS}" PIPELINE_DIR="${PIPELINE_DIR_ABS}" "${VENV_DIR_ABS}/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml

pipeline_dir = Path(os.environ["PIPELINE_DIR"]).resolve()
path = Path(os.environ["CONFIG_DST"])
with path.open(encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["models"][0]["path"] = os.path.relpath(Path(os.environ["MODEL_DIR"]).resolve(), pipeline_dir)
config["results_dir"] = os.path.relpath(Path(os.environ["RESULTS_DIR"]).resolve(), pipeline_dir)
with path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
PY

cd "${PIPELINE_DIR_ABS}"
"${VENV_DIR_ABS}/bin/python" pipeline.py download --tasks \
  zhoblimp xcomps_zh \
  hanzi_structure hanzi_pinyin \
  hanzi_structure_hidden hanzi_pinyin_hidden \
  word_fmri fmri \
  afqmc ocnli tnews cluewsc2020 c3 diagnostic_nli
if [[ "${PARALLEL_EVAL}" == "1" ]]; then
  GPU0_LOG="$(dirname "${RESULT_JSON_ABS}")/final_eval_gpu0.log"
  GPU1_LOG="$(dirname "${RESULT_JSON_ABS}")/final_eval_gpu1.log"

  CUDA_VISIBLE_DEVICES=0 "${VENV_DIR_ABS}/bin/python" pipeline.py eval \
    --config "${CONFIG_DST}" --tasks \
    zhoblimp xcomps_zh \
    hanzi_structure hanzi_pinyin \
    hanzi_structure_hidden hanzi_pinyin_hidden \
    word_fmri fmri

  CUDA_VISIBLE_DEVICES=0 "${VENV_DIR_ABS}/bin/python" pipeline.py eval \
    --config "${CONFIG_DST}" --tasks afqmc ocnli tnews \
    >"${GPU0_LOG}" 2>&1 &
  gpu0_pid=$!

  CUDA_VISIBLE_DEVICES=1 "${VENV_DIR_ABS}/bin/python" pipeline.py eval \
    --config "${CONFIG_DST}" --tasks cluewsc2020 c3 diagnostic_nli \
    >"${GPU1_LOG}" 2>&1 &
  gpu1_pid=$!

  gpu0_status=0
  gpu1_status=0
  wait "${gpu0_pid}" || gpu0_status=$?
  wait "${gpu1_pid}" || gpu1_status=$?
  if (( gpu0_status != 0 || gpu1_status != 0 )); then
    echo "Final eval workers failed: GPU0=${gpu0_status}, GPU1=${gpu1_status}" >&2
    exit 1
  fi
else
  "${VENV_DIR_ABS}/bin/python" pipeline.py eval --config "${CONFIG_DST}"
fi
"${VENV_DIR_ABS}/bin/python" pipeline.py gather --config "${CONFIG_DST}" --export "${RESULT_JSON_ABS}"

echo "Final leaderboard JSON: $(display_path "${RESULT_JSON_ABS}")"
