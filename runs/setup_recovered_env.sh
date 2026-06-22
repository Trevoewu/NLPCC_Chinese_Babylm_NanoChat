#!/usr/bin/env bash
set -euo pipefail

# Rebuild a runnable nanochat environment after container reset.
#
# Usage:
#   cd /mnt/proj/babyllm
#   bash runs/setup_recovered_env.sh

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv into ~/.local/bin"
    wget -qO- https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv install failed or ~/.local/bin is not on PATH" >&2
    exit 1
fi

uv python install 3.11
uv venv --python 3.11 .venv

. .venv/bin/activate

# nanochat pyproject pins torch==2.9.1 and defines the CUDA 12.8 index.
# RTX 5090 needs a recent CUDA-capable PyTorch build.
uv pip install -e ".[gpu]"
uv pip install transformers accelerate scikit-learn scipy pandas pyarrow matplotlib

python - <<'PY'
import torch
import datasets
import tokenizers
import tiktoken
import rustbpe
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("datasets", datasets.__version__)
print("tokenizers", tokenizers.__version__)
print("tiktoken", tiktoken.__version__)
print("rustbpe", rustbpe)
PY
