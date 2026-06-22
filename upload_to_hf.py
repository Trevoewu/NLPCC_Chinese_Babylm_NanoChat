#!/usr/bin/env python3
import os
from pathlib import Path


def drop_socks_proxy() -> None:
    # The cluster's proxy helper may set all_proxy to socks5://...
    # httpx needs the optional socksio package for SOCKS. HuggingFace upload works
    # fine through the HTTP proxy, so prefer http_proxy/https_proxy here.
    for key in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(key, "")
        if value.lower().startswith(("socks://", "socks4://", "socks5://", "socks5h://")):
            os.environ.pop(key, None)


def main() -> None:
    drop_socks_proxy()
    from huggingface_hub import HfApi

    repo_id = os.environ.get("REPO_ID")
    token = os.environ.get("HF_TOKEN")
    if not repo_id:
        raise SystemExit("REPO_ID is not set")
    if not token:
        raise SystemExit("HF_TOKEN is not set")

    folder = Path("/mnt/proj/babyllm/chinese_cache/hf_models/zh-d22-step10000")
    required = [
        "config.json",
        "configuration_nanochat.py",
        "modeling_nanochat.py",
        "tokenization_nanochat.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.pkl",
        "pytorch_model.bin",
    ]
    missing = [name for name in required if not (folder / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files under {folder}: {missing}")

    api = HfApi(token=token)

    print(f"Repo: {repo_id}", flush=True)
    print(f"Folder: {folder}", flush=True)
    print("Creating repo if needed...", flush=True)
    private = os.environ.get("HF_PRIVATE", "").lower() in {"1", "true", "yes"}
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    print("Uploading folder...", flush=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message="Upload Chinese BabyLM nanochat d22 step10000 HF export",
        ignore_patterns=["eval_hf_smoke.yaml"],
    )

    print(f"Done: https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
