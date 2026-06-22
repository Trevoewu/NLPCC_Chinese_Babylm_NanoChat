#!/usr/bin/env python3
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    repo_id = os.environ.get("REPO_ID")
    token = os.environ.get("HF_TOKEN")
    if not repo_id:
        raise SystemExit("REPO_ID is not set")
    if not token:
        raise SystemExit("HF_TOKEN is not set")

    default_model_card = (
        Path(__file__).resolve().parent
        / "chinese_cache/hf_models/zh-d22-step10000/README.modelcard.md"
    )
    model_card = Path(os.environ.get("MODEL_CARD", default_model_card))
    if not model_card.exists():
        raise SystemExit(f"Missing model card: {model_card}")

    api = HfApi(token=token)
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        commit_message="Update model card",
    )
    print(f"Updated model card: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
