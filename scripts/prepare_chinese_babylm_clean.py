"""
Clean and reshard Chinese BabyLM text for nanochat pretraining.

This is intentionally conservative: it keeps the same text-only nanochat parquet
format, applies document-level normalization/filtering, exact deduplication on
normalized text, and writes a fresh train/validation split.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset


WHITESPACE_RE = re.compile(r"\s+")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Chinese BabyLM parquet data for nanochat")
    parser.add_argument("--input-dir", type=Path, default=Path("chinese_cache/base_data_climbmix"))
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--output-name", type=str, default="base_data_climbmix")
    parser.add_argument("--num-train-shards", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.15)
    parser.add_argument("--max-most-common-char-ratio", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_base_dir(cli_base_dir: Path | None) -> Path:
    if cli_base_dir is not None:
        return cli_base_dir
    if os.environ.get("NANOCHAT_BASE_DIR"):
        return Path(os.environ["NANOCHAT_BASE_DIR"])
    return Path("chinese_cache_clean")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def keep_text(text: str, args) -> tuple[bool, str]:
    text = normalize_text(text)
    length = len(text)
    if length < args.min_chars or length > args.max_chars:
        return False, text
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return False, text
    cjk_ratio = sum(is_cjk(ch) for ch in visible) / len(visible)
    if cjk_ratio < args.min_cjk_ratio:
        return False, text
    most_common_ratio = Counter(visible).most_common(1)[0][1] / len(visible)
    if most_common_ratio > args.max_most_common_char_ratio:
        return False, text
    if "\ufffd" in text:
        return False, text
    return True, text


def load_text_dataset(input_dir: Path) -> Dataset:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")
    if input_dir.is_file():
        files = [str(input_dir)]
    else:
        files = [str(path) for path in sorted(input_dir.rglob("*.parquet"))]
    if not files:
        raise ValueError(f"No parquet files found under {input_dir}")
    return load_dataset("parquet", data_files=files, split="train")


def main() -> None:
    args = parse_args()
    base_dir = get_base_dir(args.base_dir)
    output_dir = base_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("*.parquet"))
    if existing and not args.overwrite:
        raise FileExistsError(f"{output_dir} already has parquet files; pass --overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()

    dataset = load_text_dataset(args.input_dir)
    if "text" not in dataset.column_names:
        raise ValueError(f"Expected a 'text' column, found {dataset.column_names}")

    seen: set[str] = set()
    rows = []
    stats = Counter(total=len(dataset), empty_or_non_string=0, filtered=0, duplicate=0, kept=0)
    for item in dataset:
        raw = item.get("text")
        if not isinstance(raw, str):
            stats["empty_or_non_string"] += 1
            continue
        keep, text = keep_text(raw, args)
        if not keep:
            stats["filtered"] += 1
            continue
        key = text.casefold()
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        rows.append({"text": text})
        stats["kept"] += 1

    clean = Dataset.from_list(rows)
    split = clean.train_test_split(test_size=args.val_fraction, seed=args.seed, shuffle=True)

    written = []
    for shard_idx in range(args.num_train_shards):
        shard = split["train"].shard(args.num_train_shards, shard_idx, contiguous=True)
        path = output_dir / f"shard_{shard_idx:05d}.parquet"
        shard.to_parquet(str(path))
        written.append(str(path))

    val_path = output_dir / "shard_99999.parquet"
    split["test"].to_parquet(str(val_path))
    written.append(str(val_path))

    metadata = {
        "source": str(args.input_dir),
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_cjk_ratio": args.min_cjk_ratio,
            "max_most_common_char_ratio": args.max_most_common_char_ratio,
            "normalization": "NFKC + whitespace collapse",
            "dedup": "exact normalized casefold text",
        },
        "stats": dict(stats),
        "num_rows_train": len(split["train"]),
        "num_rows_val": len(split["test"]),
        "num_train_shards": args.num_train_shards,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "written": written,
    }
    metadata_path = base_dir / "chinese_babylm_clean_data.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
