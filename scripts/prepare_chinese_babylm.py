"""
Prepare a Chinese BabyLM Hugging Face dataset for nanochat pretraining.

nanochat expects base pretraining data as parquet shards in:

    $NANOCHAT_BASE_DIR/base_data_climbmix

The last parquet file in lexical order is used as validation. This script
converts a saved BabyLM dataset with a text column into that layout.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Chinese BabyLM data for nanochat")
    parser.add_argument("--data-dir", type=Path, default=Path("data/babylm-zho-100M"))
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--num-train-shards", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_base_dir(cli_base_dir):
    if cli_base_dir is not None:
        return cli_base_dir
    if os.environ.get("NANOCHAT_BASE_DIR"):
        return Path(os.environ["NANOCHAT_BASE_DIR"])
    return Path("chinese_cache")


def get_train_dataset(dataset):
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(f"DatasetDict must contain a 'train' split; found {list(dataset.keys())}")
        return dataset["train"]
    if isinstance(dataset, Dataset):
        return dataset
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def display_path(path):
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_training_data(data_dir, text_column):
    """Load BabyLM text from common local formats.

    Supports Hugging Face `save_to_disk` datasets, cloned HF parquet repos,
    plain text files, json/jsonl files, and directories containing these files.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_dir}")

    if data_dir.is_file():
        suffix = data_dir.suffix.lower()
        if suffix in {".txt", ".text"}:
            return load_dataset("text", data_files=str(data_dir), split="train")
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(data_dir), split="train")
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(data_dir), split="train")
        raise ValueError(f"Unsupported data file type: {data_dir}")

    if any((data_dir / name).exists() for name in ("dataset_dict.json", "dataset_info.json", "state.json")):
        return get_train_dataset(load_from_disk(str(data_dir)))

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if parquet_files:
        return load_dataset("parquet", data_files=[str(path) for path in parquet_files], split="train")

    jsonl_files = sorted(data_dir.rglob("*.jsonl"))
    if jsonl_files:
        return load_dataset("json", data_files=[str(path) for path in jsonl_files], split="train")

    txt_files = sorted(data_dir.rglob("*.txt"))
    if txt_files:
        return load_dataset("text", data_files=[str(path) for path in txt_files], split="train")

    raise ValueError(f"No supported training files found under {display_path(data_dir)}")


def validate_args(args):
    if args.num_train_shards < 1:
        raise ValueError("--num-train-shards must be >= 1")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")


def main():
    args = parse_args()
    validate_args(args)

    base_dir = get_base_dir(args.base_dir)
    output_dir = base_dir / "base_data_climbmix"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_parquets = sorted(output_dir.glob("*.parquet"))
    if existing_parquets and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains parquet files. "
            "Pass --overwrite to replace them."
        )
    if args.overwrite:
        for path in existing_parquets:
            path.unlink()

    train = load_training_data(args.data_dir, args.text_column)
    if args.text_column not in train.column_names:
        raise ValueError(f"Missing text column {args.text_column!r}; found {train.column_names}")

    train = train.select_columns([args.text_column])
    train = train.rename_column(args.text_column, "text") if args.text_column != "text" else train
    train = train.filter(lambda row: isinstance(row["text"], str) and len(row["text"]) > 0)

    split = train.train_test_split(
        test_size=args.val_fraction,
        seed=args.seed,
        shuffle=True,
    )

    written = []
    for shard_idx in range(args.num_train_shards):
        shard = split["train"].shard(
            num_shards=args.num_train_shards,
            index=shard_idx,
            contiguous=True,
        )
        path = output_dir / f"shard_{shard_idx:05d}.parquet"
        shard.to_parquet(str(path))
        written.append(str(path))

    # nanochat treats the lexically last parquet as validation.
    val_path = output_dir / "shard_99999.parquet"
    split["test"].to_parquet(str(val_path))
    written.append(str(val_path))

    metadata = {
        "source": display_path(args.data_dir),
        "base_dir": display_path(base_dir),
        "output_dir": display_path(output_dir),
        "text_column": "text",
        "num_rows_total": len(train),
        "num_rows_train": len(split["train"]),
        "num_rows_val": len(split["test"]),
        "num_train_shards": args.num_train_shards,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "validation_shard": display_path(val_path),
    }
    metadata_path = base_dir / "chinese_babylm_data.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Prepared Chinese BabyLM data for nanochat in {display_path(output_dir)}")
    print(f"Rows: train={metadata['num_rows_train']:,}, val={metadata['num_rows_val']:,}")
    print(f"Shards: {len(written)}")
    print(f"Metadata: {display_path(metadata_path)}")


if __name__ == "__main__":
    main()
