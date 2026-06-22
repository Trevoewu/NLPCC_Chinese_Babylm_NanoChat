# nanochat DeepWiki Notes for Chinese BabyLM

Source: <https://deepwiki.com/karpathy/nanochat>
Saved for the Chinese BabyLM adaptation work.

Last checked: 2026-05-22.

## Pipeline

nanochat is organized as a full training pipeline:

1. Tokenizer training: `scripts/tok_train.py`
2. Base pretraining: `scripts/base_train.py`
3. Optional SFT: `scripts/chat_sft.py`
4. Optional RL: `scripts/chat_rl.py`
5. Evaluation and inference: `scripts/base_eval.py`, `scripts/chat_eval.py`, `scripts/chat_cli.py`, `scripts/chat_web.py`

The main design knob is `--depth`. The model width, heads, learning-rate scaling, and default training horizon are derived from depth in `scripts/base_train.py`.

For our Chinese BabyLM path, we keep this design and only replace the dataset preparation stage with `scripts/prepare_chinese_babylm.py`.

## Data and Tokenizer

DeepWiki's speedrun notes describe tokenizer training as:

- prepare/download text shards
- train a RustBPE tokenizer with vocab size 32768
- evaluate compression with `scripts.tok_eval`

nanochat expects pretraining data as parquet files under:

```text
$NANOCHAT_BASE_DIR/base_data_climbmix
```

The last parquet shard in lexical order is used as validation. Our converter writes:

```text
shard_00000.parquet ... shard_00015.parquet
shard_99999.parquet
```

This lets the original `nanochat.dataset.parquets_iter_batched()` logic work without patching core dataloader code.

## 2x RTX 5090 Notes

The local machine has:

```text
2 x NVIDIA GeForce RTX 5090, 32607 MiB each, compute capability 12.0
```

Relevant DeepWiki hardware notes:

- DDP should be launched with one process per GPU via `torchrun` / `torch.distributed.run`.
- Effective batch size is `device_batch_size * max_seq_len * world_size * grad_accum_steps`.
- Rank 0 handles logging, sampling, and central checkpoint metadata.
- BF16 is the correct default for CUDA SM 80+ hardware.

## Attention and Precision

DeepWiki says nanochat uses Flash Attention 3 only on Hopper GPUs with BF16. On non-Hopper hardware such as Blackwell/5090, it falls back to PyTorch SDPA.

Practical consequence for our run script:

- default `NANOCHAT_DTYPE=bfloat16`
- do not enable `--fp8` by default
- use `--window-pattern L`, because the code warns that SDPA does not support the sliding-window path efficiently

## Current Chinese Defaults

`runs/chinese_babylm.sh` defaults are intentionally conservative for 2x 32GB 5090:

```text
NUM_GPUS=2
DEPTH=12
MAX_SEQ_LEN=1024
DEVICE_BATCH_SIZE=8
TOTAL_BATCH_SIZE=131072
NANOCHAT_DTYPE=bfloat16
window pattern = L
```

For a smoke run:

```bash
RUN_TOKENIZER_EVAL=0 NUM_GPUS=1 DEPTH=4 MAX_SEQ_LEN=512 DEVICE_BATCH_SIZE=4 TOTAL_BATCH_SIZE=4096 NUM_ITERATIONS=10 bash runs/chinese_babylm.sh
```

For the normal 2-GPU Chinese BabyLM run:

```bash
bash runs/chinese_babylm.sh
```
