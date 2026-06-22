# Chinese BabyLM Training Report

Date: 2026-05-30

## Summary

This report records the recovered Chinese BabyLM training run based on `karpathy/nanochat`, including tokenizer training, base model pretraining, inference tooling, HuggingFace export, and evaluation with the official Chinese BabyLM eval pipeline.

The final usable artifact is a Chinese base language model trained as a continuation model, not an instruction/chat model. It should be prompted by plain text continuation. Chat wrappers such as `<|user_start|>` and `<|assistant_start|>` are not appropriate before SFT.

Final model:

```text
model tag: zh-d22-2gpu
checkpoint step: 10000
architecture: nanochat base causal LM
depth: 22
sequence length: 1024
vocab size: 32768
hidden size: 1408
heads: 22
estimated params: 1.123B
trained tokens: 655,360,000
```

Primary workspace after container reset:

```text
/mnt/proj/babyllm
```

Primary data source:

```text
/mnt/proj/chinese-babylm
```

## Background

The original `/home/trevor/babyllm` workspace and earlier training artifacts were lost after a container reset. We rebuilt the project under `/mnt/proj/babyllm`, using the available Chinese BabyLM data already present under `/mnt/proj`.

The recovered project keeps nanochat as the base training framework and adds Chinese BabyLM-specific preparation, tokenizer training, base-model CLI inference, checkpoint plotting, HuggingFace export, and evaluation support.

## Data Preparation

The Chinese BabyLM corpus was linked into the nanochat workspace:

```text
/mnt/proj/babyllm/data/babylm-zho-100M
```

Nanochat parquet shards were prepared with:

```bash
python scripts/prepare_chinese_babylm.py \
  --data-dir data/babylm-zho-100M \
  --base-dir "$NANOCHAT_BASE_DIR" \
  --num-train-shards 16 \
  --val-fraction 0.005 \
  --overwrite
```

Prepared split:

| Split | Rows |
| --- | ---: |
| train | 182,680 |
| validation | 918 |

Total shards: 17.

## Tokenizer

A Chinese-specific 32K tokenizer was trained from the BabyLM corpus.

Tokenizer artifacts:

```text
/mnt/proj/babyllm/chinese_cache/tokenizer/tokenizer.pkl
/mnt/proj/babyllm/chinese_cache/tokenizer/token_bytes.pt
```

Training command:

```bash
python -m scripts.tok_train \
  --vocab-size 32768 \
  --max-chars 200000000 \
  --doc-cap 10000
```

Tokenizer evaluation showed strong compression on Chinese BabyLM text:

| Split | Our bytes/token | GPT-2 bytes/token | GPT-4 bytes/token |
| --- | ---: | ---: | ---: |
| train | 4.98 | 1.43 | 2.20 |
| validation | 4.81 | 1.44 | 2.19 |

Interpretation: the tokenizer is highly specialized for the Chinese corpus and gives roughly half as many tokens as GPT-4 tokenizer on this dataset. This improves context efficiency for pretraining. The tradeoff is weaker compression on English/code/math samples, which is acceptable for this Chinese BabyLM target.

## Base Model Training

The final recovered run trained a depth-22 nanochat base model.

Checkpoint artifacts:

```text
/mnt/proj/babyllm/chinese_cache/base_checkpoints/zh-d22-2gpu/model_010000.pt
/mnt/proj/babyllm/chinese_cache/base_checkpoints/zh-d22-2gpu/meta_010000.json
```

Training log:

```text
/mnt/proj/babyllm/runs/logs/base_train_zh_d22_1gpu_recovered_20260527_181434.log
```

Loss curve artifacts:

```text
/mnt/proj/babyllm/docs/zh_d22_recovered_loss_curve.png
/mnt/proj/babyllm/docs/zh_d22_recovered_loss_curve.csv
```

Training command:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export NANOCHAT_BASE_DIR=/mnt/proj/babyllm/chinese_cache
export NANOCHAT_DTYPE=bfloat16
export NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS=1
export NANOCHAT_DISABLE_COMPILE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

python -m scripts.base_train \
  --depth 22 \
  --head-dim 64 \
  --window-pattern L \
  --max-seq-len 1024 \
  --device-batch-size 4 \
  --total-batch-size 65536 \
  --eval-every -1 \
  --core-metric-every -1 \
  --sample-every -1 \
  --save-every 1000 \
  --num-iterations 10000 \
  --run dummy \
  --model-tag zh-d22-2gpu
```

Final training stats:

| Metric | Value |
| --- | ---: |
| final step | 10000 |
| trained tokens | 655,360,000 |
| final train loss | 0.015389 |
| minimum observed train loss | 0.014984 |
| total training time | 363.75 minutes |
| peak GPU memory | about 19.1 GiB |

The model was trained on one RTX 5090. Two-GPU DDP was investigated, but under the current HAMI/NCCL container environment it repeatedly stalled before the first training step. We therefore completed the recovered final run on a single GPU.

## Inference Tooling

The original `scripts/chat_cli.py` is intended for `sft` or `rl` chat checkpoints. It injects chat-format tokens such as:

```text
<|user_start|>
<|assistant_start|>
```

That behavior is wrong for a base model. The base model should be evaluated as a continuation model.

We added a base-only CLI:

```text
/mnt/proj/babyllm/scripts/base_cli.py
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 \
NANOCHAT_BASE_DIR=/mnt/proj/babyllm/chinese_cache \
NANOCHAT_DTYPE=bfloat16 \
NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS=1 \
NANOCHAT_DISABLE_COMPILE=1 \
.venv/bin/python -m scripts.base_cli \
  --model-tag zh-d22-2gpu \
  --step 10000 \
  --prompt "春天来了，公园里的花" \
  --max-tokens 120 \
  --temperature 0.8 \
  --top-k 50 \
  --device-type cuda
```

The CLI was later adjusted to optionally display the original prompt and visible special tokens such as `<|bos|>`, making base-model continuation behavior easier to inspect.

## HuggingFace Export

To use the official eval pipeline without patching its code, the nanochat checkpoint was exported into a local HuggingFace-compatible `trust_remote_code` CausalLM directory.

Exported model:

```text
/mnt/proj/babyllm/chinese_cache/hf_models/zh-d22-step10000
```

Exporter:

```text
/mnt/proj/babyllm/scripts/export_nanochat_hf.py
```

Export contents include:

```text
config.json
configuration_nanochat.py
modeling_nanochat.py
tokenization_nanochat.py
pytorch_model.bin
tokenizer.pkl
```

Smoke tests passed:

- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- `AutoModel.from_pretrained(..., trust_remote_code=True)`
- finite logits check
- native nanochat versus HF export sanity check on input IDs and final-token prediction

Important compatibility fixes were made in the exported HF model/tokenizer code:

- support newer Transformers tokenizer keyword handling
- avoid direct assignment to tokenizer special-token ID properties
- implement `return_offsets_mapping`
- export RoPE buffers so meta initialization does not leave uninitialized tensors
- make `AutoModel` return FP32 hidden states for official finetune classifier heads

The eval pipeline itself was not modified for the final official run.

## Official Evaluation

We used a fresh copy of the official eval pipeline:

```text
/mnt/proj/chinese-babylm-eval-pipeline-official
```

Config:

```text
/mnt/proj/chinese-babylm-eval-pipeline-official/config_hf_zh_d22_step10000_official.yaml
```

Results directory:

```text
/mnt/proj/babyllm/chinese_cache/eval_results_hf_official
```

Logs:

```text
/mnt/proj/babyllm/runs/logs/eval_pipeline_official_hf/
```

The official code was checked to ensure no `nanochat` adapter remained in this fresh pipeline. The final run used the official `backend: causal` path and loaded our model through HuggingFace `trust_remote_code`.

Final gathered official scores:

| Model | zhoblimp | hanzi_structure | hanzi_pinyin | word_fmri | fmri | afqmc | ocnli | tnews | cluewsc2020 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zh-d22-step10000 | 67.45 | 51.85 | 41.60 | 55.54 | 8.02 | 68.61 | 55.59 | 51.31 | 59.54 |

These scores are from the HF-exported step-10000 checkpoint evaluated through the official pipeline.

## Observations

The base model can generate fluent Chinese-like continuation, but it is not instruction-following. Prompts like "你好，请介绍一下你自己" should not be expected to produce aligned assistant behavior before SFT. This is expected for a base model.

The tokenizer appears effective for Chinese BabyLM compression. The main quality bottleneck is no longer tokenizer round-trip or compression, but downstream modeling quality, data coverage, and whether an SFT stage is added for instruction behavior.

The official eval path is now cleaner than the earlier adapter-based path. Earlier experiments patched the eval pipeline with a `nanochat_causal` backend. The final recommended method is:

1. Train nanochat checkpoint.
2. Export checkpoint to HuggingFace CausalLM format.
3. Run the unmodified official eval pipeline with `backend: causal`.

## Remaining Work

- Add SFT data preparation if the goal shifts from base-model evaluation to interactive assistant behavior.
- Run official eval on additional checkpoints if checkpoint selection matters.
- Compare depth/model-size choices against training budget and BabyLM leaderboard behavior.
- Preserve `/mnt/proj/babyllm`, `/mnt/proj/chinese-babylm-eval-pipeline-official`, and HF export artifacts because `/home` may be lost after container reset.

