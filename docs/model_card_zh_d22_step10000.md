---
language:
- zh
pipeline_tag: text-generation
library_name: transformers
tags:
- chinese
- causal-lm
- babyllm
- nanochat
- custom-code
- trust-remote-code
inference: false
---

# Chinese BabyLM NanoChat d22 Step 10000

## Model Summary

`chinese-babylm-nanochat-d22-step10000` is a Chinese base causal language model trained with the `karpathy/nanochat` codebase and adapted for the NLPCC Chinese BabyLM setting.

This is a **base continuation model**, not an instruction-tuned or chat-aligned assistant. It should be prompted with plain text and interpreted as next-token continuation. It has not gone through SFT, RLHF, DPO, or safety alignment.

## Model Details

| Field | Value |
| --- | --- |
| Model type | Decoder-only causal language model |
| Architecture | NanoChat `NanoChatForCausalLM` |
| Training objective | Next-token prediction |
| Language | Chinese |
| Checkpoint | `zh-d22-2gpu`, step `10000` |
| Context length | 1024 tokens |
| Vocabulary size | 32768 |
| Tokenizer | Custom Chinese tokenizer trained on Chinese BabyLM text |
| Precision | bfloat16 |
| Export format | HuggingFace `trust_remote_code` |

## Architecture

The model configuration exported to HuggingFace is:

| Parameter | Value |
| --- | ---: |
| `sequence_len` | 1024 |
| `vocab_size` | 32768 |
| `n_layer` | 22 |
| `n_head` | 22 |
| `n_kv_head` | 22 |
| `n_embd` | 1408 |
| Head dimension | 64 |
| `window_pattern` | `L` |
| Estimated parameters | ~1.123B |

The HuggingFace export includes custom model and tokenizer code:

```text
configuration_nanochat.py
modeling_nanochat.py
tokenization_nanochat.py
```

Loaders must use `trust_remote_code=True`.

## Training Data

The model was trained on the Chinese BabyLM corpus available in the training environment under:

```text
/mnt/proj/chinese-babylm
/mnt/proj/babyllm/data/babylm-zho-100M
```

The corpus was converted into nanochat parquet shards:

| Split | Rows |
| --- | ---: |
| Train | 182,680 |
| Validation | 918 |

Total prepared shards: 17.

No extra instruction data was used for this checkpoint.

## Tokenizer

A 32K Chinese tokenizer was trained from the Chinese BabyLM training data.

Tokenizer artifacts in the original training workspace:

```text
/mnt/proj/babyllm/chinese_cache/tokenizer/tokenizer.pkl
/mnt/proj/babyllm/chinese_cache/tokenizer/token_bytes.pt
```

Tokenizer compression on Chinese BabyLM text:

| Split | This tokenizer bytes/token | GPT-2 bytes/token | GPT-4/cl100k bytes/token |
| --- | ---: | ---: | ---: |
| Train | 4.98 | 1.43 | 2.20 |
| Validation | 4.81 | 1.44 | 2.19 |

This tokenizer is specialized for Chinese BabyLM data. It gives much better compression on the target corpus than GPT-2 or GPT-4 tokenizers, at the cost of weaker general-purpose compression on English, code, math, and mixed-domain text.

## Training Setup

The final recovered training run used one RTX 5090 GPU. Two-GPU DDP was investigated, but the available HAMI/NCCL container environment repeatedly stalled before the first training step, so the final recovered checkpoint was trained on a single GPU.

Main training configuration:

| Field | Value |
| --- | ---: |
| Depth | 22 |
| Sequence length | 1024 |
| Device batch size | 4 |
| Total batch size | 65536 tokens |
| Iterations | 10000 |
| Trained tokens | 655,360,000 |
| Dtype | bfloat16 |
| Final train loss | 0.015389 |
| Minimum observed train loss | 0.014984 |
| Total training time | 363.75 minutes |
| Peak memory | ~19.1 GiB |

The training checkpoint in the original workspace:

```text
/mnt/proj/babyllm/chinese_cache/base_checkpoints/zh-d22-2gpu/model_010000.pt
/mnt/proj/babyllm/chinese_cache/base_checkpoints/zh-d22-2gpu/meta_010000.json
```

## Official Evaluation

The model was exported to HuggingFace format and evaluated with a fresh copy of the official Chinese BabyLM eval pipeline:

```text
SiyuanSong2004/chinese-babylm-eval-pipeline
```

The final official run used the unmodified official pipeline with `backend: causal`. The evaluation pipeline was not patched with a NanoChat-specific backend for these results.

Evaluation config in the original workspace:

```text
/mnt/proj/chinese-babylm-eval-pipeline-official/config_hf_zh_d22_step10000_official.yaml
```

Results directory:

```text
/mnt/proj/babyllm/chinese_cache/eval_results_hf_official
```

Final gathered official scores:

| Task | Score |
| --- | ---: |
| `zhoblimp` | 67.45 |
| `hanzi_structure` | 51.85 |
| `hanzi_pinyin` | 41.60 |
| `word_fmri` | 55.54 |
| `fmri` | 8.02 |
| `afqmc` | 68.61 |
| `ocnli` | 55.59 |
| `tnews` | 51.31 |
| `cluewsc2020` | 59.54 |

## Usage

Because this repository contains custom model/tokenizer code, use `trust_remote_code=True`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "l0ulan/chinese-babylm-nanochat-d22-step10000"

tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "春天来了，公园里的花"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        top_k=50,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
```

For local nanochat inference in the original workspace, use the base-model continuation CLI instead of the chat CLI:

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

## Limitations

- This is a base model, not a chat model. It does not reliably follow instructions.
- Outputs may be unrelated to user intent if prompted like an assistant.
- The model has not been safety-aligned.
- The training corpus is Chinese BabyLM-focused; general-domain multilingual performance is not the goal of this checkpoint.
- The model requires custom code loading with `trust_remote_code=True`.

## Recommended Use

Appropriate uses:

- Chinese BabyLM research
- Base LM continuation experiments
- Tokenizer/model architecture experiments
- Downstream SFT or probing baselines
- Re-running the official Chinese BabyLM eval pipeline

Not recommended as-is:

- Production assistant use
- Safety-critical applications
- Instruction-following/chat evaluation before SFT

## Acknowledgements

This checkpoint was trained with the `karpathy/nanochat` codebase and evaluated with the Chinese BabyLM eval pipeline from `SiyuanSong2004/chinese-babylm-eval-pipeline`.

