# Chinese BabyLM Tokenizer Evaluation

Date: 2026-05-22

Tokenizer artifact:

```text
/home/trevor/babyllm/chinese_cache/tokenizer/tokenizer.pkl
/home/trevor/babyllm/chinese_cache/tokenizer/token_bytes.pt
```

Training configuration:

```text
vocab_size: 32768
max_chars: 200,000,000
doc_cap: 10,000
training_time: 176.31s
base_dir: /home/trevor/babyllm/chinese_cache
```

## Round Trip Check

The trained tokenizer was loaded through `nanochat.tokenizer.get_tokenizer()` with:

```bash
export NANOCHAT_BASE_DIR=/home/trevor/babyllm/chinese_cache
```

Two Chinese examples were encoded and decoded successfully:

```text
你好世界。我们正在训练中文 BabyLM 分词器。
今天天气不错，适合继续预训练一个小语言模型。
```

Both satisfied:

```python
decode(encode(text)) == text
```

## Built-In tok_eval Results

`scripts.tok_eval` compares GPT-2, GPT-4 (`cl100k_base`), and the trained tokenizer.

On the Chinese BabyLM training text:

| Tokenizer | Tokens | Bytes/Token |
| --- | ---: | ---: |
| GPT-2 | 17,057,557 | 1.43 |
| GPT-4 | 11,097,674 | 2.20 |
| Ours | 4,901,856 | 4.98 |

On the Chinese BabyLM validation text:

| Tokenizer | Tokens | Bytes/Token |
| --- | ---: | ---: |
| GPT-2 | 1,445,873 | 1.44 |
| GPT-4 | 947,200 | 2.19 |
| Ours | 432,486 | 4.81 |

Relative token reduction versus GPT-4:

```text
train: 55.8% fewer tokens
val:   54.3% fewer tokens
```

## Chinese Sample Evaluation

Additional evaluation sampled 2000 training documents and all 918 validation documents from the Chinese BabyLM parquet data.

Train sample:

| Tokenizer | Bytes | Tokens | Bytes/Token | Median Tokens/Doc |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 | 3,765,408 | 2,635,046 | 1.429 | 410.0 |
| GPT-4 | 3,765,408 | 1,714,157 | 2.197 | 257.5 |
| Ours | 3,765,408 | 752,960 | 5.001 | 136.0 |

Validation sample:

| Tokenizer | Bytes | Tokens | Bytes/Token | Median Tokens/Doc |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 | 1,986,531 | 1,388,132 | 1.431 | 488.5 |
| GPT-4 | 1,986,531 | 904,025 | 2.197 | 306.0 |
| Ours | 1,986,531 | 401,568 | 4.947 | 157.0 |

## Interpretation

The tokenizer is strongly specialized for the Chinese BabyLM corpus. It compresses the target Chinese train/validation data much better than GPT-2 and GPT-4 tokenizers, despite using a smaller vocabulary than GPT-4.

This is expected to improve training efficiency for the Chinese base model because each fixed context window covers substantially more Chinese text.

The tradeoff is reduced general-purpose compression. On the built-in English, Korean, code, math, and science examples, this tokenizer performs worse than GPT-2/GPT-4. This is acceptable for the current objective: train a Chinese BabyLM base model.

The next decisive metric is base model validation BPB (`val_bpb`) after pretraining. Tokenizer compression is only a proxy; model quality should be compared through `val_bpb`, sample quality, and downstream Chinese evaluations.
