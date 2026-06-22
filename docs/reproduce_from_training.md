# Reproduce Chinese BabyLM NanoChat From Training

This document is the intended handoff for organizers who need to reproduce the submitted model and final leaderboard scores from training, not only from the uploaded checkpoint.

## Scope

The reproduction path is:

1. Build the Python environment.
2. Download or locate the official `chinese-babylm-org/babylm-zho-100M` corpus.
3. Prepare nanochat parquet training shards.
4. Train the 32K Chinese tokenizer.
5. Train the d22 NanoChat base model from random initialization.
6. Export the step-10000 checkpoint to HuggingFace `trust_remote_code` CausalLM format.
7. Run the official final Chinese BabyLM pipeline and gather scores.

The one-command entry point is:

```bash
cd /mnt/proj/babyllm
bash scripts/reproduce_from_training.sh
```

The script is restartable: if data, tokenizer, checkpoints, or exported model already exist under the chosen `BASE_DIR`, it reuses them.

## Hardware And Runtime

The original final model was trained on one RTX 5090.

Approximate runtime on the original environment:

| Stage | Approximate time |
| --- | ---: |
| tokenizer/data preparation | minutes to tens of minutes |
| d22 base training, 10000 steps | about 6.1 hours |
| HF export | minutes |
| final evaluation | task dependent; NLU is parallelized over 2 GPUs when available |

For RTX 5090 / Blackwell, the final evaluation script automatically replaces the official pipeline's default `torch==2.7.0` CUDA 12.6 wheel with `torch==2.7.0+cu128`, because the CUDA 12.6 wheel does not include `sm_120` kernels.

## Data

The model uses the official corpus option:

```text
chinese-babylm-org/babylm-zho-100M
```

Default local path:

```text
data/babylm-zho-100M
```

If the path does not exist, `scripts/reproduce_from_training.sh` downloads it through Hugging Face `datasets` and saves it locally.

The prepared nanochat split is generated with:

```bash
python scripts/prepare_chinese_babylm.py \
  --data-dir data/babylm-zho-100M \
  --base-dir "$BASE_DIR" \
  --num-train-shards 16 \
  --val-fraction 0.005 \
  --seed 42 \
  --overwrite
```

The original split had 182,680 train rows and 918 validation rows.

## Tokenizer

The tokenizer is trained from the prepared training split:

```bash
python -m scripts.tok_train \
  --vocab-size 32768 \
  --max-chars 200000000 \
  --doc-cap 10000
```

Expected tokenizer artifacts:

```text
$BASE_DIR/tokenizer/tokenizer.pkl
$BASE_DIR/tokenizer/token_bytes.pt
```

## Base Model Training

The submitted model is a base causal language model, trained from random initialization. It is not an SFT/chat model.

Architecture and training configuration:

| Field | Value |
| --- | ---: |
| depth | 22 |
| hidden size | 1408 |
| attention heads | 22 |
| head dim | 64 |
| vocab size | 32768 |
| context length | 1024 |
| window pattern | `L` |
| total batch size | 65536 tokens |
| device batch size | 4 |
| steps | 10000 |
| trained tokens | 655,360,000 |
| seed | 42, set inside `nanochat.common.compute_init` |

Training command used by the script:

```bash
python -m scripts.base_train \
  --depth 22 \
  --aspect-ratio 64 \
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

Expected final checkpoint:

```text
$BASE_DIR/base_checkpoints/zh-d22-2gpu/model_010000.pt
$BASE_DIR/base_checkpoints/zh-d22-2gpu/meta_010000.json
```

Note: GPU training is not guaranteed bitwise deterministic across hardware, driver, PyTorch, and kernel versions. The configuration and seed are fixed, but small score variation is possible when retraining from random initialization.

## HuggingFace Export

The final leaderboard model exposes layer 8 for cognitive representation tasks. This is important for reproducing the reported fMRI scores.

Export command:

```bash
python -m scripts.export_nanochat_hf \
  --checkpoint-dir "$BASE_DIR/base_checkpoints/zh-d22-2gpu" \
  --tokenizer-dir "$BASE_DIR/tokenizer" \
  --step 10000 \
  --representation-layer 8 \
  --representation-hidden-states-mode selected_only \
  --output-dir "$BASE_DIR/hf_models/zh-d22-step10000-layer08-selected"
```

The exported model is loaded by the official pipeline with:

```python
AutoTokenizer.from_pretrained(path, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
AutoModel.from_pretrained(path, trust_remote_code=True)
```

## Final Evaluation

The final evaluation uses the official final pipeline:

```text
https://github.com/chinese-babylm/chinese-babylm-pipeline-final
```

Config in this repository:

```text
eval_configs/config_final.yaml
```

Run directly from a trained/exported model:

```bash
PROJECT_DIR=/mnt/proj/babyllm \
MODEL_DIR=/mnt/proj/babyllm/chinese_cache_reproduce/hf_models/zh-d22-step10000-layer08-selected \
RESULTS_DIR=/mnt/proj/babyllm/chinese_cache_reproduce/final_eval_results \
RESULT_JSON=/mnt/proj/babyllm/chinese_cache_reproduce/chinesebabylm_2026_final_results.json \
bash scripts/run_final_eval.sh
```

The script first evaluates zero-shot and cognitive tasks, then parallelizes NLU fine-tuning tasks across GPU 0 and GPU 1 when `PARALLEL_EVAL=1`.

## Submitted Final Scores

These are the scores from the submitted frozen checkpoint and final pipeline run:

| Task | Score |
| --- | ---: |
| zhoblimp | 68.06 |
| xcomps_zh | 54.46 |
| hanzi_structure | 51.85 |
| hanzi_pinyin | 49.35 |
| hanzi_structure_hidden | 51.25 |
| hanzi_pinyin_hidden | 49.05 |
| word_fmri | 56.31 |
| fmri | 11.62 |
| afqmc | 68.95 |
| ocnli | 65.36 |
| tnews | 53.44 |
| cluewsc2020 | 63.16 |
| c3 | 29.01 |
| diagnostic_nli | 52.54 |

Local archived final result JSON:

```text
chinese_cache/final_eval_artifacts/chinesebabylm_2026_final_results.json
```

SHA-256:

```text
b0a402ff1a3c70f7cf796671c4dc3d6c1c66ba0b1d5708d668b6459db73410e6
```

## Useful Overrides

The reproduction script accepts environment overrides:

```bash
BASE_DIR=/mnt/proj/repro_cache \
DATA_DIR=/mnt/proj/data/babylm-zho-100M \
CUDA_VISIBLE_DEVICES=0 \
PARALLEL_EVAL=0 \
bash scripts/reproduce_from_training.sh
```

Set `PARALLEL_EVAL=0` if only one GPU is available for final evaluation.
