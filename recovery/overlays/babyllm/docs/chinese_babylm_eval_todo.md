# Chinese BabyLM Eval Pipeline TODO

Date: 2026-05-23

Goal: make our nanochat Chinese BabyLLM checkpoints evaluable with `SiyuanSong2004/chinese-babylm-eval-pipeline`, starting from the final d22 base checkpoint:

```text
/home/trevor/babyllm/chinese_cache/base_checkpoints/zh-d22-2gpu/model_005000.pt
```

## Target Outcome

- Run the Chinese BabyLM eval pipeline against our final nanochat base checkpoint.
- Support the official zero-shot, CogBench, and fine-tune task families through a nanochat backend adapter.
- Document exact commands and outputs for repeatable evaluation.

## Current Model

```text
tag: zh-d22-2gpu
step: 5000
type: nanochat causal LM
sequence_len: 1024
vocab_size: 32768
n_layer: 22
n_head: 22
n_embd: 1408
window_pattern: L
```

## Work Plan

1. Clone and inspect eval pipeline
   - Clone `https://github.com/SiyuanSong2004/chinese-babylm-eval-pipeline`.
   - Read README, config examples, zero-shot code, finetune code, and CogBench model loading code.
   - Record expected model interface and output format.

2. Decide adapter strategy
   - Preferred: export nanochat checkpoint to a local HuggingFace-compatible causal LM directory.
   - Fallback: add a nanochat backend adapter inside the eval pipeline.
   - Decision criteria:
     - Zero-shot can read logits.
     - Fine-tuning can train/evaluate classification tasks.
     - CogBench can access hidden states if needed.

3. Implement the minimum viable path
   - Start with zero-shot because it is closest to causal LM logprob scoring.
   - Load `zh-d22-2gpu` step 5000.
   - Bridge tokenizer and model forward API.
   - Run one small smoke task before the full suite.

4. Run evaluations
   - Zero-shot: `zhoblimp`, `hanzi_structure`, `hanzi_pinyin`.
   - If model interface allows it, continue to fine-tuning tasks:
     - `afqmc`
     - `ocnli`
     - `tnews`
     - `cluewsc2020`
   - Treat CogBench as a separate milestone because it likely requires hidden-state extraction.

5. Save results
   - Store logs under `/home/trevor/babyllm/runs/logs/eval_pipeline/`.
   - Store generated configs under `/home/trevor/babyllm/eval_configs/`.
   - Store final reports under `/home/trevor/babyllm/chinese_cache/eval_results/`.

## Known Risks

- The official pipeline primarily expects HuggingFace/Transformers model paths; our path depends on the local `nanochat_causal` adapter.
- nanochat checkpoints are not HuggingFace format, so architecture changes in nanochat may require adapter updates.
- Fine-tune numbers from the short stage run are useful for smoke/full-pipeline validation, but not final official numbers.
- CogBench depends on hidden-state extraction from the nanochat model; rerun a smoke test after changing model internals.

## Acceptance Criteria

- The final d22 checkpoint can run the official zero-shot, CogBench, and fine-tune task families through the eval pipeline.
- The command is documented and reproducible from a clean shell on `s220`.
- Outputs are exported under `/home/trevor/babyllm/chinese_cache/eval_results/`.
- Any non-final settings, especially short fine-tune epochs, are explicitly recorded.

## Status Log

- Completed: cloned `/home/trevor/chinese-babylm-eval-pipeline` and inspected the zero-shot, fine-tune, and pipeline entry points.
- Completed: confirmed the official zero-shot path expects a Transformers-style model returning logits, plus a tokenizer that can identify completion spans.
- Completed: implemented a `nanochat_causal` backend adapter:
  - `/home/trevor/chinese-babylm-eval-pipeline/evaluation_pipeline/sentence_zero_shot/nanochat_backend.py`
  - Patched zero-shot, fine-tune, CogBench, and pipeline entry points to load nanochat checkpoints.
- Completed: enabled HuggingFace mirror access with `HF_ENDPOINT=https://hf-mirror.com`.
- Completed: downloaded CogBench data through the HF mirror.
- Completed: prepared CLUE fine-tune data from `clue/clue` parquet files because `load_dataset("clue", ...)` fails under the installed HF stack.
- Completed: ran full evaluation on final checkpoint `zh-d22-2gpu` step 5000.
- Completed: ran the final official fine-tune sweep with `batch_size: 32`, `max_epochs: 10`, `wsc_epochs: 30`, `sequence_length: 128`, `lr: 3e-5`, `seed: 42`.

## Zero-Shot Results

Model:

```text
model_name: nanochat-zh-d22-final
checkpoint: /home/trevor/babyllm/chinese_cache/base_checkpoints/zh-d22-2gpu/model_005000.pt
backend: nanochat_causal
batch_size: 16
temperature: 1.0
```

Results:

| Task | Split | Examples | Average accuracy |
| --- | --- | ---: | ---: |
| `zhoblimp` | `full_eval` | 29,100 minimal pairs | 64.28 |
| `hanzi_structure` | `full_eval` | 2,000 minimal pairs | 53.55 |
| `hanzi_pinyin` | `full_eval` | 1,000 minimal pairs | 39.40 |

Reports and predictions:

```text
/home/trevor/babyllm/chinese_cache/eval_results/nanochat-zh-d22-final/main/zero_shot/nanochat_causal/zhoblimp/zhoblimp/best_temperature_report.txt
/home/trevor/babyllm/chinese_cache/eval_results/nanochat-zh-d22-final/main/zero_shot/nanochat_causal/hanzi_structure/hanzi_structure/best_temperature_report.txt
/home/trevor/babyllm/chinese_cache/eval_results/nanochat-zh-d22-final/main/zero_shot/nanochat_causal/hanzi_pinyin/hanzi_pinyin/best_temperature_report.txt
```

Smoke result before the full run:

```text
hanzi_pinyin fast_eval, 100 examples: 42.00 accuracy
```

## Full Eval Results

Run date: 2026-05-23

Config:

```text
/home/trevor/babyllm/eval_configs/nanochat_zh_d22_full_eval.yaml
/home/trevor/chinese-babylm-eval-pipeline/config_nanochat_zh_d22_full_eval.yaml
```

Export:

```text
/home/trevor/babyllm/chinese_cache/eval_results/nanochat_zh_d22_full_eval_1epoch.json
```

Results shown in the pipeline leaderboard scale:

| Task | Metric | Score |
| --- | --- | ---: |
| `zhoblimp` | accuracy | 64.28 |
| `hanzi_structure` | accuracy | 53.55 |
| `hanzi_pinyin` | accuracy | 39.40 |
| `word_fmri` | mean | 55.66 |
| `fmri` | mean | 7.11 |
| `afqmc` | accuracy | 69.37 |
| `ocnli` | accuracy | 64.34 |
| `tnews` | accuracy | 54.07 |
| `cluewsc2020` | accuracy | 64.14 |

Raw exported JSON:

```json
{
  "zhoblimp": {
    "accuracy": 0.6428
  },
  "hanzi_structure": {
    "accuracy": 0.5355
  },
  "hanzi_pinyin": {
    "accuracy": 0.394
  },
  "word_fmri": {
    "mean": 0.5565538055162206
  },
  "fmri": {
    "mean": 0.07108854546544337
  },
  "afqmc": {
    "accuracy": 0.6936978683966636
  },
  "ocnli": {
    "accuracy": 0.6433898305084746
  },
  "tnews": {
    "accuracy": 0.5407
  },
  "cluewsc2020": {
    "accuracy": 0.6414473684210527
  }
}
```

Fine-tune note: this is a stage run with `max_epochs: 1` and `wsc_epochs: 3`, not the official longer default. Use it as a functional full-pipeline baseline before spending more GPU time on the final fine-tune sweep.

## Official Final Eval Results

Run date: 2026-05-24

Model:

```text
model_name: nanochat-zh-d22-final-official
checkpoint: /home/trevor/babyllm/chinese_cache/base_checkpoints/zh-d22-2gpu/model_005000.pt
backend: nanochat_causal
```

Config:

```text
/home/trevor/babyllm/eval_configs/nanochat_zh_d22_official_eval.yaml
/home/trevor/chinese-babylm-eval-pipeline/config_nanochat_zh_d22_official_eval.yaml
```

Export:

```text
/home/trevor/babyllm/chinese_cache/eval_results/nanochat_zh_d22_official_eval.json
```

Fine-tune hyperparameters:

```text
lr: 3e-5
batch_size: 32
max_epochs: 10
wsc_epochs: 30
sequence_length: 128
seed: 42
```

Results shown in the pipeline leaderboard scale:

| Task | Metric | Score |
| --- | --- | ---: |
| `zhoblimp` | accuracy | 64.28 |
| `hanzi_structure` | accuracy | 53.55 |
| `hanzi_pinyin` | accuracy | 39.40 |
| `word_fmri` | mean | 55.66 |
| `fmri` | mean | 7.11 |
| `afqmc` | accuracy | 68.88 |
| `ocnli` | accuracy | 62.68 |
| `tnews` | accuracy | 55.00 |
| `cluewsc2020` | accuracy | 62.17 |

Raw exported JSON:

```json
{
  "zhoblimp": {
    "accuracy": 0.6428
  },
  "hanzi_structure": {
    "accuracy": 0.5355
  },
  "hanzi_pinyin": {
    "accuracy": 0.39399999999999996
  },
  "word_fmri": {
    "mean": 0.5565538055162206
  },
  "fmri": {
    "mean": 0.07108854546544337
  },
  "afqmc": {
    "accuracy": 0.6888322520852641
  },
  "ocnli": {
    "accuracy": 0.6267796610169492
  },
  "tnews": {
    "accuracy": 0.55
  },
  "cluewsc2020": {
    "accuracy": 0.6217105263157895
  }
}
```

Run notes:

- Zero-shot and CogBench results were reused from the same final checkpoint because they do not depend on fine-tune hyperparameters.
- Fine-tune was rerun under the official longer defaults and saved under a separate model stem, so the earlier `nanochat-zh-d22-final` 1epoch baseline remains intact.
- The run used `CUDA_VISIBLE_DEVICES=0`; official `batch_size: 32` completed without OOM.

## Repro Command

Use this full-pipeline template on `s220`:

```bash
cd /home/trevor/chinese-babylm-eval-pipeline

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export NANOCHAT_REPO=/home/trevor/babyllm
export NANOCHAT_BASE_DIR=/home/trevor/babyllm/chinese_cache
export NANOCHAT_DTYPE=bfloat16
export NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS=1
export NANOCHAT_DISABLE_COMPILE=1
export NANOCHAT_MODEL_TAG=zh-d22-2gpu
export NANOCHAT_MODEL_STEP=5000

/home/trevor/babyllm/.venv/bin/python pipeline.py eval \
  --config config_nanochat_zh_d22_full_eval.yaml

/home/trevor/babyllm/.venv/bin/python pipeline.py gather \
  --config config_nanochat_zh_d22_full_eval.yaml \
  --results_dir /home/trevor/babyllm/chinese_cache/eval_results \
  --export /home/trevor/babyllm/chinese_cache/eval_results/nanochat_zh_d22_full_eval_1epoch.json
```

Use this official final full-score template on `s220`:

```bash
cd /home/trevor/chinese-babylm-eval-pipeline

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export NANOCHAT_REPO=/home/trevor/babyllm
export NANOCHAT_BASE_DIR=/home/trevor/babyllm/chinese_cache
export NANOCHAT_DTYPE=bfloat16
export NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS=1
export NANOCHAT_DISABLE_COMPILE=1
export NANOCHAT_MODEL_TAG=zh-d22-2gpu
export NANOCHAT_MODEL_STEP=5000

/home/trevor/babyllm/.venv/bin/python pipeline.py eval \
  --config config_nanochat_zh_d22_official_eval.yaml \
  --tasks afqmc ocnli tnews cluewsc2020 \
  --force-redo

/home/trevor/babyllm/.venv/bin/python pipeline.py gather \
  --config config_nanochat_zh_d22_official_eval.yaml \
  --results_dir /home/trevor/babyllm/chinese_cache/eval_results \
  --export /home/trevor/babyllm/chinese_cache/eval_results/nanochat_zh_d22_official_eval.json
```

Use this zero-shot single-task template on `s220`:

```bash
cd /home/trevor/chinese-babylm-eval-pipeline

HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
NANOCHAT_REPO=/home/trevor/babyllm \
NANOCHAT_BASE_DIR=/home/trevor/babyllm/chinese_cache \
NANOCHAT_DTYPE=bfloat16 \
NANOCHAT_DISABLE_EXPANDABLE_SEGMENTS=1 \
NANOCHAT_DISABLE_COMPILE=1 \
NANOCHAT_MODEL_TAG=zh-d22-2gpu \
NANOCHAT_MODEL_STEP=5000 \
/home/trevor/babyllm/.venv/bin/python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name nanochat-zh-d22-final \
  --backend nanochat_causal \
  --task zhoblimp \
  --data_path evaluation_data/full_eval/zhoblimp \
  --output_dir /home/trevor/babyllm/chinese_cache/eval_results \
  --batch_size 16 \
  --save_predictions
```

Swap `--task` and `--data_path` for:

```text
hanzi_structure -> evaluation_data/full_eval/hanzi_structure
hanzi_pinyin    -> evaluation_data/full_eval/hanzi_pinyin
```

## Current Limitations

- The adapter now covers zero-shot, fine-tune, and CogBench for this checkpoint.
- The `nanochat_zh_d22_full_eval_1epoch.json` fine-tune results are from the short stage run (`max_epochs: 1`, WSC `3`); use `nanochat_zh_d22_official_eval.json` for the final official score.
- CogBench uses hidden states captured from the nanochat model before `lm_head`; this is compatible with the current checkpoint but should be re-smoked if nanochat architecture code changes.
- Dataset refresh should keep `HF_ENDPOINT=https://hf-mirror.com` in the environment on `s220`.
