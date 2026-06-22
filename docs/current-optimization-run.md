# Current Optimization Run

Date: 2026-06-02

## Goal

Optimize the current Chinese BabyLM nanochat setup using:

- checkpoint selection on the existing d22 base model
- cognitive representation layer selection without modifying the official eval pipeline
- cleaned-data retraining of a smaller decoder model

## Existing d22 Model

- Model tag: `zh-d22-2gpu`
- Final checkpoint: step 10000
- Architecture: decoder-only NanoChat CausalLM
- Layers: 22
- Hidden size: 1408
- Attention heads: 22
- Context length: 1024
- Vocabulary size: 32768
- Total parameters: 1,123,158,942
- Training tokens: 655,360,000

Official open eval score for step10000:

| Task | Score |
|---|---:|
| zhoblimp | 67.45 |
| hanzi_structure | 51.85 |
| hanzi_pinyin | 41.60 |
| word_fmri | 55.54 |
| fmri | 8.02 |
| afqmc | 68.61 |
| ocnli | 55.59 |
| tnews | 51.31 |
| cluewsc2020 | 59.54 |
| mean | 51.06 |

## Export Changes

`scripts/export_nanochat_hf.py` now supports:

- `--representation-layer`
- `--representation-hidden-states-mode all|selected_only|selected_last`

For official CogBench layer sweeps, exported models use `selected_only`, so
`outputs.hidden_states[-1]` maps to the selected layer while keeping the official
pipeline code unchanged.

Exported d22 variants:

- `zh-d22-step01000`
- `zh-d22-step05000`
- `zh-d22-step10000`
- `zh-d22-step10000-layer04-selected`
- `zh-d22-step10000-layer08-selected`
- `zh-d22-step10000-layer12-selected`
- `zh-d22-step10000-layer16-selected`
- `zh-d22-step10000-layer20-selected`
- `zh-d22-step10000-final-selected`

## Clean Data

Clean source:

- `/mnt/proj/babyllm/chinese_cache/base_data_climbmix`

Clean output:

- `/mnt/proj/babyllm/chinese_cache_clean/base_data_climbmix`

Cleaning rules:

- NFKC normalization
- whitespace collapse
- document length between 20 and 20000 characters
- minimum CJK ratio 0.15
- maximum most-common-character ratio 0.35
- reject replacement-character corrupted text
- exact dedup on normalized casefold text

Clean stats:

| Item | Count |
|---|---:|
| total | 183,598 |
| filtered | 40,601 |
| duplicate | 3 |
| kept | 142,994 |
| train rows | 142,279 |
| validation rows | 715 |

## Smaller Decoder Training

Current run:

- Model tag: `zh-d12-clean-2gpu`
- Base dir: `/mnt/proj/babyllm/chinese_cache_clean`
- Layers: 12
- Hidden size: 768
- Attention heads: 12
- Context length: 1024
- Vocabulary size: 32768
- Total parameters: 286,262,162
- Total batch size: 65,536 tokens
- Device batch size: 8 per GPU
- GPUs: 2 x RTX 5090
- Iterations: 10,000
- Training tokens: 655,360,000
- Save every: 1,000 steps
- Training log: `/mnt/proj/babyllm/chinese_cache_clean/logs/zh-d12-clean-2gpu.train.log`

## Automation

Scripts:

- `scripts/train_zh_d12_clean.sh`
- `scripts/export_optimization_variants.sh`
- `scripts/run_current_model_optimization_eval.sh`
- `scripts/run_d12_clean_posttrain_eval.sh`
- `scripts/run_post_training_optimization_suite.sh`

Background jobs on s220:

- d12 clean training PID file: `/mnt/proj/babyllm/chinese_cache_clean/logs/zh-d12-clean-2gpu.pid`
- post-training eval suite PID file: `/mnt/proj/babyllm/chinese_cache_clean/logs/post_training_optimization_suite.pid`

The post-training suite waits for d12 clean training to finish, then runs:

1. d22 checkpoint zero-shot+cogbench sweep
2. d22 cognitive layer sweep
3. d12 clean final HF export
4. d12 clean official zero-shot+cogbench eval

## Results

### d22 Checkpoint Sweep

These results use the official evaluation pipeline with the HF-exported d22
checkpoints. Finetuning tasks were not rerun in this sweep.

| Model | zhoblimp | hanzi_structure | hanzi_pinyin | word_fmri | fmri |
|---|---:|---:|---:|---:|---:|
| zh-d22-step01000 | 76.72 | 52.15 | 26.00 | 55.86 | 10.12 |
| zh-d22-step05000 | 70.18 | 51.45 | 37.50 | 55.62 | 8.37 |
| zh-d22-step10000 | 67.45 | 51.85 | 41.60 | 55.54 | 8.02 |

Checkpoint selection has a real tradeoff:

- step01000 is best for ZhoBLiMP and fMRI.
- step10000 is best for Hanzi Pinyin.
- step05000 is a middle point but does not dominate either endpoint.

### d22 Cognitive Layer Sweep

These results keep the step10000 d22 weights fixed and change only the exported
`AutoModel` representation layer via `representation_hidden_states_mode=selected_only`.

| Model | word_fmri | fmri |
|---|---:|---:|
| zh-d22-step10000-final-selected | 55.54 | 8.02 |
| zh-d22-step10000-layer04-selected | 55.85 | 9.82 |
| zh-d22-step10000-layer08-selected | 55.80 | 10.66 |
| zh-d22-step10000-layer12-selected | 55.84 | 10.64 |
| zh-d22-step10000-layer16-selected | 55.82 | 10.50 |
| zh-d22-step10000-layer20-selected | 55.74 | 10.16 |

Best cognitive setting:

- `representation_layer=8` gives the best fMRI score: 10.66.
- `representation_layer=4` gives the best word_fmri score among fixed step10000 layer exports: 55.85.
- Layer 8 is the best practical single cognitive setting because it raises fMRI
  substantially while keeping word_fmri near the best layer.

### d12 Clean Model

The cleaned-data d12 model was trained from scratch for 10000 steps and exported
to HF format.

| Model | zhoblimp | hanzi_structure | hanzi_pinyin | word_fmri | fmri |
|---|---:|---:|---:|---:|---:|
| zh-d12-clean-step10000 | 66.91 | 49.45 | 35.90 | 55.67 | 7.47 |

The d12 clean run is useful as an ablation, but it is not better than d22:

- It is smaller and cheaper, but zero-shot and sentence-fMRI are worse.
- word_fmri is slightly above the d22 final layer baseline, but below d22
  selected-layer variants.
- The very low final training loss suggests this clean-data d12 run may be
  overtrained for the filtered corpus.

## Recommendation

For the current model family, the highest-value change is to keep the d22
step10000 weights and export/submit the model with `representation_layer=8` and
`representation_hidden_states_mode=selected_only` for cognitive evaluation. This
keeps the stronger Hanzi Pinyin checkpoint while improving fMRI from 8.02 to
10.66 in the open cognitive eval.
