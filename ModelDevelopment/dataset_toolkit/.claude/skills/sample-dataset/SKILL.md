---
name: sample-dataset
description: Bias-sample from a recording-level HF Dataset to produce a sampled subset with hard + uniform examples. Use when the user wants to create a sampled subset for evaluation.
disable-model-invocation: true
allowed-tools: Bash
---

Bias-sample from a recording-level HF Dataset to produce a sampled subset.

ARGUMENTS format: `<dataset_dir> [options]`
- `dataset_dir`: path to dataset directory containing `recording_dataset/` (e.g. `datasets/2025-07_2026-02--all`)
- options: any of `--target-positives N`, `--negative-ratio N`, `--seed N`

Parse $ARGUMENTS as: first token = dataset_dir, remaining tokens passed through as options.

Run in the ModelDevelopment uv venv:

```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment

.venv/bin/python scripts/sample_dataset.py \
  --dataset-dir <dataset_dir> \
  [options]
```

Report the output path (`sampled_dataset/` in the dataset dir) and a brief summary of row counts by label and example type.
