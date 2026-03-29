---
name: sample-dataset
description: Bias-sample from a complete CSV to produce a sampled dataset with hard + uniform examples. Use when the user wants to create a sampled subset for evaluation.
disable-model-invocation: true
allowed-tools: Bash
---

Bias-sample from a complete CSV to produce a sampled dataset.

ARGUMENTS format: `<complete_csv> [options]`
- `complete_csv`: path to a `*--complete.csv` from create_dataset.py
- options: any of `--target-positives N`, `--negative-ratio N`, `--seed N`

Parse $ARGUMENTS as: first token = complete_csv, remaining tokens passed through as options.

Run in the ModelDevelopment uv venv:

```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment

.venv/bin/python scripts/sample_dataset.py \
  <complete_csv> \
  [options]
```

Report the output file path and a brief summary of row counts by label and example type.
