---
name: summarize-dataset
description: Generate dataset summary stats (sample counts, audio duration, label distribution, confidence scores, location/month breakdowns) for an HF Dataset.
disable-model-invocation: true
allowed-tools: Bash
---

Generate summary statistics for an OrcaHello HF Dataset.

ARGUMENTS format: `<dataset-path>`
- dataset-path: path to an HF Dataset directory (e.g. `datasets/2020-07_2021-06--all/recording_dataset` or `datasets/2025-07_2026-02--all/segment_dataset`). The script auto-detects whether it's a recording or segment dataset.

If no dataset-path is provided in $ARGUMENTS, list available HF datasets under `ModelDevelopment/datasets/` (look for `recording_dataset/` and `segment_dataset/` subdirectories) and ask the user which one to use.

Run in the ModelDevelopment dataset_toolkit venv:

```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment

dataset_toolkit/.venv/bin/python dataset_toolkit/scripts/summarize_dataset.py \
  --dataset-dir <dataset-path>
```

Report the generated summary stats to the user.
