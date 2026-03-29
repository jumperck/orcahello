---
name: create-dataset
description: Create a recording-level HF Dataset for given months and location, with optional audio download. Use when the user wants to build or rebuild a dataset for model training/evaluation.
disable-model-invocation: true
allowed-tools: Bash
---

Create a recording-level HF Dataset from the OrcaHello detection logbook.

ARGUMENTS format: `<months> <location> [--download] [--export-csv] [--force]`
- months: single month (`2025-07`), range (`2025-07:2026-02`), or space-separated list
- location: hydrophone slug (e.g. `north-sjc`, `orcasound-lab`) or `all`
- `--download`: optional flag to also download audio files
- `--export-csv`: optional flag to also export a flat CSV
- `--force`: optional flag to rebuild even if recording_dataset/ already exists

Parse $ARGUMENTS as: first token = months, second token = location (default `all` if omitted). Flags can appear anywhere.

Run in the ModelDevelopment uv venv:

```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment

.venv/bin/python scripts/create_dataset.py \
  --months <months> \
  --location <location> \
  --logbook-dir /Users/Akash/SideProjects/ai4orcas/orcadata/orcareports/combined_logbook \
  --cache-dir /Users/Akash/SideProjects/ai4orcas/orcadata/orcareports/fetch_cache/orcahello \
  --output-dir datasets \
  [--download] [--export-csv] [--force]
```

Report the output directory, number of recordings in the HF Dataset, and download summary if --download was used.
