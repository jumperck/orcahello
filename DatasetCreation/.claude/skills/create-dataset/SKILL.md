---
name: create-dataset
description: Create an orca detection dataset (complete CSV) for given months and location, with optional audio download. Use when the user wants to build or rebuild a dataset for model training/evaluation.
disable-model-invocation: true
allowed-tools: Bash
---

Create an orca detection dataset.

ARGUMENTS format: `<months> <location> [--download]`
- months: single month (`2025-07`), range (`2025-07:2026-02`), or space-separated list
- location: hydrophone slug (e.g. `north-sjc`, `orcasound-lab`) or `all`
- `--download`: optional flag to also download audio files

Parse $ARGUMENTS as: first token = months, second token = location (default `all` if omitted). If `--download` appears anywhere, include the flag.

Derive the output dir from the months and location following the pattern `{months_normalized}--{location}` where months range `2025-07:2026-02` → `2025-07_2026-02` (colon replaced with underscore, no spaces).

Run in the DatasetCreation uv venv:

```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/DatasetCreation

MONTHS="<months>"
LOCATION="<location>"
MONTHS_NORM=$(echo "$MONTHS" | tr ':' '_' | tr ' ' '_')
OUTPUT_DIR="datasets/${MONTHS_NORM}--${LOCATION}"

.venv/bin/python scripts/create_dataset.py \
  --months "$MONTHS" \
  --location "$LOCATION" \
  --logbook-dir /Users/Akash/SideProjects/ai4orcas/orcadata/orcareports/combined_logbook \
  --cache-dir /Users/Akash/SideProjects/ai4orcas/orcadata/orcareports/fetch_cache/orcahello \
  --output-dir "$OUTPUT_DIR" \
  [--download]
```

Report the output files created (complete CSV, and download summary if --download was used) and a brief summary of row counts.
