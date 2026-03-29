# CLAUDE.md — ModelDevelopment

Guidance for Claude Code when working in this directory.

## Overview

This directory contains the pipeline for building labeled datasets of orca detection audio, running model inference, and evaluating model performance. It operates on top of the OrcaHello detection cache maintained by the parent `orcareports` pipeline.

## Setup

```bash
cd ModelDevelopment
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

All scripts should be run from this directory using `.venv/bin/python` or after activating the venv.

## Typical Workflow

### 1. Create a dataset (complete CSV)

```bash
python scripts/create_dataset.py \
  --months 2025-07:2025-09 \
  --location all \
  --logbook-dir /path/to/orcadata/orcareports/combined_logbook \
  --cache-dir /path/to/orcadata/orcareports/fetch_cache/orcahello \
  --output-dir datasets/
```

Produces `*--complete.csv` in `datasets/{months}--{location}/` — all detections for the period (cached; reused on re-runs).

Add `--download` to also download audio as FLAC into `{csv-stem}/{YYYY-MM}/audio/{detection_id}.flac`.

### 2. (Optional) Sample from complete CSV

```bash
python scripts/sample_dataset.py \
  datasets/2025-07_2025-09--all/2025-07_2025-09--all--complete.csv
```

Produces `*--sampled.csv` — bias-sampled subset (hard + uniform, per-location).

### 3. (Optional) Run inference

Use `../InferenceSystem/scripts/run_inference.py` on the downloaded audio directory:

```bash
python ../InferenceSystem/scripts/run_inference.py \
  datasets/2025-07_2025-09--all/2025-07_2025-09--all--complete/ \
  --output inference_results/2025-07_2025-09--all--v1/
```

This produces per-file JSON results and a `summary.csv`. See that script's docstring for full usage including `--reaggregate` mode.

### 4. (Optional) Post-process: merge confidences + segment

```bash
python scripts/process_dataset.py \
  --complete-csv datasets/2025-07_2025-09--all/2025-07_2025-09--all--complete.csv \
  --inference-dir inference_results/2025-07_2025-09--all--v1/
```

Merges `global_confidence` from inference results into the complete CSV and produces a `*--complete-segmented.csv` with segment-level rows.

### 5. Evaluate

```bash
python scripts/evaluate.py \
  --ground-truth detection_downloads/2025-11/ground_truth_labels.csv \
  --predictions inference_results/2025-11--v1.2/summary.csv \
  --output-dir inference_results/2025-11--v1.2/
```

Outputs `results.txt` (AUROC, operating points, hard examples) and `roc_curve.png`.

### 6. Finetune (WIP)

Auto-labeling (segment-level labels from 60s files) and finetuning are not yet implemented.

---

## Code Structure

```
ModelDevelopment/
├── src/                           # Shared modules
│   ├── models.py                  # Pydantic schemas (DetectionRecord, SegmentRecord) + format_df
│   ├── data_loading.py            # Logbook/cache loading, month expansion, build_complete_df
│   ├── sampling.py                # Bias-sampling logic (hard + uniform)
│   ├── download.py                # HTTP utils + download orchestration
│   ├── segmentation.py            # Otsu thresholding / auto-segment (library only)
│   └── processing.py              # Merge confidences + build segmented CSV
├── scripts/                       # CLI entry points
│   ├── create_dataset.py          # Build complete CSV + optional audio download
│   ├── sample_dataset.py          # Bias-sample from a complete CSV
│   ├── process_dataset.py         # Merge inference + segment
│   ├── evaluate.py                # ROC evaluation
│   └── download_from_cache.py     # Download from raw OrcaHello cache
├── pyproject.toml
└── CLAUDE.md
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/create_dataset.py` | Create complete dataset CSV from the detection logbook (+ optional `--download`) |
| `scripts/sample_dataset.py` | Bias-sample hard + uniform examples from a complete CSV |
| `scripts/process_dataset.py` | Post-inference: merge confidences into complete CSV + produce segment-level CSV |
| `scripts/evaluate.py` | Evaluate inference predictions against ground truth; outputs ROC + metrics |
| `scripts/download_from_cache.py` | Download audio + spectrograms from raw OrcaHello cache, organized by moderation category |

## Output Directory Layout

```
ModelDevelopment/
├── datasets/                  # Output from create_dataset.py
│   └── {months}--{location}/
│       ├── *--complete.csv    # All detections (cached)
│       ├── *--sampled.csv     # Sampled working dataset (from sample_dataset.py)
│       └── {csv-stem}/        # Audio downloads (if --download used)
│           └── YYYY-MM/audio/{detection_id}.flac
├── inference_results/         # Output from run_inference.py
│   └── {period}--{version}/
│       ├── summary.csv
│       ├── results.txt        # From evaluate.py
│       └── roc_curve.png
└── detection_downloads/       # Output from download_from_cache.py
    └── YYYY-MM/
        ├── positive/
        ├── false_positive/
        ├── unmoderated/
        └── ground_truth_labels.csv
```

## Key Data Paths

The scripts expect external data directories passed as arguments (not hardcoded):

- `--logbook-dir`: `combined_logbook/` from the orcareports pipeline — contains `detections/all_detections.csv` and `hourly_events/all_hourly_events.csv`
- `--cache-dir`: `fetch_cache/orcahello/` from the orcareports pipeline — contains `{YYYY-MM}/raw_detections.json` with audio/spectrogram URIs

## Sampling Strategy (`scripts/sample_dataset.py`)

Sampling is per-location, then concatenated when `--location all`:

- **Positives:** Up to `--target-positives` (default 50), split 1:1 hard/uniform. Hard = confidence in `[hard_pos_min_conf, hard_pos_max_conf]` (default 0.1–0.7).
- **Negatives:** `--negative-ratio` × effective positive count (default 2×), split hard/uniform. Hard = confidence > `--hard-neg-min-conf` (default 0.5).
- Uses OrcaHello's own confidence score (v0) for hard sampling by default. Run `process_dataset.py` first to use model inference confidence.

## Notes

- All download scripts are idempotent — existing files are skipped.
- The `complete.csv` is cached and reused across sampling runs. Delete it to force a refresh.
- `inference_results/` contains pre-existing results for model versions v0 and v1.2 across late 2025 – early 2026 months.
