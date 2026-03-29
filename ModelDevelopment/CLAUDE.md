# CLAUDE.md — ModelDevelopment

Guidance for Claude Code when working in this directory.

## Overview

This directory contains the pipeline for building labeled datasets of orca detection audio, running model inference, and evaluating model performance. It operates on top of the OrcaHello detection cache maintained by the parent `orcareports` pipeline.

Datasets are stored as **HuggingFace Datasets** (Arrow-backed) as the primary format, with optional CSV export for backward compatibility.

## Setup

```bash
cd ModelDevelopment
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

All scripts should be run from this directory using `.venv/bin/python` or after activating the venv.

## Typical Workflow

### 1. Create a dataset

```bash
python scripts/create_dataset.py \
  --months 2025-07:2025-09 \
  --location all \
  --logbook-dir /path/to/orcadata/orcareports/combined_logbook \
  --cache-dir /path/to/orcadata/orcareports/fetch_cache/orcahello \
  --output-dir datasets/
```

Produces `recording_dataset/` (HF Dataset) in `datasets/{months}--{location}/` — all detections for the period. Cached on disk; reused on re-runs (use `--force` to rebuild).

Add `--download` to also download audio as FLAC into `audio/{YYYY-MM}/audio/{detection_id}.flac` and populate the dataset's audio column.

Add `--export-csv` to also write a flat `*--complete.csv` for backward compatibility.

### 2. (Optional) Sample from dataset

```bash
python scripts/sample_dataset.py \
  --dataset-dir datasets/2025-07_2025-09--all/
```

Produces `sampled_dataset/` — bias-sampled subset (hard + uniform, per-location).

### 3. (Optional) Run inference

Use `../InferenceSystem/scripts/run_inference.py` on the downloaded audio directory:

```bash
python ../InferenceSystem/scripts/run_inference.py \
  datasets/2025-07_2025-09--all/audio/ \
  --output inference_results/2025-07_2025-09--all--v1/
```

This produces per-file JSON results and a `summary.csv`. See that script's docstring for full usage including `--reaggregate` mode.

### 4. (Optional) Post-process: add segment annotations

```bash
python scripts/process_dataset.py \
  --dataset-dir datasets/2025-07_2025-09--all/ \
  --inference-dir inference_results/2025-07_2025-09--all--v1/
```

Merges `global_confidence` from inference results and adds `segment_annotations` to the recording dataset in-place.

Add `--build-segment-dataset` to also produce a `segment_dataset/` with individual annotated segments.

### 5. Evaluate

```bash
python scripts/evaluate.py \
  --ground-truth detection_downloads/2025-11/ground_truth_labels.csv \
  --predictions inference_results/2025-11--v1.2/summary.csv \
  --output-dir inference_results/2025-11--v1.2/
```

Outputs `results.txt` (AUROC, operating points, hard examples) and `roc_curve.png`.

### 6. Convert existing CSV datasets to HF format

```bash
python scripts/convert_to_hf.py \
  --complete-csv datasets/2025-07_2026-02--all/2025-07_2026-02--all--complete.csv \
  --audio-dir datasets/2025-07_2026-02--all/audio/ \
  --segmented-csv datasets/2025-07_2026-02--all/2025-07_2026-02--all--complete-segmented.csv
```

### 7. Finetune (WIP)

Auto-labeling (segment-level labels from 60s files) and finetuning are not yet implemented.

---

## Code Structure

```
ModelDevelopment/
├── src/                           # Shared modules
│   ├── models.py                  # Pydantic schemas (RecordingRow, SegmentRow, Tag, etc.)
│   ├── hf_dataset.py              # HF Features definitions + dataset builders
│   ├── utils.py                   # Logbook/cache loading, month expansion, build_complete_df
│   ├── sampling.py                # Bias-sampling logic (hard + uniform)
│   ├── download.py                # HTTP utils + download orchestration
│   └── segmentation.py            # Otsu thresholding / auto-segment (library only)
├── scripts/                       # CLI entry points
│   ├── create_dataset.py          # Build recording-level HF Dataset + optional audio download
│   ├── sample_dataset.py          # Bias-sample from a recording dataset
│   ├── process_dataset.py         # Add segment annotations + update confidence scores
│   ├── evaluate.py                # ROC evaluation
│   ├── convert_to_hf.py           # Convert existing CSV datasets to HF format
│   └── download_from_cache.py     # Download from raw OrcaHello cache
├── pyproject.toml
└── CLAUDE.md
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/create_dataset.py` | Create recording-level HF Dataset from the detection logbook (+ optional `--download`, `--export-csv`) |
| `scripts/sample_dataset.py` | Bias-sample hard + uniform examples from a recording dataset |
| `scripts/process_dataset.py` | Post-inference: add segment annotations + update confidence in HF Dataset |
| `scripts/evaluate.py` | Evaluate inference predictions against ground truth; outputs ROC + metrics |
| `scripts/convert_to_hf.py` | Convert existing CSV datasets to HF Dataset format |
| `scripts/download_from_cache.py` | Download audio + spectrograms from raw OrcaHello cache, organized by moderation category |

## HF Dataset Schemas

### Recording-level (`recording_dataset/`)

| Column | Type | Description |
|--------|------|-------------|
| `audio` | `Audio()` | Full recording waveform (populated after download) |
| `recording_id` | `string` | Detection UUID |
| `tags` | `Sequence({tag, score})` | File-level tags (e.g. `srkw_positive` with confidence score) |
| `metadata` | `{location_slug, year_month_pacific, ...}` | OrcaHello recording metadata |
| `comment` | `string` | Moderator comment |
| `segment_annotations` | `Sequence({start, end, tag})` | Per-segment time-span labels (added by process_dataset) |

### Segment-level (`segment_dataset/`)

| Column | Type | Description |
|--------|------|-------------|
| `audio` | `Audio()` | Extracted segment waveform |
| `label` | `int64` | Binary label (0/1) |
| `tag` | `string` | e.g. `srkw_positive` |
| `source_id` | `string` | Recording ID this segment came from |
| `start_s` | `float32` | Start time in source recording |
| `end_s` | `float32` | End time in source recording |

## Output Directory Layout

```
ModelDevelopment/
├── datasets/                      # Output from create_dataset.py
│   └── {months}--{location}/
│       ├── recording_dataset/     # HF Dataset (primary)
│       ├── sampled_dataset/       # From sample_dataset.py
│       ├── segment_dataset/       # From process_dataset.py --build-segment-dataset
│       ├── audio/                 # Downloaded audio (if --download)
│       │   └── YYYY-MM/audio/{detection_id}.flac
│       └── *--complete.csv        # Only if --export-csv
├── inference_results/             # Output from run_inference.py
│   └── {period}--{version}/
│       ├── summary.csv
│       ├── results.txt            # From evaluate.py
│       └── roc_curve.png
└── detection_downloads/           # Output from download_from_cache.py
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
- The `recording_dataset/` is cached and reused across runs. Use `--force` to rebuild.
- `inference_results/` contains pre-existing results for model versions v0 and v1.2 across late 2025 – early 2026 months.
