# OrcaHello Dataset Creation

Pipeline for building labeled orca detection datasets, running model inference, and evaluating model performance.

## Setup

```bash
cd dataset_toolkit
uv sync
source .venv/bin/activate
```

**Prerequisites:** The OrcaHello fetch cache and combined logbook must be populated — these are produced by `fetch_orcahello.py` and `preprocess_detections.py` in the parent orcareports directory.

---

## Workflow

### 1. Create a dataset

Build a complete CSV of all detections for a time range:

```bash
python dataset_toolkit/scripts/create_dataset.py \
  --months 2025-07:2025-09 \
  --location all \
  --logbook-dir /path/to/combined_logbook \
  --cache-dir /path/to/fetch_cache/orcahello \
  --output-dir datasets/
```

**Options:**
- `--months` — single month (`2025-07`), range (`2025-07:2025-09`), or space-separated list
- `--location` — hydrophone slug (e.g. `north-sjc`) or `all`
- `--download` — also download audio as FLAC after creating the CSV
- `--workers INT` — parallel download threads (default: 8)

**Output** in `datasets/{months}--{location}/`:
- `*--complete.csv` — all detections (cached; reused on re-runs unless deleted)

### 2. Sample from complete CSV (optional)

Bias-sample a working subset from the complete CSV:

```bash
python dataset_toolkit/scripts/sample_dataset.py \
  datasets/2025-07_2025-09--all/2025-07_2025-09--all--complete.csv
```

**Options:** `--target-positives INT` (default: 50), `--negative-ratio FLOAT` (default: 2.0), `--hard-pos-max-conf`, `--hard-pos-min-conf`, `--hard-neg-min-conf`, `--seed`, `--output`

**Output:** `*--sampled.csv` — bias-sampled working dataset (hard + uniform examples)

### 3. Run inference (optional)

Run the OrcaHello model on the downloaded audio using `run_inference.py` from InferenceSystem:

```bash
python ../InferenceSystem/scripts/run_inference.py \
  datasets/2025-07_2025-09--all/2025-07_2025-09--all--complete \
  --output inference_results/2025-07_2025-09--all
```

Produces per-file JSON results and a `summary.csv`. Re-run aggregation without re-running the model:

```bash
python ../InferenceSystem/scripts/run_inference.py \
  inference_results/2025-07_2025-09--all/ \
  --reaggregate --config path/to/config.yaml
```

See `../InferenceSystem/scripts/run_inference.py` for full usage.

### 4. Post-process: merge confidences + segment (optional)

Merge inference confidences into the complete CSV and produce segment-level rows:

```bash
python dataset_toolkit/scripts/process_dataset.py \
  --complete-csv datasets/2025-07_2025-09--all/2025-07_2025-09--all--complete.csv \
  --inference-dir inference_results/2025-07_2025-09--all/
```

**Options:** `--min-threshold FLOAT` (default: 0.1), `--gap-tolerance FLOAT` (default: 1.0)

**Output:** `*--complete-segmented.csv` with segment-level rows.

### 5. Evaluate

Evaluate inference results against ground truth labels:

```bash
python dataset_toolkit/scripts/evaluate.py \
  --ground-truth datasets/2025-07_2025-09--all/2025-07_2025-09--all--sampled.csv \
  --predictions inference_results/2025-07_2025-09--all/summary.csv \
  --output-dir inference_results/2025-07_2025-09--all/
```

Outputs `results.txt` (AUROC, operating points, hard examples) and `roc_curve.png`.

### 6. Finetune (WIP)

Auto-labeling (fine-grained segment labels) and finetuning are not yet implemented.

---

## Output Structure

```
datasets/
└── {months}--{location}/
    ├── *--complete.csv          # All detections for the period
    ├── *--sampled.csv           # Sampled working dataset (from sample_dataset.py)
    └── {csv-stem}/              # Audio downloads (if --download used)
        ├── YYYY-MM/
        │   └── audio/
        │       └── {detection_id}.flac
        └── summary.txt

inference_results/
└── {period}--{version}/
    ├── summary.csv              # Per-file inference results
    ├── results.txt              # Evaluation metrics (if step 5 run)
    └── roc_curve.png
```

---

## Other Scripts

### Download from raw cache

Download audio and spectrograms directly from the OrcaHello cache, organized by moderation status. Useful for building datasets without going through `create_dataset.py`:

```bash
python dataset_toolkit/scripts/download_from_cache.py \
  --months 2025-01:2025-11 \
  --category positive --category false_positive \
  --output-dir detection_downloads/
```

Output per month:
```
YYYY-MM/
├── positive/          # reviewed=True, found='yes'
├── false_positive/    # reviewed=True, found='no'
├── unmoderated/       # not yet reviewed
├── unknown/
├── ground_truth_labels.csv
└── summary.txt
```

**Options:** `--months` (single, range, or list), `--no-wav`, `--no-spec`, `--category`, `--workers`, `--cache-dir`, `--output-dir`, `--dry-run`, `--verbose`
