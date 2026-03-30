---
name: process-dataset
description: Run model_v1 inference on a dataset's audio and post-process results (add segment annotations to HF Dataset). Skips inference if summary.csv already exists. Use when the user wants to run inference on a dataset that has been downloaded.
disable-model-invocation: true
allowed-tools: Bash
---

Run model inference on a dataset's audio, then post-process: merge confidences and add segment annotations to the recording-level HF Dataset.

ARGUMENTS format: `<dataset_dir> [output_dir] [--build-segment-dataset] [--num-proc N]`
- `dataset_dir`: path to the dataset directory (e.g. `datasets/2024-07_2025-06--all`) — must contain `recording_dataset/` and `audio/` with downloaded FLAC files
- `output_dir`: optional inference results output dir (default: `inference_results/<dataset_name>`)
- `--build-segment-dataset`: optional flag to also produce a segment-level HF Dataset
- `--num-proc N`: optional number of parallel processes (default: cpu count)

Parse $ARGUMENTS: first token = dataset_dir, second token = output_dir (derive from dataset_dir name if omitted).

Steps:

**1. Run inference (skip if checkpoint exists)**

Derive `<DATASET_NAME>` from `basename(dataset_dir)` (e.g. `2020-07_2021-06--all`).
Set `<OUTPUT_DIR>` = `inference_results/<DATASET_NAME>` (or user-specified output_dir).

Check if `<OUTPUT_DIR>/summary.csv` already exists. If it does, log "Inference already complete, skipping to post-processing" and go to step 2.

Otherwise, use a timeout of at least 900000ms (15 minutes). Run mkdir in a separate Bash call first, then run inference and tee to log file in a second Bash call.

First Bash call (create output dir):
```bash
mkdir -p <OUTPUT_DIR>
```

Second Bash call (run inference, tee to log):
```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment && \
/Users/Akash/SideProjects/ai4orcas/orcahello/InferenceSystem/.venv/bin/python \
  ../InferenceSystem/scripts/run_inference.py \
  <dataset_dir>/audio \
  --output <OUTPUT_DIR> \
  2>&1 | tee <OUTPUT_DIR>/inference.log
```

**2. Post-process with process_dataset.py**

```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment && \
.venv/bin/python scripts/process_dataset.py \
  --dataset-dir <dataset_dir> \
  --inference-dir <OUTPUT_DIR> \
  [--build-segment-dataset] \
  [--num-proc N]
```

**3. Report**

Report:
- Whether inference was run or skipped (checkpoint)
- Inference output directory and number of files processed
- Number of recordings with updated confidence scores
- Whether segment annotations were added
- If --build-segment-dataset was used, number of segments in the segment dataset and stats on length distribution