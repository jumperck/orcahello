---
name: process-dataset
description: Run model_v1 inference on a downloaded dataset and post-process results (merge confidences + segment-level CSV). Skips inference if summary.csv already exists. Use when the user wants to run inference on a dataset that has been downloaded.
disable-model-invocation: true
allowed-tools: Bash
---

Run model inference on a dataset audio directory, then post-process: merge confidences into `*--complete.csv` and produce a `*--complete-segmented.csv`.

ARGUMENTS format: `<dataset_dir> [output_dir]`
- `dataset_dir`: path to the dataset directory (e.g. `datasets/2024-07_2025-06--all`) — must contain downloaded audio under `{dataset_stem}/YYYY-MM/audio/`
- `output_dir`: optional inference results output dir (default: `inference_results/<dataset_name>`)

Parse $ARGUMENTS: first token = dataset_dir, second token = output_dir (derive from dataset_dir name if omitted).

The audio lives at `<dataset_dir>/<dataset_stem>/` where `<dataset_stem>` is the directory name under `<dataset_dir>` that contains `YYYY-MM/audio/` subdirectories. Find it by listing subdirectories.

Steps:

**1. Run inference (skip if checkpoint exists)**

Check if `<OUTPUT_DIR>/summary.csv` already exists. If it does, log "Inference already complete, skipping to post-processing" and go to step 2.

Otherwise, use a timeout of at least 900000ms (15 minutes). Run mkdir in a separate Bash call first, then run inference and tee to log file in a second Bash call.

First Bash call (create output dir):
```bash
mkdir -p inference_results/<DATASET_NAME>
```

Second Bash call (run inference, tee to log):
```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment && \
/Users/Akash/SideProjects/ai4orcas/orcahello/InferenceSystem/.venv/bin/python \
  ../InferenceSystem/scripts/run_inference.py \
  <AUDIO_SUBDIR> \
  --output <OUTPUT_DIR> \
  2>&1 | tee <OUTPUT_DIR>/inference.log
```

Where:
- `<DATASET_NAME>` = `basename(dataset_dir)`, e.g. `2020-07_2021-06--all`
- `<AUDIO_SUBDIR>` = `<dataset_dir>/<DATASET_NAME>--complete`
- `<OUTPUT_DIR>` = `inference_results/<DATASET_NAME>` (or user-specified output_dir)

**2. Post-process with process_dataset.py**

Find the `*--complete.csv` in `<dataset_dir>`. Then run:

```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment && \
.venv/bin/python scripts/process_dataset.py \
  --complete-csv <dataset_dir>/<COMPLETE_CSV> \
  --inference-dir <OUTPUT_DIR>
```

**3. Write summary markdown**

After post-processing, write a summary markdown file next to the segmented CSV (`<dataset_dir>/<DATASET_NAME>--summary.md`):

```bash
cd /Users/Akash/SideProjects/ai4orcas/orcahello/ModelDevelopment && \
.venv/bin/python -c "
import pandas as pd
from pathlib import Path

seg_csv = Path('<dataset_dir>/<DATASET_NAME>--complete-segmented.csv')
complete_csv = Path('<dataset_dir>/<COMPLETE_CSV>')
df = pd.read_csv(seg_csv)
cdf = pd.read_csv(complete_csv)
df['duration_s'] = df['segment_end_s'] - df['segment_start_s']

lines = ['# Dataset Summary: <DATASET_NAME>', '']
lines.append('## Complete CSV')
n_pos = (cdf['binary_label'] == 1).sum()
n_neg = (cdf['binary_label'] == 0).sum()
lines.append(f'- **Total detections**: {len(cdf)} ({n_pos} positive, {n_neg} negative)')
has_conf = cdf['global_confidence'].notna().sum()
lines.append(f'- **global_confidence filled**: {has_conf}/{len(cdf)}')
lines.append('')
lines.append('## Segmented CSV')
lines.append('')
lines.append('| Label | Segments | Detections | Total Duration |')
lines.append('|---|---|---|---|')
for label in sorted(df['segment_binary_label'].unique()):
    g = df[df['segment_binary_label'] == label]
    total_s = g['duration_s'].sum()
    label_name = 'Positive' if label == 1 else 'Negative'
    lines.append(f'| **{label_name} ({label})** | {len(g):,} | {g.detection_id.nunique():,} | {total_s/60:.1f} min ({total_s/3600:.1f} hr) |')
total_s = df['duration_s'].sum()
lines.append(f'| **Total** | {len(df):,} | {df.detection_id.nunique():,} | {total_s/60:.1f} min ({total_s/3600:.1f} hr) |')

out = seg_csv.with_name(seg_csv.name.replace('--complete-segmented.csv', '--summary.md'))
out.write_text('\n'.join(lines) + '\n')
print(f'Wrote {out}')
"
```

**4. Report**

Report:
- Whether inference was run or skipped (checkpoint)
- Inference output directory and number of files processed
- How many rows in the complete CSV now have `global_confidence` filled vs total rows
- Path to the segmented CSV and number of segment rows written
- Path to the summary markdown
