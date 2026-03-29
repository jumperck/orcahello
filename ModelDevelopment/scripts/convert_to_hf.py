"""Convert existing CSV datasets to HF Dataset format.

Reads a complete CSV (from the old create_dataset.py) and produces a
recording-level HF Dataset. If a segmented CSV is also provided, populates
segment_annotations.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from dataset_toolkit.hf_dataset import RECORDING_FEATURES, build_recording_dataset, build_segment_dataset
from dataset_toolkit.models import LABEL_TO_TAG, LINK_TEMPLATE

logger = logging.getLogger(__name__)


def _load_segment_annotations(segmented_csv: Path) -> dict[str, list[dict]]:
    """Load segment annotations grouped by detection_id from a segmented CSV."""
    df = pd.read_csv(segmented_csv)
    annotations: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        det_id = row["detection_id"]
        label = int(row["segment_binary_label"])
        tag = LABEL_TO_TAG[label]
        ann = {
            "start": float(row["segment_start_s"]),
            "end": float(row["segment_end_s"]),
            "tag": tag,
        }
        annotations.setdefault(det_id, []).append(ann)
    return annotations


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing CSV datasets to HF Dataset format."
    )
    parser.add_argument(
        "--complete-csv", required=True, type=Path,
        help="Path to the *--complete.csv dataset file",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None,
        help="Directory with downloaded audio (layout: {YYYY-MM}/audio/{id}.flac)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: same directory as the CSV)",
    )
    parser.add_argument(
        "--segmented-csv", type=Path, default=None,
        help="Path to *--complete-segmented.csv to populate segment_annotations",
    )
    parser.add_argument(
        "--build-segment-dataset", action="store_true",
        help="Also build a segment-level HF Dataset (requires --segmented-csv)",
    )
    parser.add_argument(
        "--max-segment-s", type=float, default=10.0,
        help="Max segment duration for segment dataset (default: 10.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    complete_csv = args.complete_csv.resolve()
    output_dir = (args.output_dir or complete_csv.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV into DataFrame
    logger.info("Loading %s", complete_csv)
    df = pd.read_csv(complete_csv, dtype=str)
    df["binary_label"] = pd.to_numeric(df["binary_label"])
    df["global_confidence"] = pd.to_numeric(df["global_confidence"], errors="coerce")
    logger.info("Loaded %d rows", len(df))

    # Build recording dataset
    dataset = build_recording_dataset(df, audio_dir=args.audio_dir)

    # Optionally add segment annotations from segmented CSV
    if args.segmented_csv:
        logger.info("Loading segment annotations from %s", args.segmented_csv)
        seg_annotations = _load_segment_annotations(args.segmented_csv)

        def _add_annotations(example):
            anns = seg_annotations.get(example["recording_id"], [])
            return {"segment_annotations": anns}

        dataset = dataset.map(_add_annotations)
        n_annotated = sum(
            1 for ex in dataset if ex["segment_annotations"]
        )
        logger.info("Added segment annotations to %d/%d recordings", n_annotated, len(dataset))

    # Save recording dataset
    recording_path = output_dir / "recording_dataset"
    dataset.save_to_disk(str(recording_path))
    logger.info("Saved recording dataset to %s (%d recordings)", recording_path, len(dataset))

    # Optionally build segment dataset
    if args.build_segment_dataset:
        if not args.segmented_csv:
            logger.warning("--build-segment-dataset requires --segmented-csv; skipping")
        else:
            seg_ds = build_segment_dataset(
                dataset,
                audio_dir=args.audio_dir,
                max_segment_s=args.max_segment_s,
            )
            segment_path = output_dir / "segment_dataset"
            seg_ds.save_to_disk(str(segment_path))
            logger.info("Saved segment dataset to %s (%d segments)", segment_path, len(seg_ds))


if __name__ == "__main__":
    main()
