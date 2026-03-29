"""Post-inference processing: add segment annotations and update confidence
scores in a recording-level HF Dataset."""

import argparse
import logging
from pathlib import Path

from datasets import load_from_disk

from src.hf_dataset import add_segment_annotations, build_segment_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-inference processing: merge confidences and add segment annotations.",
    )
    parser.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="Path to the dataset directory (must contain recording_dataset/)",
    )
    parser.add_argument(
        "--inference-dir", required=True, type=Path,
        help="Path to the inference results directory (must contain summary.csv)",
    )
    parser.add_argument(
        "--build-segment-dataset", action="store_true",
        help="Also build a segment-level HF Dataset from the annotated recordings",
    )
    parser.add_argument(
        "--max-segment-s", type=float, default=10.0,
        help="Max segment duration for segment dataset (default: 10.0)",
    )
    parser.add_argument(
        "--min-threshold", type=float, default=0.1,
        help="Safety floor for dynamic per-file confidence threshold (default: 0.1)",
    )
    parser.add_argument(
        "--gap-tolerance", type=float, default=1.0,
        help="Max gap in seconds between segments to merge (default: 1.0)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    dataset_dir = args.dataset_dir.resolve()
    inference_dir = args.inference_dir.resolve()
    recording_path = dataset_dir / "recording_dataset"

    if not recording_path.exists():
        logger.error("No recording_dataset/ found in %s", dataset_dir)
        raise SystemExit(1)

    # Load recording dataset
    logger.info("Loading recording dataset from %s", recording_path)
    dataset = load_from_disk(str(recording_path))
    logger.info("Loaded %d recordings", len(dataset))

    # Add segment annotations and update confidence scores
    logger.info("Adding segment annotations from %s", inference_dir)
    dataset = add_segment_annotations(
        dataset,
        inference_dir,
        min_threshold=args.min_threshold,
        gap_tolerance_s=args.gap_tolerance,
    )

    # Save back in-place
    dataset.save_to_disk(str(recording_path))
    logger.info("Saved updated recording dataset to %s", recording_path)

    # Optionally build segment-level dataset
    if args.build_segment_dataset:
        audio_dir = dataset_dir / "audio"
        segment_path = dataset_dir / "segment_dataset"

        logger.info(
            "Building segment dataset (max_segment_s=%.1f)...",
            args.max_segment_s,
        )
        seg_ds = build_segment_dataset(
            dataset,
            audio_dir=audio_dir if audio_dir.exists() else None,
            max_segment_s=args.max_segment_s,
        )
        seg_ds.save_to_disk(str(segment_path))
        logger.info("Saved segment dataset to %s (%d segments)", segment_path, len(seg_ds))


if __name__ == "__main__":
    main()
