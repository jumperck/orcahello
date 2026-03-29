"""Post-inference processing: merge global_confidence into the complete CSV
and produce a segment-level CSV via auto_segment."""

import argparse
import logging
from pathlib import Path

from src.processing import build_segmented_csv, merge_confidences


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-inference processing: merge confidences and produce segment-level CSV.",
    )
    parser.add_argument(
        "--complete-csv", required=True, type=Path,
        help="Path to the *--complete.csv dataset file",
    )
    parser.add_argument(
        "--inference-dir", required=True, type=Path,
        help="Path to the inference results directory (must contain summary.csv)",
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

    complete_csv = args.complete_csv.resolve()
    inference_dir = args.inference_dir.resolve()

    # Step 1: merge global_confidence into complete CSV
    logger.info("Merging confidences from %s into %s", inference_dir, complete_csv)
    df = merge_confidences(complete_csv, inference_dir)

    # Step 2: build segmented CSV
    segmented_path = complete_csv.with_name(
        complete_csv.name.replace("--complete.csv", "--complete-segmented.csv")
    )
    logger.info("Building segmented CSV: %s", segmented_path)
    build_segmented_csv(
        df, inference_dir, segmented_path,
        min_threshold=args.min_threshold,
        gap_tolerance_s=args.gap_tolerance,
    )


if __name__ == "__main__":
    main()
