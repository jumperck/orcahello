"""
Auto-segment utility: binarize inference result JSONs into fine-grained
time-span segments.

Load inference JSON → compute a dynamic confidence threshold via Otsu's
histogram method → find contiguous regions above that threshold → merge
adjacent high-confidence segments into labeled spans.

Output: CSV with columns [file, start_s, end_s, confidence]

Usage:
    python auto_segment.py --json path/to/result.json
    python auto_segment.py --json-dir path/to/json_dir/ --output spans.csv
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _otsu_threshold(confidences: List[float], n_bins: int = 20) -> float:
    """Compute optimal binary threshold via Otsu's method (maximise between-class variance)."""
    if len(confidences) < 2:
        return 0.0

    counts, bin_edges = np.histogram(confidences, bins=n_bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return 0.0

    best_thresh = 0.0
    best_variance = -1.0

    cumulative_count = 0
    cumulative_sum = 0.0

    total_mean = float((counts * bin_centers).sum()) / total

    for i in range(1, n_bins):
        cumulative_count += counts[i - 1]
        cumulative_sum += counts[i - 1] * bin_centers[i - 1]

        w0 = cumulative_count / total
        w1 = 1.0 - w0
        if w0 == 0.0 or w1 == 0.0:
            continue

        mu0 = cumulative_sum / cumulative_count
        mu1 = (total_mean * total - cumulative_sum) / (total - cumulative_count)

        variance = w0 * w1 * (mu0 - mu1) ** 2
        if variance > best_variance:
            best_variance = variance
            best_thresh = float(bin_edges[i])

    return best_thresh


def compute_dynamic_threshold(
    confidences: List[float],
    min_threshold: float,
    n_bins: int = 20,
) -> float:
    """Return per-file confidence threshold via Otsu's method, floored at min_threshold."""
    otsu = _otsu_threshold(confidences, n_bins=n_bins)
    return max(otsu, min_threshold)


def _merge_contiguous_spans(
    segments: List[dict],
    threshold: float,
    gap_tolerance_s: float,
) -> List[dict]:
    """Find contiguous regions above threshold and merge them into spans."""
    spans = []
    current = None

    for seg in segments:
        seg_end = seg["start_time_s"] + seg["duration_s"]

        if seg["confidence"] >= threshold:
            if current is None:
                current = {
                    "start_s": seg["start_time_s"],
                    "end_s": seg_end,
                    "confidences": [seg["confidence"]],
                }
            else:
                gap = seg["start_time_s"] - current["end_s"]
                if gap <= gap_tolerance_s:
                    current["end_s"] = max(current["end_s"], seg_end)
                    current["confidences"].append(seg["confidence"])
                else:
                    spans.append(_finalize_span(current))
                    current = {
                        "start_s": seg["start_time_s"],
                        "end_s": seg_end,
                        "confidences": [seg["confidence"]],
                    }
        else:
            if current is not None:
                spans.append(_finalize_span(current))
                current = None

    if current is not None:
        spans.append(_finalize_span(current))

    return spans


def _finalize_span(span: dict) -> dict:
    confs = span["confidences"]
    return {
        "start_s": round(span["start_s"], 3),
        "end_s": round(span["end_s"], 3),
        "confidence": round(sum(confs) / len(confs), 4),
    }


def auto_segment_from_json(
    json_path: str,
    min_threshold: float = 0.1,
    gap_tolerance_s: float = 1.0,
    n_bins: int = 20,
) -> List[dict]:
    """Binarize a single inference result JSON into time-span segments.

    Returns list of dicts with keys: file, start_s, end_s, confidence.
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        result = json.load(f)

    filename = json_path.stem
    segments = result["segment_predictions"]

    all_confidences = [s["confidence"] for s in segments]
    threshold = compute_dynamic_threshold(all_confidences, min_threshold, n_bins)
    logger.info(
        "%s: threshold=%.4f (otsu=%.4f, min=%.4f)",
        filename, threshold,
        _otsu_threshold(all_confidences, n_bins), min_threshold,
    )

    spans = _merge_contiguous_spans(segments, threshold, gap_tolerance_s)

    if not spans:
        logger.info("%s: no segments above threshold %.4f", filename, threshold)
        return []

    for span in spans:
        span["file"] = filename

    return spans


def auto_segment_directory(
    json_dir: str,
    min_threshold: float = 0.1,
    gap_tolerance_s: float = 1.0,
    n_bins: int = 20,
) -> List[dict]:
    """Binarize all JSON files in a directory."""
    root = Path(json_dir)
    all_rows = []

    json_files = sorted(root.glob("*.json"))
    logger.info(f"Processing {len(json_files)} JSON files from {json_dir}")

    for json_path in json_files:
        rows = auto_segment_from_json(
            str(json_path), min_threshold, gap_tolerance_s, n_bins,
        )
        all_rows.extend(rows)

    return all_rows


def write_csv(rows: List[dict], output_path: str) -> None:
    fieldnames = ["file", "start_s", "end_s", "confidence"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} spans to {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Binarize inference result JSONs into time-span segments "
                    "using Otsu's dynamic thresholding."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", metavar="PATH", help="Single inference result JSON file")
    source.add_argument("--json-dir", metavar="DIR", help="Directory of inference result JSON files")

    parser.add_argument("--min-threshold", type=float, default=0.1,
                        help="Floor for the dynamic threshold (default: 0.1)")
    parser.add_argument("--n-bins", type=int, default=20,
                        help="Histogram bins for Otsu's method (default: 20)")
    parser.add_argument("--gap-tolerance", type=float, default=1.0,
                        help="Max gap in seconds to merge spans (default: 1.0)")
    parser.add_argument("--output", default="labels.csv",
                        help="Output CSV path (default: labels.csv)")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.json:
        rows = auto_segment_from_json(
            args.json,
            min_threshold=args.min_threshold,
            gap_tolerance_s=args.gap_tolerance,
            n_bins=args.n_bins,
        )
    else:
        rows = auto_segment_directory(
            args.json_dir,
            min_threshold=args.min_threshold,
            gap_tolerance_s=args.gap_tolerance,
            n_bins=args.n_bins,
        )

    write_csv(rows, args.output)
    logger.info(f"Summary: {len(rows)} spans")


if __name__ == "__main__":
    main()
