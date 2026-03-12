"""
Auto-label utility: converts file-level positive/negative labels into
fine-grained time-span labels using the existing model's segment predictions.

For positive files:
    Run inference → compute a dynamic confidence threshold via Otsu's histogram
    method → find contiguous regions above that threshold → merge adjacent
    high-confidence segments into labeled spans.

For negative files:
    Single row spanning the entire file duration, labeled 0.

The dynamic threshold is computed per-file from the distribution of segment
confidence scores using Otsu's method, which maximises the between-class
variance of a binary "low confidence / high confidence" split.  A mandatory
minimum threshold (``--min-threshold``, default 0.1) acts as a safety floor
so that near-zero confidence segments are never accepted regardless of the
file's distribution.

Output: CSV with columns [file, start_s, end_s, label, confidence]

Usage:
    python -m model_v1.finetuning.auto_label \\
        --data-dir ./data \\
        --config ./model/config.yaml \\
        --min-threshold 0.1 \\
        --output labels.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from ..inference import OrcaHelloSRKWDetectorV1
from ..types import DetectionResult, SegmentPrediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic threshold via Otsu's method
# ---------------------------------------------------------------------------

def _otsu_threshold(confidences: List[float], n_bins: int = 50) -> float:
    """
    Compute the optimal binary threshold for a list of confidence scores using
    Otsu's method (maximise between-class variance).

    Args:
        confidences: List of confidence values in [0, 1].
        n_bins: Number of histogram bins over [0, 1].

    Returns:
        Threshold value in [0, 1].  Returns 0.0 if the list is empty or
        all values are identical (caller should then apply the min threshold).
    """
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
            best_thresh = float(bin_edges[i])  # use the left edge of the upper bin

    return best_thresh


def compute_dynamic_threshold(
    confidences: List[float],
    min_threshold: float,
    n_bins: int = 50,
) -> float:
    """
    Return a per-file confidence threshold derived from the confidence score
    distribution via Otsu's histogram method, floored at ``min_threshold``.

    Args:
        confidences: All segment confidence scores for one file.
        min_threshold: Hard lower bound – the returned threshold will be at
                       least this value regardless of the histogram result.
        n_bins: Number of bins used when building the histogram.

    Returns:
        Effective threshold for this file.
    """
    otsu = _otsu_threshold(confidences, n_bins=n_bins)
    effective = max(otsu, min_threshold)
    logger.debug(
        "Dynamic threshold: otsu=%.4f  min=%.4f  effective=%.4f",
        otsu, min_threshold, effective,
    )
    return effective


# ---------------------------------------------------------------------------
# Span merging
# ---------------------------------------------------------------------------

def _merge_contiguous_spans(
    segments: List[SegmentPrediction],
    threshold: float,
    gap_tolerance_s: float,
) -> List[dict]:
    """
    Find contiguous regions of high-confidence segments and merge them.

    A new span starts when confidence jumps above threshold. Adjacent
    above-threshold segments are merged if the gap between them is
    <= gap_tolerance_s (accounts for overlapping windows where gap is 0
    or small hop-induced gaps).

    Args:
        segments: Ordered list of SegmentPrediction (start_time_s, duration_s, confidence).
        threshold: Confidence threshold for a segment to be considered positive.
        gap_tolerance_s: Max gap in seconds between segments to still merge them
                         into one span. Default should match the inference hop.

    Returns:
        List of dicts: {"start_s", "end_s", "label", "confidence"}
        where confidence is the mean of merged segment confidences.
    """
    spans = []
    current_span = None

    for seg in segments:
        seg_end = seg.start_time_s + seg.duration_s

        if seg.confidence >= threshold:
            if current_span is None:
                # Start a new span
                current_span = {
                    "start_s": seg.start_time_s,
                    "end_s": seg_end,
                    "confidences": [seg.confidence],
                }
            else:
                gap = seg.start_time_s - current_span["end_s"]
                if gap <= gap_tolerance_s:
                    # Extend current span
                    current_span["end_s"] = max(current_span["end_s"], seg_end)
                    current_span["confidences"].append(seg.confidence)
                else:
                    # Finalize current span, start new one
                    spans.append(_finalize_span(current_span, label=1))
                    current_span = {
                        "start_s": seg.start_time_s,
                        "end_s": seg_end,
                        "confidences": [seg.confidence],
                    }
        else:
            if current_span is not None:
                spans.append(_finalize_span(current_span, label=1))
                current_span = None

    # Don't forget the last span
    if current_span is not None:
        spans.append(_finalize_span(current_span, label=1))

    return spans


def _finalize_span(span: dict, label: int) -> dict:
    """Convert a working span dict into the output format."""
    confs = span["confidences"]
    return {
        "start_s": round(span["start_s"], 3),
        "end_s": round(span["end_s"], 3),
        "label": label,
        "confidence": round(sum(confs) / len(confs), 4),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_label_file(
    model: OrcaHelloSRKWDetectorV1,
    wav_path: str,
    file_label: int,
    config: Dict,
    min_threshold: float = 0.1,
    gap_tolerance_s: float = 1.0,
    n_bins: int = 50,
) -> List[dict]:
    """
    Generate fine-grained time-span labels for a single file.

    For positive files the confidence threshold is determined dynamically per
    file using Otsu's histogram method on the segment confidence distribution,
    floored at ``min_threshold``.

    Args:
        model: Pretrained OrcaHelloSRKWDetectorV1.
        wav_path: Path to WAV file.
        file_label: 1 (positive) or 0 (negative) — the file-level label.
        config: Inference config dict.
        min_threshold: Hard lower bound on the dynamic threshold (safety floor).
        gap_tolerance_s: Max gap between segments to merge into one span.
        n_bins: Histogram bin count for Otsu thresholding.

    Returns:
        List of dicts with keys: file, start_s, end_s, label, confidence.
    """
    filename = Path(wav_path).name
    result: DetectionResult = model.detect_srkw_from_file(wav_path, config)

    if file_label == 0:
        # Negative file: single span covering the whole file
        return [{
            "file": filename,
            "start_s": 0.0,
            "end_s": round(result.metadata.file_duration_s, 3),
            "label": 0,
            "confidence": round(1.0 - result.global_confidence, 4),
        }]

    # Positive file: compute a dynamic threshold then find contiguous regions
    all_confidences = [s.confidence for s in result.segment_predictions]
    threshold = compute_dynamic_threshold(all_confidences, min_threshold, n_bins)
    logger.info(
        "%s: dynamic threshold=%.4f (min=%.4f, otsu=%.4f)",
        filename,
        threshold,
        min_threshold,
        _otsu_threshold(all_confidences, n_bins),
    )

    spans = _merge_contiguous_spans(
        result.segment_predictions, threshold, gap_tolerance_s
    )

    if not spans:
        # Model didn't find any confident regions despite file-level positive label.
        # Log a warning — this file may need manual review.
        logger.warning(
            "%s: file labeled positive but no segments above dynamic threshold "
            "%.4f (max conf: %.3f)",
            filename,
            threshold,
            max(all_confidences) if all_confidences else 0.0,
        )
        return []

    for span in spans:
        span["file"] = filename

    return spans


def auto_label_directory(
    model: OrcaHelloSRKWDetectorV1,
    data_dir: str,
    config: Dict,
    min_threshold: float = 0.1,
    gap_tolerance_s: float = 1.0,
    n_bins: int = 50,
) -> List[dict]:
    """
    Generate fine-grained labels for all files in a positive/negative directory.

    Args:
        model: Pretrained model.
        data_dir: Root with positive/ and negative/ subdirectories.
        config: Inference config dict.
        min_threshold: Safety floor for the dynamic per-file threshold.
        gap_tolerance_s: Max gap for merging spans.
        n_bins: Histogram bin count for Otsu thresholding.

    Returns:
        List of label dicts (file, start_s, end_s, label, confidence).
    """
    root = Path(data_dir)
    all_rows = []

    for subdir, file_label in [("positive", 1), ("negative", 0)]:
        folder = root / subdir
        if not folder.is_dir():
            continue

        wav_files = sorted(folder.glob("*.wav"))
        logger.info(f"Processing {len(wav_files)} files from {subdir}/")

        for wav_path in wav_files:
            rows = auto_label_file(
                model, str(wav_path), file_label, config,
                min_threshold, gap_tolerance_s, n_bins,
            )
            all_rows.extend(rows)

    return all_rows


def write_csv(rows: List[dict], output_path: str) -> None:
    """Write label rows to CSV."""
    fieldnames = ["file", "start_s", "end_s", "label", "confidence"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} label spans to {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Auto-label audio files with fine-grained time spans "
                    "using the existing model's segment predictions. "
                    "The confidence threshold is computed dynamically per file "
                    "via Otsu's histogram method."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory with positive/ and negative/ WAV subdirectories",
    )
    parser.add_argument(
        "--config", default="model/config.yaml",
        help="Path to model config YAML",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", default=None, help="Local .pt checkpoint")
    source.add_argument(
        "--hub-model", default="orcasound/orcahello-srkw-detector-v1",
        help="HuggingFace Hub model ID",
    )

    parser.add_argument(
        "--min-threshold", type=float, default=0.1,
        help="Safety floor for the dynamic per-file confidence threshold. "
             "The actual threshold is determined by Otsu's histogram method "
             "but will never fall below this value. (default: 0.1)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=50,
        help="Number of histogram bins used by Otsu's method (default: 50)",
    )
    parser.add_argument(
        "--gap-tolerance", type=float, default=1.0,
        help="Max gap in seconds between segments to merge into one span "
             "(default: 1.0)",
    )
    parser.add_argument(
        "--output", default="labels.csv",
        help="Output CSV path (default: labels.csv)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config + model
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        model = OrcaHelloSRKWDetectorV1.from_checkpoint(args.checkpoint, config)
    else:
        model = OrcaHelloSRKWDetectorV1.from_pretrained(args.hub_model, config=config)

    # Run auto-labeling
    rows = auto_label_directory(
        model, args.data_dir, config,
        min_threshold=args.min_threshold,
        gap_tolerance_s=args.gap_tolerance,
        n_bins=args.n_bins,
    )

    # Write output
    write_csv(rows, args.output)

    # Summary stats
    pos_spans = [r for r in rows if r["label"] == 1]
    neg_spans = [r for r in rows if r["label"] == 0]
    logger.info(
        f"Summary: {len(pos_spans)} positive spans, "
        f"{len(neg_spans)} negative file spans"
    )


if __name__ == "__main__":
    main()
