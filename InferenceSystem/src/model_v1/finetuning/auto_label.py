"""
Auto-label utility: converts file-level positive/negative labels into
fine-grained time-span labels using the existing model's segment predictions.

For positive files:
    Run inference → find contiguous regions where confidence >= threshold →
    merge adjacent high-confidence segments into labeled spans.

For negative files:
    Single row spanning the entire file duration, labeled 0.

Output: CSV with columns [file, start_s, end_s, label, confidence]

Usage:
    python -m model_v1.finetuning.auto_label \
        --data-dir ./data \
        --config ./model/config.yaml \
        --threshold 0.5 \
        --output labels.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from ..inference import OrcaHelloSRKWDetectorV1
from ..types import DetectionResult, SegmentPrediction

logger = logging.getLogger(__name__)


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


def auto_label_file(
    model: OrcaHelloSRKWDetectorV1,
    wav_path: str,
    file_label: int,
    config: Dict,
    threshold: float = 0.5,
    gap_tolerance_s: float = 1.0,
) -> List[dict]:
    """
    Generate fine-grained time-span labels for a single file.

    Args:
        model: Pretrained OrcaHelloSRKWDetectorV1.
        wav_path: Path to WAV file.
        file_label: 1 (positive) or 0 (negative) — the file-level label.
        config: Inference config dict.
        threshold: Confidence threshold for positive spans.
        gap_tolerance_s: Max gap between segments to merge into one span.

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

    # Positive file: find contiguous high-confidence regions
    spans = _merge_contiguous_spans(
        result.segment_predictions, threshold, gap_tolerance_s
    )

    if not spans:
        # Model didn't find any confident regions despite file-level positive label.
        # Log a warning — this file may need manual review.
        logger.warning(
            f"{filename}: file labeled positive but no segments above "
            f"threshold {threshold} (max conf: "
            f"{max(s.confidence for s in result.segment_predictions):.3f})"
        )
        return []

    for span in spans:
        span["file"] = filename

    return spans


def auto_label_directory(
    model: OrcaHelloSRKWDetectorV1,
    data_dir: str,
    config: Dict,
    threshold: float = 0.5,
    gap_tolerance_s: float = 1.0,
) -> List[dict]:
    """
    Generate fine-grained labels for all files in a positive/negative directory.

    Args:
        model: Pretrained model.
        data_dir: Root with positive/ and negative/ subdirectories.
        config: Inference config dict.
        threshold: Confidence threshold.
        gap_tolerance_s: Max gap for merging spans.

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
                threshold, gap_tolerance_s,
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
                    "using the existing model's segment predictions."
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
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for positive segments (default: 0.5)",
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
        threshold=args.threshold,
        gap_tolerance_s=args.gap_tolerance,
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
