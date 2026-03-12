"""
Evaluation utilities for finetuned SRKW detector.

Computes both segment-level and file-level metrics since the model
operates on 4-second segments but labels are at the file level.

Inputs:
    model: OrcaHelloSRKWDetectorV1 (finetuned, in eval mode)
    data_loader: DataLoader of (spectrogram, label) pairs
    — or —
    wav_dir: Directory of labeled WAV files (positive/ negative/)

Outputs:
    EvalResult dataclass with precision, recall, F1, confusion matrix,
    and per-file aggregated predictions.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..inference import OrcaHelloSRKWDetectorV1

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation metrics container."""
    accuracy: float
    precision: float       # For positive class (SRKW detected)
    recall: float
    f1: float
    confusion_matrix: Dict  # {"tp": int, "fp": int, "tn": int, "fn": int}
    num_samples: int


def evaluate_segments(
    model: OrcaHelloSRKWDetectorV1,
    data_loader: DataLoader,
    threshold: float = 0.5,
) -> EvalResult:
    """
    Evaluate model on segment-level (spectrogram, label) pairs.

    Args:
        model: Finetuned model in eval mode.
        data_loader: Yields (spectrogram_batch, label_batch).
        threshold: Confidence threshold for positive prediction.

    Returns:
        EvalResult with segment-level metrics.
    """
    device = model._device
    dtype = model._dtype
    model.eval()

    tp = fp = tn = fn = 0

    with torch.no_grad():
        for specs, labels in data_loader:
            specs = specs.to(device=device, dtype=dtype)
            labels = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            confidences = model.predict_call(specs)
            preds = (confidences >= threshold).long()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    result = EvalResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        num_samples=total,
    )

    logger.info(
        f"Segment-level eval: acc={accuracy:.3f} prec={precision:.3f} "
        f"rec={recall:.3f} f1={f1:.3f} (n={total})"
    )
    return result


def evaluate_files(
    model: OrcaHelloSRKWDetectorV1,
    wav_dir: str,
    config: Dict,
    threshold: float = 0.6,
) -> EvalResult:
    """
    Evaluate model at the file level using detect_srkw_from_file().

    This runs the full inference pipeline (segmentation + aggregation)
    on each file, matching how the model is used in production.

    Args:
        model: Finetuned model.
        wav_dir: Directory with positive/ and negative/ subdirectories.
        config: Inference config dict (passed to detect_srkw_from_file).
        threshold: Global prediction threshold (overrides config if provided).

    Returns:
        EvalResult with file-level metrics.
    """
    root = Path(wav_dir)
    tp = fp = tn = fn = 0

    for subdir, true_label in [("positive", 1), ("negative", 0)]:
        folder = root / subdir
        if not folder.is_dir():
            continue

        for wav_path in sorted(folder.glob("*.wav")):
            result = model.detect_srkw_from_file(str(wav_path), config)

            # Use the global confidence against our threshold
            pred = 1 if result.global_confidence >= threshold else 0

            if pred == 1 and true_label == 1:
                tp += 1
            elif pred == 1 and true_label == 0:
                fp += 1
            elif pred == 0 and true_label == 0:
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    result = EvalResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        num_samples=total,
    )

    logger.info(
        f"File-level eval: acc={accuracy:.3f} prec={precision:.3f} "
        f"rec={recall:.3f} f1={f1:.3f} (n={total})"
    )
    return result
