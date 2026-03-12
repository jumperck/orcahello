"""
CLI entry point for finetuning OrcaHelloSRKWDetectorV1.

Usage:
    python -m model_v1.finetuning.finetune \
        --data-dir ./data \
        --config ./model/config.yaml \
        --output-dir ./finetune_output \
        --epochs 10 \
        --lr 1e-3 \
        --freeze-backbone

Expected data layout:
    data/
        positive/       # 1-min WAVs with SRKW calls
            call_001.wav
            call_002.wav
        negative/       # 1-min WAVs without SRKW calls
            noise_001.wav
            noise_002.wav

Outputs:
    finetune_output/
        best_model.pt   # State dict loadable via from_checkpoint()
        history.json    # Per-epoch training metrics
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..inference import OrcaHelloSRKWDetectorV1
from .dataset import SRKWFinetuneDataset
from .evaluate import evaluate_files, evaluate_segments
from .trainer import FinetuneConfig, finetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Finetune OrcaHello SRKW Detector V1 on new labeled audio data."
    )

    # Data
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory with positive/ and negative/ WAV subdirectories",
    )
    parser.add_argument(
        "--config", default="model/config.yaml",
        help="Path to model config YAML (default: model/config.yaml)",
    )

    # Model source (one of these)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint", default=None,
        help="Path to local .pt checkpoint file",
    )
    source.add_argument(
        "--hub-model", default="orcasound/orcahello-srkw-detector-v1",
        help="HuggingFace Hub model ID (default: orcasound/orcahello-srkw-detector-v1)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.15)

    # Backbone freezing
    parser.add_argument(
        "--freeze-backbone", action="store_true", default=True,
        help="Freeze ResNet50 backbone, only train head (default: True)",
    )
    parser.add_argument(
        "--no-freeze-backbone", dest="freeze_backbone", action="store_false",
        help="Unfreeze entire model for full finetuning",
    )

    # Augmentation
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Apply spectrogram augmentations during training (default: True)",
    )
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false",
    )

    # Output
    parser.add_argument("--output-dir", default="./finetune_output")
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip post-training evaluation",
    )

    args = parser.parse_args(argv)

    # ---- Load config ----
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # ---- Load pretrained model ----
    if args.checkpoint:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = OrcaHelloSRKWDetectorV1.from_checkpoint(args.checkpoint, config)
    else:
        logger.info(f"Loading model from HuggingFace Hub: {args.hub_model}")
        model = OrcaHelloSRKWDetectorV1.from_pretrained(args.hub_model, config=config)

    # ---- Build dataset ----
    logger.info(f"Loading data from {args.data_dir}")
    dataset = SRKWFinetuneDataset(
        data_dir=args.data_dir,
        config=config,
        augment=False,  # Augmentation applied only to train split below
        seed=args.seed,
    )
    logger.info(f"Total segments: {len(dataset)}, class counts: {dataset.class_counts()}")

    # Split by source file to prevent leakage
    train_ds, val_ds = dataset.train_val_split(
        val_fraction=args.val_fraction, seed=args.seed
    )
    logger.info(f"Train: {len(train_ds)} segments, Val: {len(val_ds)} segments")

    # Enable augmentation on train split
    if args.augment:
        train_ds.parent.augment = True

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ---- Finetune ----
    ft_config = FinetuneConfig(
        freeze_backbone=args.freeze_backbone,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=args.output_dir,
    )

    best_path, history = finetune(model, train_loader, val_loader, ft_config)
    logger.info(f"Best checkpoint saved to: {best_path}")

    # Save history
    history_path = Path(args.output_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")

    # ---- Post-training evaluation ----
    if not args.skip_eval:
        # Reload best checkpoint for clean eval
        model = OrcaHelloSRKWDetectorV1.from_checkpoint(best_path, config)

        logger.info("Running segment-level evaluation on validation set...")
        seg_result = evaluate_segments(model, val_loader)

        logger.info("Running file-level evaluation on full dataset...")
        file_result = evaluate_files(model, args.data_dir, config)

        # Save eval results
        eval_path = Path(args.output_dir) / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump({
                "segment_level": {
                    "accuracy": seg_result.accuracy,
                    "precision": seg_result.precision,
                    "recall": seg_result.recall,
                    "f1": seg_result.f1,
                    "confusion_matrix": seg_result.confusion_matrix,
                    "num_samples": seg_result.num_samples,
                },
                "file_level": {
                    "accuracy": file_result.accuracy,
                    "precision": file_result.precision,
                    "recall": file_result.recall,
                    "f1": file_result.f1,
                    "confusion_matrix": file_result.confusion_matrix,
                    "num_samples": file_result.num_samples,
                },
            }, f, indent=2)
        logger.info(f"Evaluation results saved to: {eval_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
