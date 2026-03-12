"""
Lightweight PyTorch training loop for finetuning OrcaHelloSRKWDetectorV1.

Design decisions:
    - Freeze ResNet50 backbone by default, only train the classification head.
      This is fastest and avoids catastrophic forgetting on small data.
    - Optional full-model unfreeze for larger datasets.
    - Uses AdamW + OneCycleLR (mirrors the original FastAI fit_one_cycle recipe).
    - Weighted cross-entropy to handle class imbalance (orca calls are rare).
    - Early stopping on validation loss.
    - Saves best checkpoint as a standard state_dict .pt file, directly
      loadable by OrcaHelloSRKWDetectorV1.from_checkpoint().

Inputs:
    model: OrcaHelloSRKWDetectorV1 instance (pretrained weights loaded)
    train_loader: DataLoader of (spectrogram, label) pairs
    val_loader: DataLoader of (spectrogram, label) pairs
    config: FinetuneConfig dataclass

Outputs:
    best_checkpoint_path: str — path to saved .pt state_dict
    training_history: List[dict] — per-epoch metrics
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from ..inference import OrcaHelloSRKWDetectorV1

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """All training hyperparameters in one place."""

    # Backbone freezing
    freeze_backbone: bool = True  # Only train head layers

    # Optimizer
    lr: float = 1e-3             # Peak learning rate (for OneCycleLR)
    weight_decay: float = 1e-2

    # Schedule
    epochs: int = 10
    batch_size: int = 32

    # Class weighting (auto-computed from data if None)
    class_weights: Optional[List[float]] = None

    # Early stopping
    patience: int = 3            # Epochs without val loss improvement
    min_delta: float = 1e-4      # Minimum improvement to count

    # Output
    output_dir: str = "./finetune_output"
    checkpoint_name: str = "best_model.pt"


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    lr: float


def _freeze_backbone(model: OrcaHelloSRKWDetectorV1) -> None:
    """
    Freeze all ResNet50 backbone parameters, leaving the classification
    head (model.model.fc) trainable.

    Frozen layers: conv1, bn1, layer1-4
    Trainable layers: avgpool (AdaptiveConcatPool2d), fc (head)
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the head (fc) and the concat pool
    for param in model.model.avgpool.parameters():
        param.requires_grad = True
    for param in model.model.fc.parameters():
        param.requires_grad = True


def _unfreeze_all(model: OrcaHelloSRKWDetectorV1) -> None:
    """Unfreeze all parameters for full-model finetuning."""
    for param in model.parameters():
        param.requires_grad = True


def _compute_class_weights(
    train_loader: DataLoader,
    num_classes: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training data.

    Args:
        train_loader: DataLoader yielding (spectrogram, label) batches
        num_classes: Number of classes (2 for binary)
        device: Target device for the weight tensor

    Returns:
        Tensor of shape (num_classes,) with weights inversely proportional
        to class frequency. Normalized so weights sum to num_classes.
    """
    counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for label in labels:
            counts[label] += 1

    # Inverse frequency, normalized
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights.to(device)


def finetune(
    model: OrcaHelloSRKWDetectorV1,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: FinetuneConfig,
) -> tuple[str, List[Dict]]:
    """
    Run the finetuning loop.

    Args:
        model: Pretrained OrcaHelloSRKWDetectorV1 (weights already loaded).
        train_loader: Training DataLoader of (Tensor[1,256,312], int) pairs.
        val_loader: Validation DataLoader.
        config: FinetuneConfig with hyperparameters.

    Returns:
        Tuple of:
            - best_checkpoint_path (str): Path to saved best model state_dict.
            - history (List[dict]): Per-epoch metrics dicts with keys:
              epoch, train_loss, val_loss, val_accuracy, lr.
    """
    device = model._device
    dtype = model._dtype

    # --- Freeze / unfreeze ---
    if config.freeze_backbone:
        _freeze_backbone(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Backbone frozen: {trainable:,}/{total:,} parameters trainable")
    else:
        _unfreeze_all(model)
        logger.info("Full model unfrozen for finetuning")

    # --- Loss function with class weighting ---
    if config.class_weights is not None:
        weights = torch.tensor(config.class_weights, device=device, dtype=torch.float32)
    else:
        weights = _compute_class_weights(train_loader, model.num_classes, device)
        logger.info(f"Auto-computed class weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)

    # --- Optimizer (only trainable params) ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)

    # --- LR scheduler: OneCycleLR matches original FastAI fit_one_cycle ---
    total_steps = config.epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=0.3,       # 30% warmup (FastAI default)
        anneal_strategy="cos",
    )

    # --- Output directory ---
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(output_dir / config.checkpoint_name)

    # --- Training loop ---
    history: List[Dict] = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for specs, labels in train_loader:
            specs = specs.to(device=device, dtype=dtype)
            labels = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits.float(), labels)  # .float() for stable CE with fp16 logits
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device=device, dtype=dtype)
                labels = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)

                logits = model(specs)
                loss = criterion(logits.float(), labels)
                val_loss_sum += loss.item()

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        val_accuracy = val_correct / max(val_total, 1)
        current_lr = scheduler.get_last_lr()[0]

        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "lr": current_lr,
        }
        history.append(metrics)

        logger.info(
            f"Epoch {epoch+1}/{config.epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_acc={val_accuracy:.3f}  "
            f"lr={current_lr:.2e}"
        )

        # ---- Checkpointing + early stopping ----
        if avg_val_loss < best_val_loss - config.min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            logger.info(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {config.patience} epochs)")
                break

    return best_path, history
