"""
PyTorch Dataset for file-level labeled audio data.

New data arrives as 1-minute WAV files in two folders:
    positive/   — files containing SRKW calls
    negative/   — files without SRKW calls

Each file is segmented into overlapping 4-second chunks (matching the model's
input_pad_s). All segments from a file inherit the file-level label.

The existing audio_frontend.py pipeline handles:
    load → downmix → resample → mel spectrogram → standardize (pad/crop to 312 frames)

Inputs:
    data_dir: str — path with positive/ and negative/ subdirectories
    config: dict  — the standard model config.yaml (audio + spectrogram + model sections)

Outputs per __getitem__:
    spectrogram: Tensor (1, 256, 312) — model-ready mel spectrogram
    label: int (0 or 1)
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..audio_frontend import load_processed_waveform, prepare_waveform


class FileLabel:
    """Metadata for a single source file."""
    __slots__ = ("path", "label")

    def __init__(self, path: str, label: int):
        self.path = path
        self.label = label


def _discover_files(data_dir: str) -> List[FileLabel]:
    """
    Walk positive/ and negative/ subdirectories and collect WAV paths + labels.

    Args:
        data_dir: Root directory containing positive/ and negative/ folders.

    Returns:
        List of FileLabel(path, label) sorted by path for reproducibility.
    """
    root = Path(data_dir)
    files: List[FileLabel] = []

    for subdir, label in [("positive", 1), ("negative", 0)]:
        folder = root / subdir
        if not folder.is_dir():
            continue
        for wav_path in sorted(folder.glob("*.wav")):
            files.append(FileLabel(str(wav_path), label))

    return files


def _segment_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_s: float,
    hop_s: float,
) -> List[torch.Tensor]:
    """
    Slice a waveform into fixed-length overlapping segments.

    Args:
        waveform: Tensor of shape (1, total_samples)
        sample_rate: Sample rate in Hz
        segment_s: Segment duration in seconds (e.g. 4.0)
        hop_s: Hop between segment starts in seconds (e.g. 2.0)

    Returns:
        List of waveform tensors, each shape (1, segment_samples).
        Final segment is dropped if it doesn't fill segment_s completely.
    """
    segment_samples = int(segment_s * sample_rate)
    hop_samples = int(hop_s * sample_rate)
    total_samples = waveform.shape[-1]

    segments = []
    start = 0
    while start + segment_samples <= total_samples:
        segments.append(waveform[:, start : start + segment_samples])
        start += hop_samples

    return segments


class SRKWFinetuneDataset(Dataset):
    """
    PyTorch Dataset that converts file-level labeled WAVs into segment-level
    (spectrogram, label) pairs ready for training.

    Segments are pre-computed on __init__ so DataLoader workers don't do
    redundant I/O.  For typical finetuning data sizes (tens to low hundreds
    of 1-min files) this fits comfortably in RAM.

    Args:
        data_dir: Path with positive/ and negative/ subdirectories of WAV files.
        config: The standard config dict (audio, spectrogram, model sections).
        segment_hop_s: Hop between segments in seconds. Default 2.0 matches
                       inference windowing.
        augment: If True, apply simple spectrogram augmentations (freq masking).
        seed: Random seed for reproducibility of augmentations.
    """

    def __init__(
        self,
        data_dir: str,
        config: Dict,
        segment_hop_s: float = 2.0,
        augment: bool = False,
        seed: int = 42,
    ):
        self.config = config
        self.augment = augment
        self.rng = random.Random(seed)

        audio_cfg = config["audio"]
        input_pad_s = config["model"]["input_pad_s"]  # 4.0

        files = _discover_files(data_dir)
        if len(files) == 0:
            raise FileNotFoundError(
                f"No WAV files found in {data_dir}/positive/ or {data_dir}/negative/"
            )

        # Pre-segment all files and store (spectrogram, label) pairs
        self.spectrograms: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.source_files: List[str] = []  # for debugging / per-file eval

        for finfo in files:
            waveform, sr = load_processed_waveform(finfo.path, audio_cfg)
            segments = _segment_waveform(waveform, sr, input_pad_s, segment_hop_s)

            for seg in segments:
                mel = prepare_waveform(seg, sr, config)
                self.spectrograms.append(mel)
                self.labels.append(finfo.label)
                self.source_files.append(finfo.path)

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spec = self.spectrograms[idx]

        if self.augment:
            spec = self._apply_augmentation(spec)

        return spec, self.labels[idx]

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _apply_augmentation(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Lightweight spectrogram augmentation matching original training:
        frequency masking (SpecAugment-style).

        Args:
            spec: Tensor (1, n_mels, n_frames)

        Returns:
            Augmented tensor (same shape, original not mutated).
        """
        spec = spec.clone()
        _, n_mels, _ = spec.shape

        # Frequency masking: zero out a random band of up to 20 mel bins
        max_mask = min(20, n_mels // 8)
        mask_width = self.rng.randint(1, max_mask)
        mask_start = self.rng.randint(0, n_mels - mask_width)
        spec[:, mask_start : mask_start + mask_width, :] = 0.0

        return spec

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def class_counts(self) -> Dict[int, int]:
        """Return {label: count} for dataset balancing decisions."""
        counts: Dict[int, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def train_val_split(
        self, val_fraction: float = 0.15, seed: int = 42
    ) -> Tuple["SRKWFinetuneDataset", "SRKWFinetuneDataset"]:
        """
        Split into train/val subsets *by source file* so segments from the
        same file don't leak across the split.

        Returns:
            (train_dataset, val_dataset) — lightweight views sharing the
            underlying spectrogram tensors.
        """
        # Group indices by source file
        file_to_indices: Dict[str, List[int]] = {}
        for i, f in enumerate(self.source_files):
            file_to_indices.setdefault(f, []).append(i)

        files = sorted(file_to_indices.keys())
        rng = random.Random(seed)
        rng.shuffle(files)

        n_val = max(1, int(len(files) * val_fraction))
        val_files = set(files[:n_val])

        train_ds = _SubsetView(self, [i for f in files[n_val:] for i in file_to_indices[f]])
        val_ds = _SubsetView(self, [i for f in files[:n_val] for i in file_to_indices[f]])
        return train_ds, val_ds


class _SubsetView(Dataset):
    """Lightweight view into a parent SRKWFinetuneDataset."""

    def __init__(self, parent: SRKWFinetuneDataset, indices: List[int]):
        self.parent = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.parent[self.indices[idx]]

    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for i in self.indices:
            label = self.parent.labels[i]
            counts[label] = counts.get(label, 0) + 1
        return counts
