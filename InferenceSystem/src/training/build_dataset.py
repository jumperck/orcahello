"""
Build HF datasets from CSV annotations + audio folder.

Two dataset levels:
1. Recording-level: full 1-min audio + metadata + segment annotations
2. Segment-level: individual annotated segments, with large ones broken up

Usage:
    python -m src.training.build_dataset \
        --csv annotations.csv \
        --audio-dir /path/to/audio \
        --output-dir ./datasets \
        --max-segment-s 10.0

    # With one-time audio preprocessing (resample, downmix, normalize):
    python -m src.training.build_dataset \
        --csv annotations.csv \
        --audio-dir /path/to/audio \
        --output-dir ./datasets \
        --model-config model/config.yaml
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, Features, Sequence, Value

from .schemas import (
    LABEL_TO_TAG,
    AnnotationCSVRow,
    RecordingRow,
    SegmentAnnotation,
    SegmentRow,
    Tag,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _read_csv(csv_path: str) -> list[AnnotationCSVRow]:
    """Read and validate the annotation CSV."""
    rows = []
    with open(csv_path) as f:
        for i, raw in enumerate(csv.DictReader(f), start=2):  # line 2 = first data row
            try:
                rows.append(AnnotationCSVRow(**raw))
            except Exception as e:
                raise ValueError(f"CSV row {i}: {e}") from e
    return rows


def _break_segment(start_s: float, end_s: float, max_segment_s: float):
    """Break a segment into non-overlapping chunks of at most max_segment_s.

    Chunks are equal-sized to avoid tiny remainders. For example, a 25 s segment
    with max_segment_s=10 yields 3 chunks of ~8.33 s each rather than 10+10+5.
    """
    duration = end_s - start_s
    if duration <= max_segment_s:
        yield start_s, end_s
        return

    n_chunks = int(np.ceil(duration / max_segment_s))
    chunk_s = duration / n_chunks
    for i in range(n_chunks):
        yield start_s + i * chunk_s, start_s + (i + 1) * chunk_s


# =============================================================================
# HF Feature definitions
# =============================================================================


RECORDING_FEATURES = Features(
    {
        "audio": Audio(),
        "id": Value("string"),
        "tags": Sequence({"tag": Value("string"), "score": Value("float32")}),
        "comment": Value("string"),
        "segment_annotations": Sequence(
            {
                "start": Value("float32"),
                "end": Value("float32"),
                "tag": Value("string"),
            }
        ),
    }
)

SEGMENT_FEATURES = Features(
    {
        "audio": Audio(),
        "label": Value("int64"),
        "tag": Value("string"),
        "source_id": Value("string"),
        "start_s": Value("float32"),
        "end_s": Value("float32"),
    }
)


# =============================================================================
# Dataset builders
# =============================================================================


def build_recording_dataset(csv_path: str, audio_dir: str) -> Dataset:
    """Build recording-level HF dataset from CSV annotations + audio folder.

    Groups annotations by filename. Each row stores the full audio file,
    file-level tags (derived from annotations), and the list of segment
    annotations.

    Args:
        csv_path: Path to annotation CSV (columns: filename, start_s, end_s, label).
        audio_dir: Directory containing the audio files referenced in the CSV.

    Returns:
        HF Dataset with columns defined by RECORDING_FEATURES.
    """
    rows = _read_csv(csv_path)
    audio_dir = Path(audio_dir)

    by_file: dict[str, list[AnnotationCSVRow]] = defaultdict(list)
    for row in rows:
        by_file[row.filename].append(row)

    records = []
    for filename, file_rows in by_file.items():
        audio_path = audio_dir / filename
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        annotations = []
        tags: set[str] = set()
        for row in file_rows:
            tag = LABEL_TO_TAG[row.label]
            annotations.append(
                SegmentAnnotation(start=row.start_s, end=row.end_s, tag=tag)
            )
            tags.add(tag)

        rec = RecordingRow(
            id=Path(filename).stem,
            tags=[Tag(tag=t, score=1.0) for t in sorted(tags)],
            comment="",
            segment_annotations=annotations,
        )

        records.append(
            {
                "audio": str(audio_path),
                "id": rec.id,
                "tags": [t.model_dump() for t in rec.tags],
                "comment": rec.comment,
                "segment_annotations": [
                    a.model_dump(exclude={"duration_s"})
                    for a in rec.segment_annotations
                ],
            }
        )

    logger.info("Built recording dataset: %d recordings", len(records))
    return Dataset.from_list(records, features=RECORDING_FEATURES)


def build_segment_dataset(
    csv_path: str,
    audio_dir: str,
    max_segment_s: float = 10.0,
) -> Dataset:
    """Build segment-level HF dataset with large segments broken up.

    Each annotated segment in the CSV becomes one or more rows. Segments
    longer than ``max_segment_s`` are split into equal-sized chunks so that
    sample distribution is more representative (e.g. a 60 s negative annotation
    becomes 6 x 10 s rows).

    Audio slices are read directly from disk with soundfile seek — only the
    needed frames are loaded per segment.

    Args:
        csv_path: Path to annotation CSV (columns: filename, start_s, end_s, label).
        audio_dir: Directory containing the audio files referenced in the CSV.
        max_segment_s: Maximum segment duration; longer segments are split.

    Returns:
        HF Dataset with columns defined by SEGMENT_FEATURES.
    """
    rows = _read_csv(csv_path)
    audio_dir = Path(audio_dir)

    # Group by file so we read sf.info once per file
    by_file: dict[str, list[AnnotationCSVRow]] = defaultdict(list)
    for row in rows:
        by_file[row.filename].append(row)

    records = []
    for filename, file_rows in by_file.items():
        audio_path = audio_dir / filename
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        info = sf.info(str(audio_path))
        sr = info.samplerate

        for row in file_rows:
            label = row.label
            tag = LABEL_TO_TAG[label]
            source_id = Path(filename).stem

            for chunk_start, chunk_end in _break_segment(
                row.start_s, row.end_s, max_segment_s
            ):
                start_frame = int(chunk_start * sr)
                stop_frame = int(chunk_end * sr)
                data, _ = sf.read(
                    str(audio_path),
                    start=start_frame,
                    stop=stop_frame,
                    dtype="float32",
                )

                # Validate via Pydantic schema
                seg = SegmentRow(
                    label=label,
                    tag=tag,
                    source_id=source_id,
                    start_s=chunk_start,
                    end_s=chunk_end,
                )

                records.append(
                    {
                        "audio": {"array": data, "sampling_rate": sr},
                        **seg.model_dump(exclude={"duration_s"}),
                    }
                )

    logger.info(
        "Built segment dataset: %d segments from %d recordings "
        "(max_segment_s=%.1f)",
        len(records),
        len(by_file),
        max_segment_s,
    )
    return Dataset.from_list(records, features=SEGMENT_FEATURES)


# =============================================================================
# One-time audio preprocessing
# =============================================================================


def preprocess_audio_column(
    dataset: Dataset,
    audio_config: dict,
) -> Dataset:
    """Apply one-time, materialized audio transforms to the audio column.

    Applies the transforms defined in model_config.audio (resample, downmix,
    normalize) and materializes the result to a new Arrow cache. This avoids
    repeating expensive resampling every epoch.

    Uses ``cast_column`` for resampling (handled natively by HF Audio) and
    ``.map()`` for downmix + normalize.

    Args:
        dataset: HF Dataset with an ``audio`` column.
        audio_config: Dict matching AudioConfig fields:
            - resample_rate (int): target sample rate
            - downmix_mono (bool): average channels to mono
            - normalize (bool): peak-normalize so max(|x|) = 1.0

    Returns:
        New Dataset with preprocessed audio column.
    """
    target_sr = audio_config.get("resample_rate")
    downmix = audio_config.get("downmix_mono", True)
    normalize = audio_config.get("normalize", False)

    # Resampling: HF Audio handles this efficiently at decode time.
    # cast_column sets the target SR; .map() below triggers the actual resample.
    if target_sr:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    if not (downmix or normalize):
        # Force materialization of resampled audio so it's cached in Arrow
        if target_sr:
            dataset = dataset.map(lambda x: x)
        return dataset

    def _process(example):
        audio = example["audio"]
        array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        if downmix and array.ndim > 1:
            array = array.mean(axis=-1)

        if normalize:
            peak = np.abs(array).max()
            if peak > 0:
                array = array / peak

        return {"audio": {"array": array, "sampling_rate": sr}}

    return dataset.map(_process)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Build HF datasets from CSV annotations + audio folder."
    )
    parser.add_argument("--csv", required=True, help="Path to annotation CSV")
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for saved datasets",
    )
    parser.add_argument(
        "--max-segment-s",
        type=float,
        default=10.0,
        help="Max segment duration for segment dataset (default: 10.0)",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to model config YAML; if provided, applies audio preprocessing",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="HF Hub repo prefix, e.g. 'orcasound/srkw'. "
        "Creates {prefix}-recordings and {prefix}-segments.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build datasets ---
    logger.info("Building recording dataset...")
    recording_ds = build_recording_dataset(args.csv, args.audio_dir)

    logger.info("Building segment dataset (max_segment_s=%.1f)...", args.max_segment_s)
    segment_ds = build_segment_dataset(args.csv, args.audio_dir, args.max_segment_s)

    # --- Optional audio preprocessing ---
    if args.model_config:
        from src.model.types import DetectorInferenceConfig

        config = DetectorInferenceConfig.from_yaml(args.model_config)
        audio_cfg = config.as_dict()["audio"]
        logger.info("Applying audio preprocessing: %s", audio_cfg)
        recording_ds = preprocess_audio_column(recording_ds, audio_cfg)
        segment_ds = preprocess_audio_column(segment_ds, audio_cfg)

    # --- Save ---
    rec_path = output_dir / "recording_dataset"
    seg_path = output_dir / "segment_dataset"
    recording_ds.save_to_disk(str(rec_path))
    segment_ds.save_to_disk(str(seg_path))
    logger.info("Saved recording dataset to %s", rec_path)
    logger.info("Saved segment dataset to %s", seg_path)

    # --- Optional push to Hub ---
    if args.push_to_hub:
        prefix = args.push_to_hub
        recording_ds.push_to_hub(f"{prefix}-recordings")
        segment_ds.push_to_hub(f"{prefix}-segments")
        logger.info("Pushed to Hub: %s-recordings, %s-segments", prefix, prefix)


if __name__ == "__main__":
    main()
