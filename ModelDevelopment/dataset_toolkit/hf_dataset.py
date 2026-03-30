"""HuggingFace Dataset builders for OrcaHello recording and segment datasets."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, Dataset, Features, Sequence, Value

from .models import LABEL_TO_TAG, OrcaHelloRecordingMetadata, Tag
from .segmentation import auto_segment_from_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HF Feature definitions
# ---------------------------------------------------------------------------

RECORDING_FEATURES = Features(
    {
        "audio": Audio(),
        "recording_id": Value("string"),
        "tags": Sequence({"tag": Value("string"), "score": Value("float32")}),
        "metadata": {
            "location_slug": Value("string"),
            "year_month_pacific": Value("string"),
            "date_hour_pacific": Value("string"),
            "timestamp_pacific": Value("string"),
            "audio_uri": Value("string"),
            "spectrogram_uri": Value("string"),
        },
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _resolve_audio_path(
    audio_dir: Path, year_month: str, detection_id: str
) -> str | None:
    """Return the FLAC path if it exists under audio_dir, else None."""
    path = audio_dir / year_month / "audio" / f"{detection_id}.flac"
    if path.exists():
        return str(path)
    return None


# ---------------------------------------------------------------------------
# Recording-level dataset builder
# ---------------------------------------------------------------------------


def build_recording_dataset(
    df: pd.DataFrame,
    audio_dir: Path | None = None,
) -> Dataset:
    """Build recording-level HF Dataset from a DataFrame (output of build_complete_df).

    Args:
        df: DataFrame with columns from build_complete_df (detection_id,
            binary_label, global_confidence, comments, location_slug,
            year_month_pacific, etc.)
        audio_dir: If provided, populate audio column with local FLAC paths.
            Expected layout: {audio_dir}/{year_month}/audio/{detection_id}.flac

    Returns:
        HF Dataset with RECORDING_FEATURES schema.
    """
    records = []
    for _, row in df.iterrows():
        detection_id = row["detection_id"]
        binary_label = int(row.get("binary_label", 0))
        tag = LABEL_TO_TAG[binary_label]

        confidence = row.get("global_confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        tags = [{"tag": tag, "score": confidence}]

        metadata = OrcaHelloRecordingMetadata(
            location_slug=str(row.get("location_slug", "")),
            year_month_pacific=str(row.get("year_month_pacific", "")),
            date_hour_pacific=str(row.get("date_hour_pacific", "")),
            timestamp_pacific=str(row.get("timestamp_pacific", "")),
            audio_uri=str(row.get("audio_uri", "")),
            spectrogram_uri=str(row.get("spectrogram_uri", "")),
        )

        audio_path = None
        if audio_dir:
            audio_path = _resolve_audio_path(
                audio_dir, metadata.year_month_pacific, detection_id
            )

        records.append(
            {
                "audio": audio_path,
                "recording_id": detection_id,
                "tags": tags,
                "metadata": metadata.model_dump(),
                "comment": str(row.get("comments", "") or ""),
                "segment_annotations": [],
            }
        )

    logger.info("Built recording dataset: %d recordings", len(records))
    return Dataset.from_list(records, features=RECORDING_FEATURES)


# ---------------------------------------------------------------------------
# Segment annotation enrichment
# ---------------------------------------------------------------------------


def add_segment_annotations(
    dataset: Dataset,
    inference_dir: Path,
    min_threshold: float = 0.1,
    gap_tolerance_s: float = 1.0,
) -> Dataset:
    """Add segment_annotations to a recording-level dataset using inference JSONs.

    Also updates tags[0].score with the global_confidence from inference.
    Modifies dataset in-place (returns a new Dataset object).

    Args:
        dataset: Recording-level HF Dataset.
        inference_dir: Directory containing inference results. Expected layout:
            {inference_dir}/summary.csv (for global_confidence)
            {inference_dir}/{year_month}/audio/{detection_id}.json
        min_threshold: Safety floor for dynamic per-file confidence threshold.
        gap_tolerance_s: Max gap in seconds between segments to merge.
    """
    # Load global confidences from summary.csv
    summary_csv = inference_dir / "summary.csv"
    conf_map: dict[str, float] = {}
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        summary["detection_id"] = summary["file_path"].apply(lambda p: Path(p).stem)
        conf_map = dict(zip(summary["detection_id"], summary["global_confidence"]))
        logger.info("Loaded %d confidence scores from %s", len(conf_map), summary_csv)

    def _process(example):
        recording_id = example["recording_id"]
        year_month = example["metadata"]["year_month_pacific"]

        # Update confidence from inference
        # tags is columnar: {'tag': [...], 'score': [...]}
        new_conf = conf_map.get(recording_id)
        tag_list = list(example["tags"]["tag"])
        score_list = list(example["tags"]["score"])
        if new_conf is not None and tag_list:
            score_list[0] = float(new_conf)
        tags = {"tag": tag_list, "score": score_list}

        # Find inference JSON
        json_path = inference_dir / year_month / "audio" / f"{recording_id}.json"
        if not json_path.exists():
            return {"tags": tags, "segment_annotations": example["segment_annotations"]}

        # Determine label from existing tag
        is_positive = tag_list[0] == "srkw_positive" if tag_list else False

        if not is_positive:
            # Negative: single span covering the whole file
            with open(json_path) as f:
                result = json.load(f)
            file_duration_s = result["metadata"]["file_duration_s"]
            annotations = [
                {"start": 0.0, "end": round(file_duration_s, 3), "tag": "srkw_negative"}
            ]
        else:
            spans = auto_segment_from_json(
                str(json_path),
                min_threshold=min_threshold,
                gap_tolerance_s=gap_tolerance_s,
            )
            annotations = [
                {"start": s["start_s"], "end": s["end_s"], "tag": "srkw_positive"}
                for s in spans
            ]

        return {"tags": tags, "segment_annotations": annotations}

    result = dataset.map(_process, writer_batch_size=100)
    logger.info("Added segment annotations to %d recordings", len(result))
    return result


# ---------------------------------------------------------------------------
# Segment-level dataset builder
# ---------------------------------------------------------------------------


def build_segment_dataset(
    dataset_path: str | Path,
    max_segment_s: float = 10.0,
    num_proc: int = 1,
) -> Dataset:
    """Build segment-level HF Dataset from a recording-level dataset with segment_annotations.

    Args:
        dataset_path: Path to a recording-level HF Dataset on disk.
        max_segment_s: Maximum segment duration; longer segments are split.
        num_proc: Number of parallel processes.

    Returns:
        HF Dataset with SEGMENT_FEATURES schema.
    """
    import io

    if num_proc < 1:
        raise ValueError(f"num_proc must be >= 1, got {num_proc}")

    from datasets import load_from_disk

    dataset_path = str(dataset_path)
    # Load once to get length; workers will re-load (memory-mapped) independently
    n = len(load_from_disk(dataset_path))

    def _generate_segments(indices, ds_path):
        from datasets import Audio as _Audio
        from datasets import load_from_disk as _load

        ds = _load(ds_path).cast_column("audio", _Audio(decode=False))
        for idx in indices:
            example = ds[idx]
            recording_id = example["recording_id"]
            annotations = example.get("segment_annotations", [])

            if not annotations:
                continue

            audio_col = example.get("audio")
            flac_bytes = audio_col.get("bytes") if audio_col else None
            if not flac_bytes:
                continue

            buf = io.BytesIO(flac_bytes)
            sr = sf.info(buf).samplerate

            for ann in annotations:
                tag = ann["tag"]
                label = 1 if tag == "srkw_positive" else 0

                for chunk_start, chunk_end in _break_segment(
                    ann["start"], ann["end"], max_segment_s
                ):
                    buf.seek(0)
                    start_frame = int(chunk_start * sr)
                    stop_frame = int(chunk_end * sr)
                    data, _ = sf.read(
                        buf, start=start_frame, stop=stop_frame, dtype="float32",
                    )
                    audio_value = {"array": data, "sampling_rate": sr}

                    yield {
                        "audio": audio_value,
                        "label": label,
                        "tag": tag,
                        "source_id": recording_id,
                        "start_s": chunk_start,
                        "end_s": chunk_end,
                    }

    result = Dataset.from_generator(
        _generate_segments,
        features=SEGMENT_FEATURES,
        gen_kwargs={
            "indices": list(range(n)),
            "ds_path": dataset_path,
        },
        num_proc=num_proc,
    )
    logger.info(
        "Built segment dataset: %d segments from %d recordings (max_segment_s=%.1f, num_proc=%d)",
        len(result),
        n,
        max_segment_s,
        num_proc,
    )
    return result


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------


def preprocess_audio_column(
    dataset: Dataset,
    audio_config: dict,
) -> Dataset:
    """Apply one-time, materialized audio transforms to the audio column.

    Uses the InferenceSystem audio frontend to apply the same preprocessing
    used during inference: resample, downmix to mono, and optional peak
    normalization.

    Args:
        dataset: HF Dataset with an ``audio`` column.
        audio_config: Dict with keys:
            - resample_rate (int): target sample rate
            - downmix_mono (bool): average channels to mono
            - normalize (bool): peak-normalize so max(|x|) = 1.0
    """
    import torch

    from inference.model.audio_frontend import _downmix_to_mono, _resample_audio

    target_sr = audio_config.get("resample_rate")
    downmix = audio_config.get("downmix_mono", True)
    normalize = audio_config.get("normalize", False)

    def _process(example):
        audio = example["audio"]
        if audio is None:
            return example
        array = np.array(audio["array"], dtype=np.float32)
        orig_sr = audio["sampling_rate"]

        if array.ndim == 1:
            waveform = torch.from_numpy(array.reshape(1, -1))
        else:
            waveform = torch.from_numpy(array.T)

        if downmix and waveform.shape[0] > 1:
            waveform = _downmix_to_mono(waveform)

        if target_sr and target_sr != orig_sr:
            waveform = _resample_audio(waveform, orig_sr, target_sr)
            orig_sr = target_sr

        if normalize:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        return {
            "audio": {
                "array": waveform.squeeze(0).numpy(),
                "sampling_rate": orig_sr,
            }
        }

    return dataset.map(_process)
