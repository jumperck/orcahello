"""Generate dataset summary stats and push an HF Dataset to the Hugging Face Hub."""

import argparse
import io
import logging
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_from_disk

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Dataset type detection
# ---------------------------------------------------------------------------

RECORDING_COLUMNS = {"recording_id", "tags", "metadata", "segment_annotations"}
SEGMENT_COLUMNS = {"label", "tag", "source_id", "start_s", "end_s"}


def detect_dataset_type(ds) -> str:
    """Detect whether an HF Dataset is recording-level or segment-level."""
    cols = set(ds.column_names)
    if SEGMENT_COLUMNS.issubset(cols):
        return "segment"
    if RECORDING_COLUMNS.issubset(cols):
        return "recording"
    raise ValueError(
        f"Cannot detect dataset type from columns: {sorted(cols)}. "
        f"Expected recording columns {RECORDING_COLUMNS} or segment columns {SEGMENT_COLUMNS}."
    )


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def _compute_recording_stats(ds) -> dict:
    """Compute summary statistics for a recording-level dataset."""
    n = len(ds)
    tag_counts: Counter = Counter()
    conf_by_tag: dict[str, list[float]] = {}
    location_counts: Counter = Counter()
    month_counts: Counter = Counter()
    has_audio = 0
    has_segments = 0
    total_duration_s = 0.0
    duration_by_tag: dict[str, float] = {}

    ds_raw = ds.cast_column("audio", Audio(decode=False))

    for i in range(n):
        ex = ds_raw[i]
        tags = ex["tags"]
        # HF returns list-of-dicts when iterating row-by-row
        if isinstance(tags, list):
            has_tags = len(tags) > 0
            tag = tags[0]["tag"] if has_tags else "unknown"
            score = tags[0]["score"] if has_tags else 0.0
        else:
            # dict-of-lists (batch mode)
            has_tags = tags and tags.get("tag") and len(tags["tag"]) > 0
            tag = tags["tag"][0] if has_tags else "unknown"
            score = tags["score"][0] if has_tags else 0.0

        tag_counts[tag] += 1
        conf_by_tag.setdefault(tag, []).append(score)

        meta = ex["metadata"]
        if meta:
            location_counts[meta["location_slug"]] += 1
            month_counts[meta["year_month_pacific"]] += 1

        audio = ex.get("audio")
        if audio:
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            audio_path = audio.get("path") if isinstance(audio, dict) else None
            if audio_bytes:
                has_audio += 1
                try:
                    dur = sf.info(io.BytesIO(audio_bytes)).duration
                    total_duration_s += dur
                    duration_by_tag[tag] = duration_by_tag.get(tag, 0.0) + dur
                except Exception:
                    pass
            elif audio_path and Path(audio_path).exists():
                has_audio += 1
                try:
                    dur = sf.info(audio_path).duration
                    total_duration_s += dur
                    duration_by_tag[tag] = duration_by_tag.get(tag, 0.0) + dur
                except Exception:
                    pass

        anns = ex.get("segment_annotations")
        if anns:
            if isinstance(anns, list) and len(anns) > 0:
                has_segments += 1
            elif isinstance(anns, dict) and anns.get("start") and len(anns["start"]) > 0:
                has_segments += 1

    conf_stats = {}
    for tag, scores in conf_by_tag.items():
        scores_f = [s for s in scores if s is not None]
        if scores_f:
            conf_stats[tag] = {
                "mean": sum(scores_f) / len(scores_f),
                "min": min(scores_f),
                "max": max(scores_f),
            }

    return {
        "total_recordings": n,
        "tag_counts": dict(tag_counts.most_common()),
        "location_counts": dict(location_counts.most_common()),
        "month_counts": dict(sorted(month_counts.items())),
        "has_audio": has_audio,
        "has_segment_annotations": has_segments,
        "total_duration_s": total_duration_s,
        "duration_by_tag": duration_by_tag,
        "confidence_stats": conf_stats,
    }


def _compute_segment_stats(ds) -> dict:
    """Compute summary statistics for a segment-level dataset."""
    n = len(ds)
    label_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    source_ids: set = set()
    total_duration_s = 0.0
    duration_by_label: dict[int, float] = {}
    has_audio = 0

    ds_raw = ds.cast_column("audio", Audio(decode=False))

    for i in range(n):
        ex = ds_raw[i]
        label = ex["label"]
        tag = ex["tag"]
        label_counts[label] += 1
        tag_counts[tag] += 1
        source_ids.add(ex["source_id"])

        seg_dur = ex["end_s"] - ex["start_s"]
        total_duration_s += seg_dur
        duration_by_label[label] = duration_by_label.get(label, 0.0) + seg_dur

        audio = ex.get("audio")
        if audio:
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            if audio_bytes:
                has_audio += 1

    return {
        "total_segments": n,
        "label_counts": dict(sorted(label_counts.items())),
        "tag_counts": dict(tag_counts.most_common()),
        "unique_source_recordings": len(source_ids),
        "has_audio": has_audio,
        "total_duration_s": total_duration_s,
        "duration_by_label": duration_by_label,
    }


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.0f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.0f}s"
    return f"{secs:.1f}s"


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_recording_summary_md(ds, name: str) -> str:
    """Generate Markdown summary for a recording-level dataset."""
    stats = _compute_recording_stats(ds)
    lines = []
    lines.append(f"# Dataset Summary: `{name}`\n")
    lines.append(f"- **Total recordings**: {stats['total_recordings']}")
    lines.append(f"- **With audio**: {stats['has_audio']}")
    lines.append(f"- **With segment annotations**: {stats['has_segment_annotations']}")
    if stats["total_duration_s"] > 0:
        lines.append(f"- **Total audio duration**: {_fmt_duration(stats['total_duration_s'])}")
    lines.append("")

    lines.append("## Label Distribution\n")
    lines.append("| Tag | Count | % |")
    lines.append("|-----|------:|--:|")
    for tag, count in stats["tag_counts"].items():
        pct = 100 * count / stats["total_recordings"]
        lines.append(f"| `{tag}` | {count} | {pct:.1f}% |")
    lines.append("")

    if stats["duration_by_tag"]:
        lines.append("## Audio Duration by Label\n")
        lines.append("| Tag | Duration | % |")
        lines.append("|-----|----------|--:|")
        for tag, dur in stats["duration_by_tag"].items():
            pct = 100 * dur / stats["total_duration_s"] if stats["total_duration_s"] > 0 else 0
            lines.append(f"| `{tag}` | {_fmt_duration(dur)} | {pct:.1f}% |")
        lines.append("")

    if stats["confidence_stats"]:
        lines.append("## Confidence Scores\n")
        lines.append("| Tag | Mean | Min | Max |")
        lines.append("|-----|-----:|----:|----:|")
        for tag, cs in stats["confidence_stats"].items():
            lines.append(f"| `{tag}` | {cs['mean']:.3f} | {cs['min']:.3f} | {cs['max']:.3f} |")
        lines.append("")

    if stats["location_counts"]:
        lines.append("## Location Distribution\n")
        lines.append("| Location | Count |")
        lines.append("|----------|------:|")
        for loc, count in stats["location_counts"].items():
            lines.append(f"| {loc} | {count} |")
        lines.append("")

    if stats["month_counts"]:
        lines.append("## Month Distribution\n")
        lines.append("| Month | Count |")
        lines.append("|-------|------:|")
        for month, count in stats["month_counts"].items():
            lines.append(f"| {month} | {count} |")
        lines.append("")

    return "\n".join(lines)


def generate_segment_summary_md(ds, name: str) -> str:
    """Generate Markdown summary for a segment-level dataset."""
    from dataset_toolkit.models import LABEL_TO_TAG

    stats = _compute_segment_stats(ds)
    lines = []
    lines.append(f"# Dataset Summary: `{name}`\n")
    lines.append(f"- **Total segments**: {stats['total_segments']}")
    lines.append(f"- **With audio**: {stats['has_audio']}")
    lines.append(f"- **Unique source recordings**: {stats['unique_source_recordings']}")
    if stats["total_duration_s"] > 0:
        lines.append(f"- **Total segment duration**: {_fmt_duration(stats['total_duration_s'])}")
    lines.append("")

    lines.append("## Label Distribution\n")
    lines.append("| Label | Tag | Count (%) | Duration |")
    lines.append("|------:|-----|----------:|---------:|")
    for label, count in stats["label_counts"].items():
        tag = LABEL_TO_TAG.get(label, "unknown")
        pct = 100 * count / stats["total_segments"]
        dur = stats["duration_by_label"].get(label, 0)
        lines.append(f"| {label} | `{tag}` | {count} ({pct:.1f}%) | {_fmt_duration(dur)} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate summary statistics for an HF Dataset.",
    )
    parser.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="Path to an HF Dataset on disk (e.g. datasets/2020-07_2021-06--all/recording_dataset)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        logger.error("Dataset directory does not exist: %s", dataset_dir)
        sys.exit(1)

    ds = load_from_disk(str(dataset_dir))
    ds_type = detect_dataset_type(ds)
    logger.info("Detected %s dataset with %d rows.", ds_type, len(ds))

    name = dataset_dir.name
    if ds_type == "recording":
        summary_md = generate_recording_summary_md(ds, name)
    else:
        summary_md = generate_segment_summary_md(ds, name)

    summary_path = dataset_dir.parent / f"{name}--summary.md"
    summary_path.write_text(summary_md)
    logger.info("Wrote summary to %s", summary_path)
    print(summary_md)


if __name__ == "__main__":
    main()
