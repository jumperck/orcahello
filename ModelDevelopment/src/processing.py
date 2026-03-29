"""Post-inference processing: merge global_confidence and produce segment-level CSVs."""

import json
import logging
from pathlib import Path

import pandas as pd

from .segmentation import auto_segment_from_json

logger = logging.getLogger(__name__)


def merge_confidences(complete_csv: Path, inference_dir: Path) -> pd.DataFrame:
    """Merge global_confidence from summary.csv into the complete CSV (in-place)."""
    summary_csv = inference_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"No summary.csv found in {inference_dir}")

    df = pd.read_csv(complete_csv, dtype=str)
    summary = pd.read_csv(summary_csv)
    summary["detection_id"] = summary["file_path"].apply(lambda p: Path(p).stem)
    conf_map = dict(zip(summary["detection_id"], summary["global_confidence"].astype(str)))

    before_missing = df["global_confidence"].isna().sum() if "global_confidence" in df.columns else len(df)
    df["global_confidence"] = df["detection_id"].map(conf_map).fillna(
        df.get("global_confidence", pd.NA)
    )
    after_missing = df["global_confidence"].isna().sum()

    df.to_csv(complete_csv, index=False)
    matched = len(df) - after_missing
    logger.info(
        "Updated %s: %d/%d rows have global_confidence (was %d missing, now %d missing)",
        complete_csv, matched, len(df), before_missing, after_missing,
    )
    return df


def build_segmented_csv(
    df: pd.DataFrame,
    inference_dir: Path,
    output_path: Path,
    min_threshold: float = 0.1,
    gap_tolerance_s: float = 1.0,
) -> None:
    """Build segment-level CSV by running auto_segment on each detection's JSON."""
    metadata_cols = [
        "location_slug", "year_month_pacific", "date_hour_pacific",
        "timestamp_pacific", "detection_id", "detection_link",
    ]
    rows = []
    missing = 0

    for _, row in df.iterrows():
        detection_id = row["detection_id"]
        year_month = row["year_month_pacific"]
        json_path = inference_dir / year_month / "audio" / f"{detection_id}.json"

        if not json_path.exists():
            missing += 1
            logger.warning("JSON not found, skipping: %s", json_path)
            continue

        binary_label = int(row["binary_label"])

        if binary_label == 0:
            # Negative: single span covering the whole file, no segmentation needed
            with open(json_path) as f:
                result = json.load(f)
            file_duration_s = result["metadata"]["file_duration_s"]
            global_conf = result["global_confidence"]
            meta = {col: row.get(col, "") for col in metadata_cols}
            rows.append({
                **meta,
                "segment_start_s": 0.0,
                "segment_end_s": round(file_duration_s, 3),
                "segment_confidence": round(1.0 - global_conf, 4),
                "segment_binary_label": 0,
            })
            continue

        spans = auto_segment_from_json(
            str(json_path),
            min_threshold=min_threshold,
            gap_tolerance_s=gap_tolerance_s,
        )

        if not spans:
            continue

        meta = {col: row.get(col, "") for col in metadata_cols}
        for span in spans:
            rows.append({
                **meta,
                "segment_start_s": span["start_s"],
                "segment_end_s": span["end_s"],
                "segment_confidence": span["confidence"],
                "segment_binary_label": 1,
            })

    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(output_path, index=False)
    logger.info(
        "Wrote %d segment rows to %s (%d detections had no JSON)",
        len(rows), output_path, missing,
    )
