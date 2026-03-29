"""Data loading, post-inference processing, and shared utilities."""

import json
import logging
from pathlib import Path

import pandas as pd

from .segmentation import auto_segment_from_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Month expansion
# ---------------------------------------------------------------------------

def expand_months(raw: list[str]) -> list[str]:
    """Expand month args, supporting colon-separated ranges like '2025-11:2026-02'."""
    months = []
    for token in raw:
        if ":" in token:
            start, end = token.split(":", 1)
            cur = pd.Timestamp(start + "-01")
            stop = pd.Timestamp(end + "-01")
            while cur <= stop:
                months.append(cur.strftime("%Y-%m"))
                cur += pd.offsets.MonthBegin(1)
        else:
            months.append(token)
    return months


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(source: str, months: list[str], logbook_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load detections and hourly events, filtered to source+months (all locations)."""
    detections_csv = logbook_dir / "detections" / "all_detections.csv"
    hourly_csv = logbook_dir / "hourly_events" / "all_hourly_events.csv"
    det_df = pd.read_csv(detections_csv, dtype=str)
    det_df = det_df[
        (det_df["source"] == source)
        & (det_df["year_month_pacific"].isin(months))
    ].copy()
    det_df["binary_label"] = (det_df["srkw_positive"] == "true").astype(int)
    det_df["meta_orcahello_confidence"] = pd.to_numeric(det_df["meta_orcahello_confidence"], errors="coerce") / 100.0

    hourly = pd.read_csv(hourly_csv, dtype=str)
    hourly = hourly[
        (hourly["source"] == source)
        & (hourly["year_month_pacific"].isin(months))
    ].copy()

    return det_df, hourly


def print_location_summary(det_df: pd.DataFrame, hourly: pd.DataFrame):
    """Print detection and hourly event counts by location."""
    det_by_loc = det_df.groupby("location_slug").agg(
        detections=("detection_id", "size"),
        srkw_positive=("binary_label", "sum"),
    ).reset_index()
    det_by_loc["srkw_negative"] = det_by_loc["detections"] - det_by_loc["srkw_positive"]

    hourly_by_loc = hourly.groupby("location_slug").size().reset_index(name="hourly_events")

    summary = det_by_loc.merge(hourly_by_loc, on="location_slug", how="outer").fillna(0)
    summary = summary.sort_values("detections", ascending=False)

    print("\n  By location:")
    print(f"  {'location':<25} {'detections':>10} {'positive':>10} {'negative':>10} {'hourly_events':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")
    for _, r in summary.iterrows():
        print(f"  {r['location_slug']:<25} {int(r['detections']):>10} {int(r['srkw_positive']):>10} {int(r['srkw_negative']):>10} {int(r['hourly_events']):>14}")
    print(f"  {'TOTAL':<25} {int(summary['detections'].sum()):>10} {int(summary['srkw_positive'].sum()):>10} {int(summary['srkw_negative'].sum()):>10} {int(summary['hourly_events'].sum()):>14}")
    print()


def join_hourly_info(det_df: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Join date_hour_pacific from hourly events via exploded detection_ids."""
    hourly = hourly.copy()
    hourly["detection_id"] = hourly["detection_ids"].str.split(";")
    hourly = hourly.explode("detection_id")
    hourly["date_hour_pacific"] = hourly["date_pacific"] + " " + hourly["hour_pacific"].str.zfill(2) + ":00"

    det_df = det_df.merge(
        hourly[["detection_id", "date_hour_pacific"]],
        on="detection_id",
        how="left",
    )
    return det_df


def load_uris_from_cache(months: list[str], cache_dir: Path) -> pd.DataFrame:
    """Load audioUri and spectrogramUri from raw_detections.json for each month.

    # TODO: eliminate this cache dependency by deriving URIs from blob path schema
    #   once the naming convention is documented / stabilised.
    #   Audio URIs follow the pattern:
    #     https://livemlaudiospecstorage.blob.core.windows.net/audiowavs/<stem>.wav
    #   where <stem> encodes location + recording timestamp (e.g. rpi_north_sjc_2025_07_01_13_53_18_PDT)
    #   which cannot be reliably reconstructed from timestamp_pacific alone.
    """
    rows = []
    for month in months:
        cache_file = cache_dir / month / "raw_detections.json"
        if not cache_file.exists():
            print(f"  Warning: no cache file for {month}: {cache_file}")
            continue
        with open(cache_file) as f:
            detections = json.load(f)
        for d in detections:
            rows.append({
                "detection_id": d.get("id"),
                "audio_uri": d.get("audioUri", ""),
                "spectrogram_uri": d.get("spectrogramUri", ""),
            })
    if not rows:
        return pd.DataFrame(columns=["detection_id", "audio_uri", "spectrogram_uri"])
    return pd.DataFrame(rows)


def build_complete_df(
    *,
    source: str,
    months: list[str],
    location: str,
    logbook_dir: Path,
    cache_dir: Path,
) -> pd.DataFrame:
    """Load data and build the complete (pre-sampling) dataframe.

    Args:
        source: Detection source to filter (e.g. 'orcahello_moderated').
        months: List of YYYY-MM month strings.
        location: Location slug or 'all'.
        logbook_dir: Path to combined_logbook directory.
        cache_dir: Path to OrcaHello raw detection cache.
    """
    print(f"Loading detections for source={source}, months={months}")
    det_df, hourly = load_data(source, months, logbook_dir)
    print(f"  {len(det_df)} total detections, {len(hourly)} hourly events")
    print_location_summary(det_df, hourly)

    if location != "all":
        print(f"Filtering to location={location}")
        det_df = det_df[det_df["location_slug"] == location].copy()
        hourly = hourly[hourly["location_slug"] == location].copy()
    else:
        print("Using all locations")
    print(f"  {len(det_df)} detections, {len(hourly)} hourly events")
    if det_df.empty:
        return det_df

    det_df = join_hourly_info(det_df, hourly)

    # Use v0 detector confidence as default global_confidence
    print("No inference version specified, using confidence_detector_v0 for sampling")
    det_df["global_confidence"] = det_df["meta_orcahello_confidence"]

    # Join audio/spectrogram URIs from raw cache
    print(f"Loading URIs from cache ({cache_dir})...")
    uri_df = load_uris_from_cache(months, cache_dir)
    print(f"  {len(uri_df)} URIs loaded")
    det_df = det_df.merge(uri_df, on="detection_id", how="left")

    return det_df


# ---------------------------------------------------------------------------
# Post-inference processing
# ---------------------------------------------------------------------------

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
