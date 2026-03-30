"""Data loading utilities: logbook/cache loading, month expansion, build_complete_df."""

import json
from pathlib import Path

import pandas as pd


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
