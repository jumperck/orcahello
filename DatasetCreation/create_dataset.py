"""Create a curated evaluation dataset by bias-sampling hard examples from the detection logbook."""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


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

BASE_DIR = Path(__file__).parent
DEFAULT_LOGBOOK_DIR = BASE_DIR / "combined_logbook"
DEFAULT_CACHE_DIR = BASE_DIR / "fetch_cache" / "orcahello"
INFERENCE_DIR = BASE_DIR / "inference_results"
OUTPUT_DIR = BASE_DIR / "datasets"

LINK_TEMPLATE = "https://aifororcas.azurewebsites.net/detections/detection/{detection_id}"

# Regex to extract UUID from inference summary file_path column
UUID_RE = re.compile(r"--([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.\w+$")


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


def load_inference_confidences(months: list[str], inference_version: str) -> pd.DataFrame:
    """Load global_confidence from inference summary.csv files, keyed by detection_id."""
    rows = []
    for month in months:
        summary_path = INFERENCE_DIR / f"{month}--{inference_version}" / "summary.csv"
        if not summary_path.exists():
            print(f"  Warning: no inference results at {summary_path}")
            continue

        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            m = UUID_RE.search(row["file_path"])
            if not m:
                continue
            detection_id = m.group(1)
            # Infer category from the subfolder in file_path
            category = row["file_path"].split("/")[0]
            rows.append({
                "detection_id": detection_id,
                "confidence_detector_v1": row["global_confidence"],
                "inference_category": category,
            })

    if not rows:
        return pd.DataFrame(columns=["detection_id", "confidence_detector_v1", "inference_category"])
    return pd.DataFrame(rows)


def _sample_label(pool: pd.DataFrame, hard_pool: pd.DataFrame, target: int,
                   hard_label: str, uniform_label: str, seed: int) -> pd.DataFrame:
    """Sample up to `target` examples, split 1:1 between hard and uniform with backfill."""
    target_hard = target // 2
    target_uniform = target - target_hard

    # Take hard examples (already sorted by priority)
    hard = hard_pool.head(target_hard).assign(example_type=hard_label)

    # Uniform from remainder
    remaining = pool[~pool.index.isin(hard.index)]
    n_uniform = min(target_uniform, len(remaining))
    uniform = remaining.sample(n=n_uniform, random_state=seed).assign(example_type=uniform_label)

    # Backfill: if hard pool was too small, fill from uniform pool
    total = len(hard) + len(uniform)
    if total < target:
        backfill_pool = pool[~pool.index.isin(hard.index) & ~pool.index.isin(uniform.index)]
        n_backfill = min(target - total, len(backfill_pool))
        if n_backfill > 0:
            backfill = backfill_pool.sample(n=n_backfill, random_state=seed).assign(example_type=uniform_label)
            uniform = pd.concat([uniform, backfill])

    return pd.concat([hard, uniform])


def bias_sample(
    df: pd.DataFrame,
    target_positives: int,
    negative_ratio: float,
    hard_pos_max_conf: float,
    hard_pos_min_conf: float,
    hard_neg_min_conf: float,
    seed: int,
    location: str = "",
) -> pd.DataFrame:
    """Bias-sample hard + uniform examples from detections with inference results."""
    prefix = f"  [{location}] " if location else "  "
    has_inf = df.dropna(subset=["confidence_detector_v1"]).copy()
    n_without = len(df) - len(has_inf)

    positives = has_inf[has_inf["binary_label"] == 1]
    negatives = has_inf[has_inf["binary_label"] == 0]

    print(f"{prefix}{len(df)} detections ({len(positives)} pos, {len(negatives)} neg"
          f"{f', {n_without} no inference' if n_without else ''})")

    if len(positives) < target_positives:
        print(f"{prefix}Warning: {len(positives)} positives available, below target of {target_positives}")

    # --- Positive sampling (cap at target_positives) ---
    hard_pos_pool = positives[
        (positives["confidence_detector_v1"] < hard_pos_max_conf)
        & (positives["confidence_detector_v1"] > hard_pos_min_conf)
    ].sort_values("confidence_detector_v1")

    target_pos = min(target_positives, len(positives))
    sampled_pos = _sample_label(positives, hard_pos_pool, target_pos,
                                "positive_hard", "positive_uniform", seed)

    # --- Negative sampling (cap at ratio * actual positives sampled) ---
    hard_neg_pool = negatives[
        negatives["confidence_detector_v1"] > hard_neg_min_conf
    ].sort_values("confidence_detector_v1", ascending=False)

    # If positives are scarce (< half target), use target-based negative count
    # so we still get good negative coverage for evaluation
    if len(sampled_pos) < target_positives // 2:
        target_neg = min(int(target_positives * negative_ratio), len(negatives))
    else:
        target_neg = min(int(len(sampled_pos) * negative_ratio), len(negatives))
    sampled_neg = _sample_label(negatives, hard_neg_pool, target_neg,
                                "negative_hard", "negative_uniform", seed)

    return pd.concat([sampled_pos, sampled_neg])


def format_complete(det_df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for complete (pre-sampling) output."""
    df = det_df.copy()
    df["confidence_detector_v0"] = df["meta_orcahello_confidence"]
    df["link"] = df["detection_id"].apply(lambda d: LINK_TEMPLATE.format(detection_id=d))

    output_cols = [
        "location_slug",
        "year_month_pacific",
        "date_hour_pacific",
        "timestamp_pacific",
        "detection_id",
        "binary_label",
        "comments",
        "confidence_detector_v0",
        "confidence_detector_v1",
        "audio_uri",
        "spectrogram_uri",
        "link",
    ]
    # audio_uri / spectrogram_uri may be absent if cache unavailable — fill with empty string
    for col in ("audio_uri", "spectrogram_uri"):
        if col not in df.columns:
            df[col] = ""
    return df[output_cols].sort_values(["year_month_pacific", "date_hour_pacific", "timestamp_pacific"])


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for final sampled output (same schema as complete, plus example_type)."""
    df = df.copy()
    df["confidence_detector_v0"] = df["meta_orcahello_confidence"]
    df["link"] = df["detection_id"].apply(lambda d: LINK_TEMPLATE.format(detection_id=d))

    for col in ("audio_uri", "spectrogram_uri"):
        if col not in df.columns:
            df[col] = ""

    output_cols = [
        "location_slug",
        "year_month_pacific",
        "date_hour_pacific",
        "timestamp_pacific",
        "detection_id",
        "binary_label",
        "example_type",
        "comments",
        "confidence_detector_v0",
        "confidence_detector_v1",
        "audio_uri",
        "spectrogram_uri",
        "link",
    ]
    return df[output_cols].sort_values(["year_month_pacific", "date_hour_pacific", "timestamp_pacific"])


def print_summary(sampled_df: pd.DataFrame):
    """Print summary statistics."""
    print(f"\nSampled dataset: {len(sampled_df)} rows")
    ct = sampled_df.groupby(["binary_label", "example_type"]).size().reset_index(name="count")
    for _, row in ct.iterrows():
        print(f"  label={row['binary_label']} type={row['example_type']}: {row['count']}")


def build_complete_df(args, months: list[str]) -> pd.DataFrame:
    """Load data and build the complete (pre-sampling) dataframe."""
    print(f"Loading detections for source={args.source}, months={months}")
    det_df, hourly = load_data(args.source, months, args.logbook_dir)
    print(f"  {len(det_df)} total detections, {len(hourly)} hourly events")
    print_location_summary(det_df, hourly)

    if args.location != "all":
        print(f"Filtering to location={args.location}")
        det_df = det_df[det_df["location_slug"] == args.location].copy()
        hourly = hourly[hourly["location_slug"] == args.location].copy()
    else:
        print("Using all locations")
    print(f"  {len(det_df)} detections, {len(hourly)} hourly events")
    if det_df.empty:
        return det_df

    det_df = join_hourly_info(det_df, hourly)

    if args.inference_version:
        print(f"Loading inference results ({args.inference_version})...")
        inf_df = load_inference_confidences(months, args.inference_version)
        print(f"  {len(inf_df)} inference results found")
        det_df = det_df.merge(inf_df, on="detection_id", how="left")
    else:
        print("No inference version specified, using confidence_detector_v0 for sampling")
        det_df["confidence_detector_v1"] = det_df["meta_orcahello_confidence"]

    # Join audio/spectrogram URIs from raw cache
    print(f"Loading URIs from cache ({args.cache_dir})...")
    uri_df = load_uris_from_cache(months, args.cache_dir)
    print(f"  {len(uri_df)} URIs loaded")
    det_df = det_df.merge(uri_df, on="detection_id", how="left")

    return det_df


def main():
    parser = argparse.ArgumentParser(description="Create a curated evaluation dataset")
    parser.add_argument("--logbook-dir", type=Path, default=DEFAULT_LOGBOOK_DIR,
                        help=f"Path to combined_logbook directory (default: {DEFAULT_LOGBOOK_DIR})")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help=f"Path to OrcaHello raw detection cache (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Directory to write output CSV files (default: {OUTPUT_DIR})")
    parser.add_argument("--location", default="port-townsend", help="Location slug to filter")
    parser.add_argument("--months", nargs="+", default=["2025-11:2026-02"],
                        help="Year-month values or colon-separated ranges (e.g. 2025-11:2026-02)")
    parser.add_argument("--source", default="orcahello_moderated", help="Detection source to filter")
    parser.add_argument("--target-positives", type=int, default=50, help="Target number of positive examples (cap)")
    parser.add_argument("--negative-ratio", type=float, default=2.0, help="Negative:positive ratio")
    parser.add_argument("--inference-version", default=None,
                        help="Inference version to join (e.g. v1.2). Default: use v0 detector confidence")
    parser.add_argument("--hard-pos-max-conf", type=float, default=0.7,
                        help="Max confidence for hard positives")
    parser.add_argument("--hard-pos-min-conf", type=float, default=0.1,
                        help="Min confidence for hard positives (exclude bad audio)")
    parser.add_argument("--hard-neg-min-conf", type=float, default=0.5,
                        help="Min confidence for hard negatives")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    months = expand_months(args.months)
    months_suffix = "_".join(args.months).replace(":", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    complete_path = args.output_dir / f"{months_suffix}--{args.location}--complete.csv"
    sampled_path = args.output_dir / f"{months_suffix}--{args.location}--sampled.csv"

    # Step 1: Load complete dataset (or use cached)
    if complete_path.exists():
        print(f"Loading cached complete dataset from {complete_path}")
        det_df = pd.read_csv(complete_path, dtype=str)
        # Restore numeric types needed for sampling
        det_df["binary_label"] = pd.to_numeric(det_df["binary_label"])
        det_df["meta_orcahello_confidence"] = pd.to_numeric(det_df["confidence_detector_v0"], errors="coerce")
        det_df["confidence_detector_v1"] = pd.to_numeric(det_df["confidence_detector_v1"], errors="coerce")
        # audio_uri / spectrogram_uri stay as str (already in CSV)
        for col in ("audio_uri", "spectrogram_uri"):
            if col not in det_df.columns:
                det_df[col] = ""
        print(f"  {len(det_df)} detections loaded from cache")
    else:
        det_df = build_complete_df(args, months)
        if det_df.empty:
            print("No detections found. Exiting.")
            sys.exit(0)
        complete_out = format_complete(det_df)
        complete_out.to_csv(complete_path, index=False)
        print(f"Wrote {len(complete_out)} rows to {complete_path}")

    # Step 2: Bias-sample per location
    print("Sampling...")
    sampled_parts = []
    for loc, loc_df in det_df.groupby("location_slug"):
        loc_sampled = bias_sample(
            loc_df,
            target_positives=args.target_positives,
            negative_ratio=args.negative_ratio,
            hard_pos_max_conf=args.hard_pos_max_conf,
            hard_pos_min_conf=args.hard_pos_min_conf,
            hard_neg_min_conf=args.hard_neg_min_conf,
            seed=args.seed,
            location=loc,
        )
        sampled_parts.append(loc_sampled)
    sampled = pd.concat(sampled_parts)

    # Step 3: Format and write sampled output
    output = format_output(sampled)
    output.to_csv(sampled_path, index=False)
    print(f"\nWrote {len(output)} rows to {sampled_path}")

    print_summary(sampled)


if __name__ == "__main__":
    main()
