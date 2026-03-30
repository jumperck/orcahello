"""Create a recording-level HF Dataset from the OrcaHello detection logbook."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

from dataset_toolkit.download import (
    DEFAULT_WORKERS,
    load_detections_from_csv,
    process_month_downloads,
    setup_logging,
    write_download_summary_txt,
)
from dataset_toolkit.hf_dataset import build_recording_dataset
from dataset_toolkit.models import LINK_TEMPLATE
from dataset_toolkit.utils import build_complete_df, expand_months

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOGBOOK_DIR = BASE_DIR / "combined_logbook"
DEFAULT_CACHE_DIR = BASE_DIR / "fetch_cache" / "orcahello"
OUTPUT_DIR = BASE_DIR / "datasets"


def _export_csv(dataset, output_path: Path) -> None:
    """Export a recording-level HF Dataset to a flat CSV (backward compat)."""
    rows = []
    for ex in dataset:
        tags = ex["tags"]
        # HF Sequence stores as dict-of-lists: {"tag": [...], "score": [...]}
        has_tags = tags and tags.get("tag") and len(tags["tag"]) > 0
        tag = tags["tag"][0] if has_tags else ""
        score = tags["score"][0] if has_tags else None
        meta = ex["metadata"]
        rows.append({
            "location_slug": meta["location_slug"],
            "year_month_pacific": meta["year_month_pacific"],
            "date_hour_pacific": meta["date_hour_pacific"],
            "timestamp_pacific": meta["timestamp_pacific"],
            "detection_id": ex["recording_id"],
            "binary_label": 1 if tag == "srkw_positive" else 0,
            "global_confidence": score,
            "comments": ex["comment"],
            "detection_link": LINK_TEMPLATE.format(detection_id=ex["recording_id"]),
            "audio_uri": meta["audio_uri"],
            "spectrogram_uri": meta["spectrogram_uri"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["year_month_pacific", "date_hour_pacific", "timestamp_pacific"],
        na_position="last",
    )
    df.to_csv(output_path, index=False)
    print(f"Exported CSV: {output_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Create a recording-level HF Dataset from the OrcaHello detection logbook"
    )
    parser.add_argument("--logbook-dir", type=Path, default=DEFAULT_LOGBOOK_DIR,
                        help=f"Path to combined_logbook directory (default: {DEFAULT_LOGBOOK_DIR})")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help=f"Path to OrcaHello raw detection cache (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Directory to write output datasets (default: {OUTPUT_DIR})")
    parser.add_argument("--location", default="port-townsend", help="Location slug to filter")
    parser.add_argument("--months", nargs="+", default=["2025-11:2026-02"],
                        help="Year-month values or colon-separated ranges (e.g. 2025-11:2026-02)")
    parser.add_argument("--source", default="orcahello_moderated", help="Detection source to filter")
    parser.add_argument("--download", action="store_true",
                        help="Download audio files after creating the dataset")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded (with --download)")
    parser.add_argument("--export-csv", action="store_true",
                        help="Also export a flat CSV alongside the HF Dataset")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if recording_dataset/ already exists")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        setup_logging(verbose=True)

    months = expand_months(args.months)
    months_suffix = "_".join(args.months).replace(":", "_")

    dataset_dir = args.output_dir / f"{months_suffix}--{args.location}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    recording_dataset_path = dataset_dir / "recording_dataset"
    audio_dir = dataset_dir / "audio"

    # Build or load recording dataset
    if recording_dataset_path.exists() and not args.force:
        print(f"Loading cached recording dataset from {recording_dataset_path}")
        dataset = load_from_disk(str(recording_dataset_path))
        print(f"  {len(dataset)} recordings loaded from cache")
    else:
        det_df = build_complete_df(
            source=args.source,
            months=months,
            location=args.location,
            logbook_dir=args.logbook_dir,
            cache_dir=args.cache_dir,
        )
        if det_df.empty:
            print("No detections found. Exiting.")
            sys.exit(0)

        # Build HF Dataset (audio_dir=None initially; populated after download)
        existing_audio = audio_dir if audio_dir.exists() else None
        dataset = build_recording_dataset(det_df, audio_dir=existing_audio)
        dataset.save_to_disk(str(recording_dataset_path))
        print(f"Saved recording dataset: {recording_dataset_path} ({len(dataset)} recordings)")

    # Optional: export CSV
    if args.export_csv:
        csv_path = dataset_dir / f"{months_suffix}--{args.location}--complete.csv"
        _export_csv(dataset, csv_path)

    # Optional: download audio
    if args.download:
        print(f"\nDownloading audio files...")

        # Export a temporary CSV for the download helpers
        tmp_csv = dataset_dir / ".tmp_download.csv"
        _export_csv(dataset, tmp_csv)

        detections_by_month, df = load_detections_from_csv(tmp_csv)

        total_processed = 0
        total_downloaded = 0
        total_failed = 0

        for month in sorted(detections_by_month.keys()):
            dets = detections_by_month[month]
            print(f"\n--- {month} ({len(dets)} detections) ---")
            processed, downloaded, failed = process_month_downloads(
                output_dir=audio_dir,
                month=month,
                detections=dets,
                workers=args.workers,
                dry_run=args.dry_run,
            )
            total_processed += processed
            total_downloaded += downloaded
            total_failed += failed

        if not args.dry_run:
            write_download_summary_txt(audio_dir, df)

            # Re-build dataset with audio paths populated
            print("\nUpdating dataset with audio paths...")
            det_df = build_complete_df(
                source=args.source,
                months=months,
                location=args.location,
                logbook_dir=args.logbook_dir,
                cache_dir=args.cache_dir,
            )
            dataset = build_recording_dataset(det_df, audio_dir=audio_dir)
            dataset.save_to_disk(str(recording_dataset_path))
            print(f"Updated recording dataset with audio paths")

        tmp_csv.unlink(missing_ok=True)

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Download complete!")
        print(f"  Processed: {total_processed}")
        print(f"  Downloaded: {total_downloaded}")
        print(f"  Failed: {total_failed}")
        print(f"  Audio dir: {audio_dir}")


if __name__ == "__main__":
    main()
