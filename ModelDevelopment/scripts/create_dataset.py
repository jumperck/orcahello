"""Create a complete detection dataset CSV, with optional audio download."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.utils import build_complete_df, expand_months
from src.download import (
    DEFAULT_WORKERS,
    load_detections_from_csv,
    process_month_downloads,
    setup_logging,
    write_download_summary_txt,
)
from src.models import DetectionRecord, format_df

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOGBOOK_DIR = BASE_DIR / "combined_logbook"
DEFAULT_CACHE_DIR = BASE_DIR / "fetch_cache" / "orcahello"
OUTPUT_DIR = BASE_DIR / "datasets"


def main():
    parser = argparse.ArgumentParser(description="Create a complete detection dataset CSV")
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
    parser.add_argument("--download", action="store_true",
                        help="Download audio files after creating the CSV")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded (with --download)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        setup_logging(verbose=True)

    months = expand_months(args.months)
    months_suffix = "_".join(args.months).replace(":", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    complete_path = args.output_dir / f"{months_suffix}--{args.location}--complete.csv"

    # Build complete dataset (or use cached)
    if complete_path.exists():
        print(f"Loading cached complete dataset from {complete_path}")
        det_df = pd.read_csv(complete_path, dtype=str)
        print(f"  {len(det_df)} detections loaded from cache")
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
        complete_out = format_df(det_df, DetectionRecord)
        complete_out.to_csv(complete_path, index=False)
        print(f"Wrote {len(complete_out)} rows to {complete_path}")

    # Optional: download audio
    if args.download:
        print(f"\nDownloading audio files...")
        detections_by_month, df = load_detections_from_csv(complete_path)
        download_dir = complete_path.parent / complete_path.stem

        total_processed = 0
        total_downloaded = 0
        total_failed = 0

        for month in sorted(detections_by_month.keys()):
            dets = detections_by_month[month]
            print(f"\n--- {month} ({len(dets)} detections) ---")
            processed, downloaded, failed = process_month_downloads(
                output_dir=download_dir,
                month=month,
                detections=dets,
                workers=args.workers,
                dry_run=args.dry_run,
            )
            total_processed += processed
            total_downloaded += downloaded
            total_failed += failed

        if not args.dry_run:
            write_download_summary_txt(download_dir, df)

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Download complete!")
        print(f"  Processed: {total_processed}")
        print(f"  Downloaded: {total_downloaded}")
        print(f"  Failed: {total_failed}")
        print(f"  Output: {download_dir}")


if __name__ == "__main__":
    main()
