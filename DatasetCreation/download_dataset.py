#!/usr/bin/env python3
"""Download dataset audio files from a create_dataset.py CSV.

Reads a complete-*.csv or sampled-*.csv and downloads audio as FLAC,
organized by month:

    <output>/
        YYYY-MM/
            audio/
                <detection_id>.flac
        summary.txt
"""

import argparse
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from download_utils import (
    AUDIO_EXT,
    DEFAULT_WORKERS,
    create_http_session,
    download_wav_as_flac,
    setup_logging,
)

logger = logging.getLogger(__name__)


def load_detections_from_csv(csv_path: Path) -> Tuple[Dict[str, List[dict]], pd.DataFrame]:
    """Load detection records from a create_dataset.py CSV.

    Returns (month -> list of detection dicts, full dataframe).
    """
    df = pd.read_csv(csv_path, dtype=str)

    required = {"detection_id", "year_month_pacific", "audio_uri"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing columns: {missing}. "
            "Re-run create_dataset.py with --cache-dir to regenerate."
        )

    result: Dict[str, List[dict]] = {}
    for month, group in df.groupby("year_month_pacific"):
        dets = []
        for _, row in group.iterrows():
            audio_uri = row.get("audio_uri", "")
            dets.append({
                "id": row["detection_id"],
                "audioUri": audio_uri if isinstance(audio_uri, str) else "",
            })
        result[month] = dets

    return result, df


def download_one(
    detection: dict,
    output_dir: Path,
    month: str,
    dry_run: bool,
) -> Tuple[str, bool]:
    """Download a single detection's audio file.

    Output: <output_dir>/<month>/audio/<detection_id>.flac
    Returns (status_message, success_flag).
    """
    det_id = detection.get("id", "unknown")
    audio_url = detection.get("audioUri", "")

    flac_path = output_dir / month / "audio" / f"{det_id}{AUDIO_EXT}"

    if dry_run:
        return (f"[DRY RUN] {det_id}{AUDIO_EXT}", True)

    if not audio_url:
        logger.warning(f"No audio URI for {det_id}, skipping")
        return ("no_uri", False)

    if flac_path.exists():
        logger.debug(f"Skipping existing: {det_id}{AUDIO_EXT}")
        return ("ok", True)

    session = create_http_session()
    if download_wav_as_flac(session, audio_url, flac_path):
        logger.debug(f"Downloaded: {det_id}{AUDIO_EXT}")
        return ("ok", True)
    else:
        return (f"Failed: {det_id}{AUDIO_EXT}", False)


def process_month(
    output_dir: Path,
    month: str,
    detections: List[dict],
    workers: int,
    dry_run: bool,
) -> Tuple[int, int, int]:
    """Download detections for a single month.

    Returns (processed, downloaded, failed).
    """
    if not detections:
        return 0, 0, 0

    processed = 0
    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_one, det, output_dir, month, dry_run)
            for det in detections
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=month, unit="det"):
            status, success = future.result()
            processed += 1
            if success:
                downloaded += 1
            else:
                failed += 1

    return processed, downloaded, failed


def write_summary_txt(output_dir: Path, df: pd.DataFrame) -> None:
    """Write summary.txt with audio counts by year-month and location."""
    ext = AUDIO_EXT.lower()
    lines = [f"Audio ({AUDIO_EXT}) download summary", ""]

    # Build location lookup from CSV
    has_location = "location_slug" in df.columns

    total_all = 0
    for month_dir in sorted(output_dir.iterdir()):
        if not month_dir.is_dir() or not month_dir.name[:4].isdigit():
            continue
        month = month_dir.name
        audio_dir = month_dir / "audio"
        if not audio_dir.is_dir():
            continue

        files = [e for e in audio_dir.iterdir() if e.is_file() and e.suffix.lower() == ext]
        n = len(files)
        total_all += n
        lines.append(f"{month}: {n}")

        if has_location:
            loc_counts: Dict[str, int] = defaultdict(int)
            month_df = df[df["year_month_pacific"] == month]
            loc_by_id = dict(zip(month_df["detection_id"], month_df["location_slug"]))
            for f in files:
                det_id = f.stem
                loc = loc_by_id.get(det_id, "unknown")
                loc_counts[loc] += 1
            for loc, cnt in sorted(loc_counts.items()):
                lines.append(f"  {loc}: {cnt}")
        lines.append("")

    lines.append(f"Total: {total_all}")
    outpath = output_dir / "summary.txt"
    outpath.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote summary to {outpath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download dataset audio from a create_dataset.py CSV"
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to a complete-*.csv or sampled-*.csv from create_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <csv-stem>/ next to the CSV)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel download threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    csv_path = args.input_csv.resolve()
    if not csv_path.exists():
        logger.error(f"Input CSV not found: {csv_path}")
        return 1

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = csv_path.parent / csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output: {output_dir}")

    detections_by_month, df = load_detections_from_csv(csv_path)
    months = sorted(detections_by_month.keys())
    logger.info(f"Months: {months}")

    total_processed = 0
    total_downloaded = 0
    total_failed = 0

    for month in months:
        dets = detections_by_month[month]
        logger.info(f"\n--- {month} ({len(dets)} detections) ---")

        processed, downloaded, failed = process_month(
            output_dir=output_dir,
            month=month,
            detections=dets,
            workers=args.workers,
            dry_run=args.dry_run,
        )

        logger.info(f"{month}: processed={processed}, downloaded={downloaded}, failed={failed}")
        total_processed += processed
        total_downloaded += downloaded
        total_failed += failed

    if not args.dry_run:
        write_summary_txt(output_dir, df)

    logger.info(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done!")
    logger.info(f"  Processed: {total_processed}")
    logger.info(f"  Downloaded: {total_downloaded}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
