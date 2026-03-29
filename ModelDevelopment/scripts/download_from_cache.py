"""Download detection WAV and spectrogram files directly from OrcaHello cache.

Reads raw_detections.json from the fetch cache and organizes downloads
by moderation status:

    output_dir/
        YYYY-MM/
            positive/          # reviewed=True, found='yes'
            false_positive/    # reviewed=True, found='no'
            unmoderated/       # reviewed=False
            unknown/           # reviewed=True, found='don't know' or other
                <stem>--<id>.flac
                <stem>--<id>.png
            ground_truth_labels.csv
            summary.txt
"""

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from dataset_toolkit.utils import expand_months
from dataset_toolkit.download import (
    AUDIO_EXT,
    DEFAULT_WORKERS,
    create_http_session,
    download_png,
    download_wav_as_flac,
    setup_logging,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("./fetch_cache/orcahello")
DEFAULT_OUTPUT_DIR = Path("./detection_downloads")


def _get_last_month() -> str:
    now = datetime.now()
    if now.month == 1:
        return f"{now.year - 1}-12"
    return f"{now.year}-{now.month - 1:02d}"


def classify_detection(detection: dict) -> str:
    """Classify detection into category based on moderation status."""
    reviewed = detection.get("reviewed", False)
    found = detection.get("found", "").lower()

    if not reviewed:
        return "unmoderated"
    if found == "yes":
        return "positive"
    elif found == "no":
        return "false_positive"
    else:
        return "unknown"


def load_detections(cache_dir: Path, month: str) -> List[dict]:
    month_dir = cache_dir / month
    detections_file = month_dir / "raw_detections.json"

    if not detections_file.exists():
        logger.warning(f"No detections file for {month}: {detections_file}")
        return []

    with open(detections_file, "r") as f:
        return json.load(f)


def download_detection(
    detection: dict,
    output_dir: Path,
    month: str,
    download_wav: bool,
    download_spectrogram: bool,
    categories: Optional[List[str]],
    dry_run: bool,
    session,
) -> Tuple[str, bool]:
    """Download a single detection's files into category subfolders."""
    det_id = detection.get("id", "unknown")
    category = classify_detection(detection)

    if categories and category not in categories:
        return ("skipped", False)

    month_output = output_dir / month / category

    audio_url = detection.get("audioUri", "")
    spec_url = detection.get("spectrogramUri", "")

    wav_stem = audio_url.rsplit("/", 1)[-1].rsplit(".", 1)[0] if audio_url else det_id
    png_stem = spec_url.rsplit("/", 1)[-1].rsplit(".", 1)[0] if spec_url else det_id

    flac_name = f"{wav_stem}--{det_id}.flac"
    png_name = f"{png_stem}--{det_id}.png"

    flac_path = month_output / flac_name
    png_path = month_output / png_name

    if dry_run:
        return (f"[DRY RUN] {flac_name}", True)

    if download_wav and audio_url:
        if flac_path.exists():
            logger.debug(f"Skipping existing: {flac_name}")
        else:
            if download_wav_as_flac(session, audio_url, flac_path):
                logger.debug(f"Downloaded: {flac_name}")
            else:
                return (f"Failed WAV: {flac_name}", False)

    if download_spectrogram and spec_url:
        if png_path.exists():
            logger.debug(f"Skipping existing: {png_name}")
        else:
            if download_png(session, spec_url, png_path):
                logger.debug(f"Downloaded: {png_name}")
            else:
                return (f"Failed PNG: {png_name}", False)

    return ("ok", True)


def process_month(
    cache_dir: Path,
    output_dir: Path,
    month: str,
    download_wav: bool,
    download_spectrogram: bool,
    categories: Optional[List[str]],
    workers: int,
    dry_run: bool,
) -> Tuple[int, int, int]:
    """Process detections for a single month.

    Returns (processed_count, downloaded_count, skipped_count).
    """
    detections = load_detections(cache_dir, month)
    if not detections:
        return 0, 0, 0

    processed = 0
    downloaded = 0
    skipped = 0

    session = create_http_session()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_detection, det, output_dir, month,
                download_wav, download_spectrogram, categories, dry_run, session,
            )
            for det in detections
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=month, unit="detection"):
            status, success = future.result()
            processed += 1
            if status == "skipped":
                skipped += 1
            elif success:
                downloaded += 1

    return processed, downloaded, skipped


def write_ground_truth_csv(month_dir: Path) -> None:
    """Write ground_truth_labels.csv from positive/ and false_positive/ audio files."""
    label_folders = ("positive", "false_positive")
    rows: List[Tuple[str, int, str]] = []
    ext = AUDIO_EXT.lower()
    for folder in label_folders:
        dirpath = month_dir / folder
        if not dirpath.is_dir():
            continue
        label_binary = 1 if folder == "positive" else 0
        for entry in sorted(dirpath.iterdir()):
            if not entry.is_file() or entry.suffix.lower() != ext:
                continue
            rel_path = f"{folder}/{entry.name}"
            rows.append((rel_path, label_binary, folder))
    outpath = month_dir / "ground_truth_labels.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_path", "label_binary", "label"])
        w.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {outpath}")


def write_summary_txt(month_dir: Path) -> None:
    """Write summary.txt with audio file counts per subfolder."""
    ext = AUDIO_EXT.lower()
    counts: List[Tuple[str, int]] = []
    total = 0
    for subdir in sorted(month_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        n = sum(1 for e in subdir.iterdir() if e.is_file() and e.suffix.lower() == ext)
        counts.append((subdir.name, n))
        total += n
    lines = [f"Audio ({AUDIO_EXT}) counts per subfolder:", ""]
    lines.extend(f"{name}: {n}" for name, n in counts)
    lines.append(f"Total: {total}")
    outpath = month_dir / "summary.txt"
    outpath.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote summary to {outpath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download detection files from OrcaHello raw cache"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"OrcaHello cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--months", nargs="+", default=None,
        help="Month(s) to download: single (2025-07), range (2025-01:2025-11), or list. Default: last month",
    )
    parser.add_argument("--no-wav", action="store_true", help="Skip downloading WAV files")
    parser.add_argument("--no-spec", action="store_true", help="Skip downloading spectrogram PNGs")
    parser.add_argument(
        "--category",
        type=str,
        action="append",
        choices=["positive", "false_positive", "unmoderated", "unknown"],
        help="Filter by category (can specify multiple). Default: all",
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

    logger.info("OrcaHello Cache Downloader")

    if not args.cache_dir.exists():
        logger.error(f"Cache directory does not exist: {args.cache_dir}")
        return 1
    logger.info(f"Cache directory: {args.cache_dir}")

    output_dir = args.output_dir
    logger.info(f"Output directory: {output_dir}")

    download_wav = not args.no_wav
    download_spectrogram = not args.no_spec
    categories = args.category

    if not download_wav and not download_spectrogram:
        logger.error("Nothing to download: both --no-wav and --no-spec specified")
        return 1

    if args.months:
        months = expand_months(args.months)
    else:
        months = [_get_last_month()]

    logger.info(f"Months: {months}")
    logger.info(f"Download WAV (as FLAC): {download_wav}")
    logger.info(f"Download spectrogram: {download_spectrogram}")
    logger.info(f"Categories: {categories or 'all'}")
    logger.info(f"Workers: {args.workers}")

    if args.dry_run:
        logger.info("[DRY RUN MODE]")

    total_processed = 0
    total_downloaded = 0
    total_skipped = 0

    for month in months:
        logger.info(f"\n--- Processing {month} ---")

        processed, downloaded, skipped = process_month(
            cache_dir=args.cache_dir,
            output_dir=output_dir,
            month=month,
            download_wav=download_wav,
            download_spectrogram=download_spectrogram,
            categories=categories,
            workers=args.workers,
            dry_run=args.dry_run,
        )

        logger.info(f"{month}: processed={processed}, downloaded={downloaded}, skipped={skipped}")
        total_processed += processed
        total_downloaded += downloaded
        total_skipped += skipped

        if not args.dry_run:
            month_dir = output_dir / month
            if month_dir.exists():
                write_ground_truth_csv(month_dir)
                write_summary_txt(month_dir)

    logger.info(f"\n{'[DRY RUN] ' if args.dry_run else ''}Download complete!")
    logger.info(f"  Total processed: {total_processed}")
    logger.info(f"  Total downloaded: {total_downloaded}")
    logger.info(f"  Total skipped (filtered): {total_skipped}")
    logger.info(f"  Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
