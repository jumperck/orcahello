#!/usr/bin/env python3
"""Download detection WAV and spectrogram files from OrcaHello cache.

Can operate in two modes:
  1. Classic mode (--month / --from-month / --to-month): reads raw_detections.json from cache.
  2. CSV mode (--input-csv): reads detection_ids from a complete-*.csv or sampled-*.csv file.

Organizes downloads into subfolders by moderation status:
  - positive: reviewed=True and found='yes' (confirmed whale calls)
  - false_positive: reviewed=True and found='No'/'no' (not whales)
  - unmoderated: reviewed=False (pending review)
  - unknown: reviewed=True but found='don't know' or other

Directory structure (CSV mode, output rooted at <csv-stem>/):
  <csv-stem>/
    YYYY-MM/
      positive/
        <stem>--<id>.flac
        <stem>--<id>.png
      false_positive/
        ...
      unmoderated/
        ...
      unknown/
        ...
    labels.csv       # all months combined (positive/false_positive audio files)
    summary.txt      # breakdown by year-month and location

Directory structure (classic mode, output rooted at --output-dir):
  output_dir/
    YYYY-MM/
      positive/ false_positive/ unmoderated/ unknown/
      ground_truth_labels.csv
      summary.txt
"""

import argparse
import csv
import io
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import soundfile as sf
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("./fetch_cache/orcahello")
DEFAULT_OUTPUT_DIR = Path("./detection_downloads")
DEFAULT_WORKERS = 8
AUDIO_EXT = ".flac"


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_http_session() -> requests.Session:
    """Create HTTP session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_current_month() -> str:
    """Get current month as YYYY-MM."""
    return datetime.now().strftime("%Y-%m")


def get_last_month() -> str:
    """Get last month as YYYY-MM."""
    now = datetime.now()
    if now.month == 1:
        return f"{now.year - 1}-12"
    return f"{now.year}-{now.month - 1:02d}"


def parse_month(month_str: str) -> Tuple[int, int]:
    """Parse YYYY-MM string to (year, month)."""
    parts = month_str.split("-")
    return int(parts[0]), int(parts[1])


def get_months_in_range(start_month: str, end_month: str) -> List[str]:
    """Get list of months in range (inclusive)."""
    start_year, start_m = parse_month(start_month)
    end_year, end_m = parse_month(end_month)

    months = []
    year, month = start_year, start_m
    while (year, month) <= (end_year, end_m):
        months.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


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
    """Load detections from cache for a month."""
    month_dir = cache_dir / month
    detections_file = month_dir / "raw_detections.json"

    if not detections_file.exists():
        logger.warning(f"No detections file for {month}: {detections_file}")
        return []

    with open(detections_file, "r") as f:
        return json.load(f)


def load_detections_from_csv(csv_path: Path) -> Dict[str, List[dict]]:
    """Load detection records from a complete-*.csv or sampled-*.csv produced by create_dataset.py.

    Requires audio_uri and spectrogram_uri columns (present in both complete and sampled CSVs).
    Returns a dict mapping year_month -> list of detection dicts compatible with download_detection().
    """
    df = pd.read_csv(csv_path, dtype=str)

    required = {"detection_id", "year_month_pacific", "audio_uri", "spectrogram_uri", "binary_label"}
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
            # binary_label=1 → reviewed=True, found='yes' (positive)
            # binary_label=0 → reviewed=True, found='no'  (false_positive)
            binary = str(row.get("binary_label", "")).strip()
            reviewed = binary in ("0", "1")
            found = "yes" if binary == "1" else ("no" if binary == "0" else "")
            audio_uri = row.get("audio_uri", "")
            spec_uri = row.get("spectrogram_uri", "")
            dets.append({
                "id": row["detection_id"],
                "audioUri": audio_uri if isinstance(audio_uri, str) else "",
                "spectrogramUri": spec_uri if isinstance(spec_uri, str) else "",
                "reviewed": reviewed,
                "found": found,
            })
        result[month] = dets
        logger.debug(f"{month}: {len(dets)} detections from CSV")

    return result


def download_wav_as_flac(
    session: requests.Session,
    url: str,
    output_path: Path,
    timeout: int = 30,
) -> bool:
    """Download WAV from URL, convert to FLAC in memory, write to output_path."""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        wav_bytes = io.BytesIO(response.content)
        data, samplerate = sf.read(wav_bytes)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, data, samplerate, format="FLAC")

        return True
    except Exception as e:
        logger.error(f"Failed to download/convert {url}: {e}")
        return False


def download_png(
    session: requests.Session,
    url: str,
    output_path: Path,
    timeout: int = 30,
) -> bool:
    """Download PNG from URL to output_path."""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_detection(
    detection: dict,
    output_dir: Path,
    month: str,
    download_wav: bool,
    download_spectrogram: bool,
    categories: Optional[List[str]],
    dry_run: bool,
) -> Tuple[str, bool]:
    """Download a single detection's files.

    Returns (status_message, success_flag)
    """
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

    session = create_http_session()

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
    detections_override: Optional[List[dict]] = None,
) -> Tuple[int, int, int]:
    """Process detections for a single month using batched parallel downloads.

    Returns (processed_count, downloaded_count, skipped_count)
    """
    detections = detections_override if detections_override is not None else load_detections(cache_dir, month)
    if not detections:
        return 0, 0, 0

    processed = 0
    downloaded = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for detection in detections:
            future = executor.submit(
                download_detection,
                detection,
                output_dir,
                month,
                download_wav,
                download_spectrogram,
                categories,
                dry_run,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc=month, unit="detection"):
            status, success = future.result()
            processed += 1
            if status == "skipped":
                skipped += 1
            elif success:
                downloaded += 1

    return processed, downloaded, skipped


def write_ground_truth_csv(month_dir: Path) -> None:
    """Write ground_truth_labels.csv from positive/ and false_positive/ (audio files only)."""
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


def write_labels_csv(output_dir: Path) -> None:
    """Write labels.csv combining all months (positive/false_positive audio files only)."""
    ext = AUDIO_EXT.lower()
    label_folders = ("positive", "false_positive")
    rows: List[Tuple[str, str, int, str]] = []  # (month, file_path, label_binary, label)

    for month_dir in sorted(output_dir.iterdir()):
        if not month_dir.is_dir() or not month_dir.name[:4].isdigit():
            continue
        month = month_dir.name
        for folder in label_folders:
            dirpath = month_dir / folder
            if not dirpath.is_dir():
                continue
            label_binary = 1 if folder == "positive" else 0
            for entry in sorted(dirpath.iterdir()):
                if not entry.is_file() or entry.suffix.lower() != ext:
                    continue
                rel_path = f"{month}/{folder}/{entry.name}"
                rows.append((month, rel_path, label_binary, folder))

    outpath = output_dir / "labels.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year_month", "file_path", "label_binary", "label"])
        w.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {outpath}")


def write_combined_summary_txt(output_dir: Path, csv_path: Optional[Path] = None) -> None:
    """Write summary.txt with audio counts by year-month and location.

    Uses the detection CSV (if provided) to cross-reference location_slug.
    Falls back to file counts per month/category if no CSV.
    """
    ext = AUDIO_EXT.lower()
    lines = [f"Audio ({AUDIO_EXT}) counts by year-month and category", ""]

    # Load location mapping from CSV if available
    loc_by_id: Dict[str, str] = {}
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)
        if "detection_id" in df.columns and "location_slug" in df.columns:
            loc_by_id = dict(zip(df["detection_id"], df["location_slug"]))

    # Collect month-level stats
    total_all = 0
    for month_dir in sorted(output_dir.iterdir()):
        if not month_dir.is_dir() or not month_dir.name[:4].isdigit():
            continue
        month = month_dir.name
        month_total = 0
        month_lines = []
        loc_counts: Dict[str, int] = defaultdict(int)

        for subdir in sorted(month_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            files = [e for e in subdir.iterdir() if e.is_file() and e.suffix.lower() == ext]
            n = len(files)
            month_lines.append(f"  {subdir.name}: {n}")
            month_total += n

            # Map files to locations via detection_id in filename (stem--uuid.flac)
            if loc_by_id:
                for f in files:
                    # Extract UUID: last part after '--'
                    parts = f.stem.rsplit("--", 1)
                    if len(parts) == 2:
                        det_id = parts[1]
                        loc = loc_by_id.get(det_id, "unknown")
                        loc_counts[loc] += 1

        lines.append(f"{month}: {month_total} total")
        lines.extend(month_lines)
        if loc_counts:
            lines.append("  by location:")
            for loc, cnt in sorted(loc_counts.items()):
                lines.append(f"    {loc}: {cnt}")
        lines.append("")
        total_all += month_total

    lines.append(f"Grand total: {total_all}")
    outpath = output_dir / "summary.txt"
    outpath.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote combined summary to {outpath}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download detection WAV and spectrogram files from OrcaHello cache"
    )

    # CSV input mode
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Path to a complete-*.csv or sampled-*.csv from create_dataset.py. "
             "When provided, downloads only the detections listed in the CSV. "
             "Output is rooted at <csv-stem>/ in the same directory as the CSV.",
    )

    # Cache/output directories
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"OrcaHello cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for downloads (default: {DEFAULT_OUTPUT_DIR} in classic mode, "
             f"or <csv-stem>/ next to the input CSV in CSV mode)",
    )

    # Month range selection (classic mode only)
    parser.add_argument(
        "--month",
        type=str,
        help="Single month to process (YYYY-MM). Default: last month",
    )
    parser.add_argument(
        "--from-month",
        type=str,
        help="Start month for range (YYYY-MM)",
    )
    parser.add_argument(
        "--to-month",
        type=str,
        help="End month for range (YYYY-MM)",
    )

    # Download options
    parser.add_argument(
        "--no-wav",
        action="store_true",
        help="Skip downloading WAV files",
    )
    parser.add_argument(
        "--no-spec",
        action="store_true",
        help="Skip downloading spectrogram PNG files",
    )

    # Category filter
    parser.add_argument(
        "--category",
        type=str,
        action="append",
        choices=["positive", "false_positive", "unmoderated", "unknown"],
        help="Filter by category (can specify multiple). Default: all categories",
    )

    # Parallelism options
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel download threads (default: {DEFAULT_WORKERS})",
    )

    # Output options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("OrcaHello Detection Downloader")

    # In CSV mode the cache is not used; only validate it for classic mode
    if not args.input_csv and not args.cache_dir.exists():
        logger.error(f"Cache directory does not exist: {args.cache_dir}")
        return 1
    if not args.input_csv:
        logger.info(f"Cache directory: {args.cache_dir}")

    download_wav = not args.no_wav
    download_spectrogram = not args.no_spec
    categories = args.category

    if not download_wav and not download_spectrogram:
        logger.error("Nothing to download: both --no-wav and --no-spec specified")
        return 1

    # -----------------------------------------------------------------------
    # CSV mode
    # -----------------------------------------------------------------------
    if args.input_csv:
        csv_path = args.input_csv.resolve()
        if not csv_path.exists():
            logger.error(f"Input CSV not found: {csv_path}")
            return 1

        # Output rooted at <csv-stem>/ next to the CSV (or --output-dir if given)
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = csv_path.parent / csv_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CSV mode: {csv_path}")
        logger.info(f"Output directory: {output_dir}")

        # Load detection records directly from CSV (audio_uri/spectrogram_uri columns)
        detections_by_month = load_detections_from_csv(csv_path)
        months = sorted(detections_by_month.keys())
        logger.info(f"Months in CSV: {months}")

        total_processed = 0
        total_downloaded = 0
        total_skipped = 0

        for month in months:
            month_dets = detections_by_month[month]
            logger.info(f"\n--- Processing {month} ({len(month_dets)} detections) ---")

            processed, downloaded, skipped = process_month(
                cache_dir=args.cache_dir,
                output_dir=output_dir,
                month=month,
                download_wav=download_wav,
                download_spectrogram=download_spectrogram,
                categories=categories,
                workers=args.workers,
                dry_run=args.dry_run,
                detections_override=month_dets,
            )

            logger.info(f"{month}: processed={processed}, downloaded={downloaded}, skipped={skipped}")
            total_processed += processed
            total_downloaded += downloaded
            total_skipped += skipped

        if not args.dry_run:
            write_labels_csv(output_dir)
            write_combined_summary_txt(output_dir, csv_path=csv_path)

        logger.info(f"\n{'[DRY RUN] ' if args.dry_run else ''}Download complete!")
        logger.info(f"  Total processed: {total_processed}")
        logger.info(f"  Total downloaded: {total_downloaded}")
        logger.info(f"  Total skipped (filtered): {total_skipped}")
        logger.info(f"  Output directory: {output_dir}")
        return 0

    # -----------------------------------------------------------------------
    # Classic mode
    # -----------------------------------------------------------------------
    output_dir = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    logger.info(f"Output directory: {output_dir}")

    if args.month:
        months = [args.month]
    elif args.from_month and args.to_month:
        months = get_months_in_range(args.from_month, args.to_month)
    elif args.from_month:
        months = get_months_in_range(args.from_month, get_current_month())
    elif args.to_month:
        months = [args.to_month]
    else:
        months = [get_last_month()]

    logger.info(f"Months to process: {months}")
    logger.info(f"Download WAV (as FLAC): {download_wav}")
    logger.info(f"Download spectrogram: {download_spectrogram}")
    if categories:
        logger.info(f"Categories: {categories}")
    else:
        logger.info("Categories: all")
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
