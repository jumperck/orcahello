"""HTTP utilities and download orchestration for OrcaHello detection audio."""

import io
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import soundfile as sf
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Dataset CSV download helpers (from download_dataset.py)
# ---------------------------------------------------------------------------

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
    session: requests.Session,
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

    if download_wav_as_flac(session, audio_url, flac_path):
        logger.debug(f"Downloaded: {det_id}{AUDIO_EXT}")
        return ("ok", True)
    else:
        return (f"Failed: {det_id}{AUDIO_EXT}", False)


def process_month_downloads(
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

    session = create_http_session()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_one, det, output_dir, month, dry_run, session)
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


def write_download_summary_txt(output_dir: Path, df: pd.DataFrame) -> None:
    """Write summary.txt with audio counts by year-month and location."""
    ext = AUDIO_EXT.lower()
    lines = [f"Audio ({AUDIO_EXT}) download summary", ""]

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
