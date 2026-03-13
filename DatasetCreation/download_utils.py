#!/usr/bin/env python3
"""Shared utilities for downloading OrcaHello detection audio and spectrograms."""

import io
import logging
from pathlib import Path

import requests
import soundfile as sf
from requests.adapters import HTTPAdapter
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
