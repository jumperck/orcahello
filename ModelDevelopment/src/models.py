"""Pydantic schemas for detection/segment records and shared formatting helpers."""

from typing import Optional

import pandas as pd
from pydantic import BaseModel


LINK_TEMPLATE = "https://aifororcas.azurewebsites.net/detections/detection/{detection_id}"

LABEL_MAP = {"yes": 1, "no": 0}


class DetectionRecord(BaseModel):
    """Schema for a single detection in the complete CSV."""

    location_slug: str
    year_month_pacific: str
    date_hour_pacific: str
    timestamp_pacific: str
    detection_id: str
    binary_label: int
    global_confidence: Optional[float] = None
    comments: Optional[str] = None
    detection_link: str
    audio_uri: str = ""
    spectrogram_uri: str = ""


class SampledDetectionRecord(DetectionRecord):
    """Detection record with an additional example_type column for sampled datasets."""

    example_type: str


class SegmentRecord(BaseModel):
    """Schema for a segment-level row in the segmented CSV."""

    location_slug: str
    year_month_pacific: str
    date_hour_pacific: str
    timestamp_pacific: str
    detection_id: str
    detection_link: str
    segment_start_s: float
    segment_end_s: float
    segment_confidence: float
    segment_binary_label: int


def format_df(df: pd.DataFrame, model_cls: type[BaseModel]) -> pd.DataFrame:
    """Select and order columns based on a Pydantic model's field names.

    Adds detection_link if missing, fills missing URI columns with empty strings,
    and fills global_confidence from meta_orcahello_confidence if available.
    """
    df = df.copy()

    # Derive detection_link if absent
    if "detection_link" not in df.columns and "detection_id" in df.columns:
        df["detection_link"] = df["detection_id"].apply(
            lambda d: LINK_TEMPLATE.format(detection_id=d)
        )

    # Fill global_confidence from v0 score if needed
    if "meta_orcahello_confidence" in df.columns:
        df["global_confidence"] = df["global_confidence"].fillna(df["meta_orcahello_confidence"])

    # Ensure URI columns exist
    for col in ("audio_uri", "spectrogram_uri"):
        if col not in df.columns:
            df[col] = ""

    cols = list(model_cls.model_fields.keys())
    # Only select columns that exist in the dataframe
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(
        ["year_month_pacific", "date_hour_pacific", "timestamp_pacific"],
        na_position="last",
    )
