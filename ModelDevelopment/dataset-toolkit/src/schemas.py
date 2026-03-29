"""
Pydantic schemas for fine-tuning dataset rows.

Defines the row-level structure for both recording-level and segment-level
HF datasets used in SRKW detector fine-tuning.
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Shared types
# =============================================================================

LABEL_TO_TAG: dict[int, str] = {0: "srkw_negative", 1: "srkw_positive"}
TAG_TO_LABEL: dict[str, int] = {v: k for k, v in LABEL_TO_TAG.items()}


class Tag(BaseModel):
    """A scored tag on a recording (e.g. from a classifier or annotator)."""

    tag: str
    score: float = Field(ge=0.0, le=1.0, default=1.0)


class SegmentAnnotation(BaseModel):
    """A single annotated time-span within a recording."""

    start: float = Field(ge=0.0, description="Segment start in seconds")
    end: float = Field(gt=0.0, description="Segment end in seconds")
    tag: str = Field(description="Annotation tag, e.g. 'srkw_positive'")

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self

    @property
    def duration_s(self) -> float:
        return self.end - self.start


# =============================================================================
# Recording-level dataset row
# =============================================================================


class RecordingRow(BaseModel):
    """Schema for one row in the recording-level HF dataset.

    Stores the full 1-min audio file together with file-level metadata
    and a list of segment annotations.

    HF columns:
        audio        Audio()                         – full recording
        id           Value("string")                 – unique recording id
        tags         Sequence({tag, score})           – file-level tags
        comment      Value("string")                 – free-text note
        segment_annotations  Sequence({start, end, tag}) – per-segment labels
    """

    id: str
    tags: list[Tag] = Field(default_factory=list)
    comment: str = ""
    segment_annotations: list[SegmentAnnotation] = Field(default_factory=list)
    # `audio` is managed by HF Audio feature, not validated here


# =============================================================================
# Segment-level dataset row
# =============================================================================


class SegmentRow(BaseModel):
    """Schema for one row in the segment-level HF dataset.

    Each row is an individual annotated audio segment (possibly a chunk of a
    larger annotation that was broken up via max_segment_s).

    HF columns:
        audio      Audio()            – extracted segment waveform
        label      Value("int64")     – binary label (0 / 1)
        tag        Value("string")    – e.g. "srkw_positive"
        source_id  Value("string")    – recording id this segment came from
        start_s    Value("float32")   – start time in source recording
        end_s      Value("float32")   – end time in source recording
    """

    label: int = Field(ge=0, le=1)
    tag: str
    source_id: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    # `audio` is managed by HF Audio feature, not validated here

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.end_s <= self.start_s:
            raise ValueError(f"end_s ({self.end_s}) must be > start_s ({self.start_s})")
        return self

    @model_validator(mode="after")
    def _tag_matches_label(self):
        expected = LABEL_TO_TAG.get(self.label)
        if expected and self.tag != expected:
            raise ValueError(
                f"tag '{self.tag}' does not match label {self.label} "
                f"(expected '{expected}')"
            )
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# =============================================================================
# CSV input row (what the user provides)
# =============================================================================


class AnnotationCSVRow(BaseModel):
    """Schema for one row of the input CSV file.

    Expected CSV columns: filename, start_s, end_s, label
    """

    filename: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    label: int = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.end_s <= self.start_s:
            raise ValueError(f"end_s ({self.end_s}) must be > start_s ({self.start_s})")
        return self
