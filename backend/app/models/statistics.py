"""Structured statistics and timeline models (Section 10.3 & 13)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.metadata import DataStatus, DatasetMetadata, QualityMetadata, ResultNature


class StatisticItem(BaseModel):
    """Single contextualized statistic (never a bare number without context)."""

    metric_id: str
    label: str
    value: Optional[float | int] = Field(
        default=None,
        description="Numeric value, or null when observations are unavailable/insufficient.",
    )
    unit: str
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    region_id: str
    source: str
    processing_method: str
    result_nature: ResultNature = ResultNature.DERIVED
    quality: QualityMetadata
    status: DataStatus
    comparison: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional comparison against baseline or previous period (delta, pct_change, period).",
    )
    notes: Optional[str] = None


class StatisticsResponse(BaseModel):
    """Envelope for dataset or combined statistics."""

    dataset: str
    region_id: str
    status: DataStatus
    metadata: DatasetMetadata
    statistics: List[StatisticItem] = Field(default_factory=list)
    breakdown: Dict[str, Any] = Field(default_factory=dict)


class TimelineEntry(BaseModel):
    """Single timestamped entry in a dataset or unified timeline."""

    timestamp: str = Field(..., description="ISO 8601 date or timestamp")
    end_timestamp: Optional[str] = Field(
        default=None,
        description="End of observation interval when entry represents a period rather than an instant",
    )
    dataset: str
    event_type: str
    value: Optional[float | int] = None
    unit: Optional[str] = None
    label: str
    product_id: Optional[str] = None
    status: DataStatus = DataStatus.AVAILABLE
    details: Dict[str, Any] = Field(default_factory=dict)


class TimelineResponse(BaseModel):
    """Time-series and observation timeline response."""

    dataset: str
    region_id: str
    status: DataStatus
    unit: Optional[str] = None
    temporal_resolution: Optional[str] = None
    metadata: DatasetMetadata
    series: List[TimelineEntry] = Field(default_factory=list)
