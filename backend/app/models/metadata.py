"""Standardized dataset metadata, status, and quality models (Section 4 & Section 10.1)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DataStatus(str, Enum):
    """Explicit data statuses mandated by Section 4.5."""

    AVAILABLE = "available"
    PROCESSING = "processing"
    UNAVAILABLE = "unavailable"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    FAILED = "failed"
    DEMO = "demo"


class ResultNature(str, Enum):
    """Explicit classification of whether a result is observed, derived, estimated, or illustrative."""

    OBSERVED = "observed"
    DERIVED = "derived"
    ESTIMATED = "estimated"
    ILLUSTRATIVE = "illustrative"


class QualityLevel(str, Enum):
    """Scientific quality level of a processed output."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class QualityMetadata(BaseModel):
    """Quality indicators, warnings, and scientific limitations."""

    level: QualityLevel = Field(default=QualityLevel.MEDIUM)
    observation_count: int = Field(default=0, ge=0)
    spatial_coverage_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    temporal_gap_days: Optional[int] = Field(default=None, ge=0)
    consistent_orbit_geometry: Optional[bool] = Field(default=None)
    confidence_method: Optional[str] = Field(
        default=None,
        description="Documented method used if a confidence indicator is provided; null if not calculated.",
    )
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    """Standardized metadata accompanying every processed output (Section 4.4 & 10.1)."""

    dataset: str = Field(..., description="Canonical dataset identifier (e.g., sentinel-1, gfw-alerts, nisar, gpm-imerg)")
    provider: str = Field(..., description="Official data provider (e.g., ESA / Copernicus, WRI / GFW, NASA / ISRO ASF DAAC, NASA GES DISC)")
    instrument: str = Field(..., description="Satellite or sensor instrument name")
    product_id: Optional[str] = Field(default=None, description="Official collection or product identifier")
    status: DataStatus = Field(..., description="Availability or processing status")
    result_nature: ResultNature = Field(
        default=ResultNature.DERIVED,
        description="Whether the result is observed, derived, estimated, or illustrative",
    )
    observation_start: Optional[str] = Field(default=None, description="Start date/time of observation period (ISO 8601)")
    observation_end: Optional[str] = Field(default=None, description="End date/time of observation period (ISO 8601)")
    baseline_start: Optional[str] = Field(default=None, description="Baseline reference start date if applicable")
    baseline_end: Optional[str] = Field(default=None, description="Baseline reference end date if applicable")
    acquisition_dates: List[str] = Field(default_factory=list, description="Exact acquisition timestamps/dates used")
    processing_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description="UTC timestamp when processing occurred",
    )
    region_id: str = Field(..., description="Study area identifier")
    geographic_coverage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Bounding box and area in km² of the processed region",
    )
    crs: str = Field(default="EPSG:4326", description="Coordinate Reference System of delivered output")
    area_calculation_crs: Optional[str] = Field(
        default="ESRI:102033",
        description="Equal-area or geodesic reference used for area calculations",
    )
    resolution_m: Optional[float] = Field(default=None, description="Nominal spatial resolution in meters when known")
    method: str = Field(..., description="Scientific processing or aggregation method applied")
    quality: QualityMetadata = Field(default_factory=QualityMetadata)
    extra: Dict[str, Any] = Field(default_factory=dict, description="Dataset-specific technical attributes")
