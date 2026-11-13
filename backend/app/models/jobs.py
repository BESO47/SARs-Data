"""Models for asynchronous processing jobs (Section 12 & 15)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.metadata import DataStatus


class JobStatus(str, Enum):
    """Lifecycle states of an asynchronous geospatial processing job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingJobRequest(BaseModel):
    """Payload to submit an asynchronous satellite processing job."""

    job_type: str = Field(
        default="combined_analysis",
        description="Job pipeline type: sentinel1_flood, gfw_forest, nisar_lband, gpm_rainfall, or combined_analysis",
    )
    region_id: str = Field(default="madeira")
    bbox: Optional[List[float]] = None
    polygon_geojson: Optional[Dict[str, Any]] = None
    start_date: str = Field(default="2024-03-01", description="Observation start date (YYYY-MM-DD)")
    end_date: str = Field(default="2024-04-15", description="Observation end date (YYYY-MM-DD)")
    baseline_start_date: Optional[str] = Field(
        default="2023-08-01",
        description="Dry-season or reference baseline start date (YYYY-MM-DD)",
    )
    baseline_end_date: Optional[str] = Field(
        default="2023-10-15",
        description="Dry-season or reference baseline end date (YYYY-MM-DD)",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ProcessingJobInfo(BaseModel):
    """Status and metadata of a background processing job."""

    job_id: str
    job_type: str
    status: JobStatus
    data_status: DataStatus = DataStatus.PROCESSING
    region_id: str
    start_date: str
    end_date: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    completed_at: Optional[str] = None
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = "Job queued for execution"
    error: Optional[str] = None
    cache_key: Optional[str] = None
