"""Common API request/response schemas and query parameter parsing."""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.regions import validate_and_resolve_region
from app.models.metadata import DataStatus


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str
    version: str
    timestamp: str
    environment: str


class DatasetStatusEntry(BaseModel):
    """Availability and credential status of each external satellite dataset."""

    dataset: str
    provider: str
    collection_or_endpoint: str
    status: DataStatus
    credentials_configured: bool
    live_connection_verified: bool
    notes: str


class DatasetsStatusResponse(BaseModel):
    """Overall status of all four environmental monitoring datasets."""

    timestamp: str
    datasets: List[DatasetStatusEntry]


class StudyAreasResponse(BaseModel):
    """Available predefined study areas in the Amazon Basin."""

    default_region_id: str = "madeira"
    areas: List[Dict[str, Any]]


def parse_spatial_query(
    region_id: Optional[str] = "madeira",
    bbox_str: Optional[str] = None,
    polygon_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse query parameters for region_id, comma-separated bbox, or GeoJSON string."""
    bbox_list: Optional[List[float]] = None
    if bbox_str:
        try:
            bbox_list = [float(x.strip()) for x in bbox_str.split(",")]
        except ValueError as exc:
            raise ValueError(f"Invalid bbox parameter '{bbox_str}'. Expected 'min_lon,min_lat,max_lon,max_lat'.") from exc

    poly_dict: Optional[Dict[str, Any]] = None
    if polygon_json:
        try:
            poly_dict = json.loads(polygon_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid polygon_json parameter: {exc}") from exc

    return validate_and_resolve_region(
        region_id=region_id,
        bbox=bbox_list,
        polygon_geojson=poly_dict,
    )
