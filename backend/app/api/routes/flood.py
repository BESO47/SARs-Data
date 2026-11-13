"""FastAPI routes for Sentinel-1 SAR flood monitoring (Section 12)."""

from typing import Optional

from fastapi import APIRouter, Query

from app.models.layers import CesiumLayerResponse
from app.models.statistics import StatisticsResponse, TimelineResponse
from app.processing.sentinel1.acquisition import (
    S1CoverageResult,
    discover_sentinel1_observations,
)
from app.processing.sentinel1.outputs import run_sentinel1_flood_pipeline
from app.schemas.common import parse_spatial_query

router = APIRouter(prefix="/flood", tags=["Sentinel-1 Flood Monitoring"])


@router.get("/coverage", response_model=S1CoverageResult)
async def get_flood_coverage(
    region_id: Optional[str] = Query(default="madeira", description="Study region ID"),
    bbox: Optional[str] = Query(default=None, description="Optional min_lon,min_lat,max_lon,max_lat"),
    polygon_json: Optional[str] = Query(default=None, description="Optional GeoJSON Polygon string"),
    start_date: str = Query(default="2024-03-01", description="Observation start date (YYYY-MM-DD)"),
    end_date: str = Query(default="2024-04-15", description="Observation end date (YYYY-MM-DD)"),
    orbit_pass: Optional[str] = Query(default=None, description="Optional ASCENDING or DESCENDING filter"),
) -> S1CoverageResult:
    """Retrieve Sentinel-1 IW GRD acquisition metadata, temporal gaps, and orbit consistency."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    return discover_sentinel1_observations(
        region=region,
        start_date=start_date,
        end_date=end_date,
        orbit_pass=orbit_pass,
    )


@router.get("/layer", response_model=CesiumLayerResponse)
async def get_flood_layer(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    baseline_start_date: str = Query(default="2023-08-01"),
    baseline_end_date: str = Query(default="2023-10-15"),
) -> CesiumLayerResponse:
    """Return CesiumJS-compatible flood layer with XYZ tile template, GeoJSON features, legend, and metadata."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    result = run_sentinel1_flood_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        baseline_start_date=baseline_start_date,
        baseline_end_date=baseline_end_date,
    )
    return result["layer"]


@router.get("/statistics", response_model=StatisticsResponse)
async def get_flood_statistics(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    baseline_start_date: str = Query(default="2023-08-01"),
    baseline_end_date: str = Query(default="2023-10-15"),
) -> StatisticsResponse:
    """Return estimated inundated area in km², area by classification, and comparison with previous period."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    result = run_sentinel1_flood_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        baseline_start_date=baseline_start_date,
        baseline_end_date=baseline_end_date,
    )
    return result["statistics"]


@router.get("/timeline", response_model=TimelineResponse)
async def get_flood_timeline(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
) -> TimelineResponse:
    """Return pass-by-pass Sentinel-1 flood observation timeline."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    result = run_sentinel1_flood_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
    )
    return result["timeline"]
