"""FastAPI routes for NASA GPM IMERG V07 rainfall monitoring (Section 12)."""

from typing import Optional

from fastapi import APIRouter, Query

from app.models.layers import CesiumLayerResponse
from app.models.statistics import StatisticsResponse, TimelineResponse
from app.processing.rainfall.aggregation import run_gpm_rainfall_pipeline
from app.schemas.common import parse_spatial_query

router = APIRouter(prefix="/rainfall", tags=["NASA GPM IMERG Rainfall"])


@router.get("/layer", response_model=CesiumLayerResponse)
async def get_rainfall_layer(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
) -> CesiumLayerResponse:
    """Return CesiumJS-compatible accumulated rainfall raster tile layer and metadata."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_gpm_rainfall_pipeline(region=region, start_date=start_date, end_date=end_date)
    return res["layer"]


@router.get("/statistics", response_model=StatisticsResponse)
async def get_rainfall_statistics(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
) -> StatisticsResponse:
    """Return accumulated rainfall statistics, daily/monthly summaries, and spatial distribution."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_gpm_rainfall_pipeline(region=region, start_date=start_date, end_date=end_date)
    return res["statistics"]


@router.get("/timeline", response_model=TimelineResponse)
async def get_rainfall_timeline(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
) -> TimelineResponse:
    """Return daily accumulated rainfall time series (integrated from 0.5-hour rate granules)."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_gpm_rainfall_pipeline(region=region, start_date=start_date, end_date=end_date)
    return res["timeline"]
