"""FastAPI routes for Global Forest Watch disturbance monitoring (Section 12)."""

from typing import Optional

from fastapi import APIRouter, Query

from app.models.layers import CesiumLayerResponse, GeoJSONFeatureCollection
from app.models.statistics import StatisticsResponse, TimelineResponse
from app.processing.forest.statistics import run_forest_disturbance_pipeline
from app.schemas.common import parse_spatial_query

router = APIRouter(prefix="/forest", tags=["Global Forest Watch Disturbance"])


@router.get("/alerts", response_model=GeoJSONFeatureCollection)
async def get_forest_alerts(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    alert_type: Optional[str] = Query(
        default=None,
        description="Optional filter: wur_radd_alerts, umd_glad_sentinel2_alerts, or umd_glad_landsat_alerts",
    ),
    confidence: Optional[str] = Query(default=None, description="Optional filter: high or nominal"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=200, ge=1, le=1000),
) -> GeoJSONFeatureCollection:
    """Retrieve normalized, paginated RADD and GLAD forest disturbance alerts as GeoJSON."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    alert_types = [alert_type] if alert_type else None
    res = run_forest_disturbance_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        alert_types=alert_types,
        confidence_filter=confidence,
        page=page,
        page_size=page_size,
    )
    return res["alerts"]


@router.get("/layer", response_model=CesiumLayerResponse)
async def get_forest_layer(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    alert_type: Optional[str] = Query(default=None),
    confidence: Optional[str] = Query(default=None),
) -> CesiumLayerResponse:
    """Return CesiumJS-compatible forest disturbance alert layer."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    alert_types = [alert_type] if alert_type else None
    res = run_forest_disturbance_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        alert_types=alert_types,
        confidence_filter=confidence,
    )
    return res["layer"]


@router.get("/statistics", response_model=StatisticsResponse)
async def get_forest_statistics(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    alert_type: Optional[str] = Query(default=None),
    confidence: Optional[str] = Query(default=None),
) -> StatisticsResponse:
    """Return forest disturbance alert counts, area in km², and breakdown by source dataset and confidence."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    alert_types = [alert_type] if alert_type else None
    res = run_forest_disturbance_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        alert_types=alert_types,
        confidence_filter=confidence,
    )
    return res["statistics"]


@router.get("/timeline", response_model=TimelineResponse)
async def get_forest_timeline(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    alert_type: Optional[str] = Query(default=None),
    confidence: Optional[str] = Query(default=None),
) -> TimelineResponse:
    """Return daily forest disturbance alert timeline."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    alert_types = [alert_type] if alert_type else None
    res = run_forest_disturbance_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        alert_types=alert_types,
        confidence_filter=confidence,
    )
    return res["timeline"]
