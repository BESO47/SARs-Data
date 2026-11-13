"""FastAPI routes for combined spatial and temporal analysis (Section 12)."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from app.models.statistics import TimelineResponse
from app.processing.combined.timeline import run_combined_multimodal_analysis
from app.schemas.common import parse_spatial_query

router = APIRouter(prefix="/analysis", tags=["Combined Spatial & Temporal Analysis"])


@router.get("/summary")
async def get_analysis_summary(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    distance_km: float = Query(default=5.0, gt=0.0, le=100.0),
) -> Dict[str, Any]:
    """Return unified regional summary, dataset availability, overlap/proximity metrics, and rainfall context."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_combined_multimodal_analysis(
        region=region,
        start_date=start_date,
        end_date=end_date,
        distance_km=distance_km,
    )
    return res["summary"]


@router.get("/overlap")
async def get_analysis_overlap(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
) -> Dict[str, Any]:
    """
    Calculate `Area(FloodGeometry ∩ ForestDisturbanceGeometry)` in km², overlap percentages,
    and return intersection GeoJSON polygons.
    """
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_combined_multimodal_analysis(
        region=region,
        start_date=start_date,
        end_date=end_date,
    )
    return res["overlap"]


@router.get("/proximity")
async def get_analysis_proximity(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    distance_km: float = Query(default=5.0, gt=0.0, le=100.0, description="Proximity distance threshold in km (default 5 km)"),
) -> Dict[str, Any]:
    """
    Calculate forest disturbance alert area within `distance_km` of flood inundation,
    explicitly distinguishing direct intersection from buffer proximity.
    """
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_combined_multimodal_analysis(
        region=region,
        start_date=start_date,
        end_date=end_date,
        distance_km=distance_km,
    )
    return res["proximity"]


@router.get("/timeline", response_model=TimelineResponse)
async def get_analysis_timeline(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
) -> TimelineResponse:
    """Return unified chronological timeline across Sentinel-1, GFW Alerts, NISAR, and GPM IMERG."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_combined_multimodal_analysis(
        region=region,
        start_date=start_date,
        end_date=end_date,
    )
    return res["timeline"]
