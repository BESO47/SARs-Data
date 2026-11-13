"""FastAPI routes for NASA–ISRO NISAR L-band SAR analysis (Section 12)."""

from typing import Optional

from fastapi import APIRouter, Query

from app.models.layers import CesiumLayerResponse
from app.models.statistics import StatisticsResponse, TimelineResponse
from app.processing.nisar.change_analysis import run_nisar_analysis_pipeline
from app.processing.nisar.discovery import NisarDiscoveryResult, discover_nisar_products
from app.schemas.common import parse_spatial_query

router = APIRouter(prefix="/nisar", tags=["NASA-ISRO NISAR L-Band"])


@router.get("/coverage", response_model=NisarDiscoveryResult)
async def get_nisar_coverage(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    product_type: str = Query(default="GCOV", description="NISAR product type: GCOV, GUNW, or RSLC"),
) -> NisarDiscoveryResult:
    """Discover NISAR L-band acquisitions via ASF DAAC and check repeat-pass availability."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    return discover_nisar_products(
        region=region,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type,
    )


@router.get("/observations", response_model=TimelineResponse)
async def get_nisar_observations(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    product_type: str = Query(default="GCOV"),
) -> TimelineResponse:
    """Return NISAR L-band observation timeline with product maturity, track/frame, and quality flags."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_nisar_analysis_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type,
    )
    return res["observations"]


@router.get("/layer", response_model=CesiumLayerResponse)
async def get_nisar_layer(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    product_type: str = Query(default="GCOV"),
) -> CesiumLayerResponse:
    """Return CesiumJS-compatible NISAR L-band polarimetric indicator layer."""
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_nisar_analysis_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type,
    )
    return res["layer"]


@router.get("/statistics", response_model=StatisticsResponse)
async def get_nisar_statistics(
    region_id: Optional[str] = Query(default="madeira"),
    bbox: Optional[str] = Query(default=None),
    polygon_json: Optional[str] = Query(default=None),
    start_date: str = Query(default="2024-03-01"),
    end_date: str = Query(default="2024-04-15"),
    product_type: str = Query(default="GCOV"),
) -> StatisticsResponse:
    """
    Return NISAR L-band statistics explicitly separating direct measurements,
    derived indicators, interpretations (with Sentinel-1 C-band comparison), and missing observations.
    """
    region = parse_spatial_query(region_id=region_id, bbox_str=bbox, polygon_json=polygon_json)
    res = run_nisar_analysis_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type,
    )
    return res["statistics"]
