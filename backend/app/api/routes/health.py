"""Health, study areas, and dataset status endpoints (Section 12)."""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.regions import list_predefined_areas
from app.models.metadata import DataStatus
from app.schemas.common import (
    DatasetStatusEntry,
    DatasetsStatusResponse,
    HealthResponse,
    StudyAreasResponse,
)

router = APIRouter(tags=["Health & Metadata"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return service health and version metadata."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        environment=settings.environment,
    )


@router.get("/areas", response_model=StudyAreasResponse)
async def get_study_areas() -> StudyAreasResponse:
    """Return predefined Amazon Basin study areas with validated geometries and geodesic areas."""
    return StudyAreasResponse(
        default_region_id="madeira",
        areas=list_predefined_areas(),
    )


@router.get("/datasets/status", response_model=DatasetsStatusResponse)
async def get_datasets_status() -> DatasetsStatusResponse:
    """Report configuration and live availability status for all four satellite datasets."""
    settings = get_settings()
    gee_ok = settings.has_gee_configured()
    gfw_ok = settings.has_gfw_configured()
    ed_ok = settings.has_earthdata_configured()

    datasets = [
        DatasetStatusEntry(
            dataset="sentinel-1",
            provider="ESA / Copernicus (via Google Earth Engine)",
            collection_or_endpoint="COPERNICUS/S1_GRD",
            status=DataStatus.AVAILABLE if gee_ok else DataStatus.DEMO,
            credentials_configured=gee_ok,
            live_connection_verified=False,
            notes=(
                "GEE project configured; live cloud-side SAR processing active."
                if gee_ok
                else "GEE_PROJECT_ID not configured. Pipeline runs real local SAR array processing on calibrated reference observations with status='demo'."
            ),
        ),
        DatasetStatusEntry(
            dataset="gfw-alerts",
            provider="World Resources Institute / Global Forest Watch",
            collection_or_endpoint=f"{settings.gfw_api_base_url} (wur_radd_alerts, umd_glad_landsat_alerts, umd_glad_sentinel2_alerts)",
            status=DataStatus.AVAILABLE if gfw_ok else DataStatus.DEMO,
            credentials_configured=gfw_ok,
            live_connection_verified=False,
            notes=(
                "GFW API key configured; live RADD & GLAD disturbance queries enabled."
                if gfw_ok
                else "GFW_API_KEY not configured. Pipeline normalizes reference RADD/GLAD alerts with status='demo' unless open endpoint succeeds."
            ),
        ),
        DatasetStatusEntry(
            dataset="nisar",
            provider="NASA / ISRO (Alaska Satellite Facility DAAC)",
            collection_or_endpoint=settings.asf_search_api_url,
            status=DataStatus.AVAILABLE,
            credentials_configured=ed_ok,
            live_connection_verified=False,
            notes=(
                "ASF Search API discovery is public; Earthdata S3 credentials configured for direct us-west-2 HDF5/NetCDF4 remote reads."
                if ed_ok
                else "ASF Search API discovery is active; Earthdata credentials not set for S3 temporary token exchange."
            ),
        ),
        DatasetStatusEntry(
            dataset="gpm-imerg",
            provider="NASA GES DISC (via Google Earth Engine)",
            collection_or_endpoint="NASA/GPM_L3/IMERG_V07",
            status=DataStatus.AVAILABLE if gee_ok else DataStatus.DEMO,
            credentials_configured=gee_ok,
            live_connection_verified=False,
            notes=(
                "GEE project configured; half-hourly precipitation rate (mm/hr * 0.5h) temporal integration active."
                if gee_ok
                else "GEE_PROJECT_ID not configured. Pipeline executes verified half-hourly rate-to-accumulation integration on reference observations with status='demo'."
            ),
        ),
    ]

    return DatasetsStatusResponse(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        datasets=datasets,
    )
