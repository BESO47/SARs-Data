"""Phase 2 tests: Google Earth Engine authentication service and Sentinel-1 data discovery."""

import pytest

from app.core.regions import validate_and_resolve_region
from app.models.metadata import DataStatus
from app.processing.sentinel1.acquisition import (
    S1_COLLECTION_ID,
    discover_sentinel1_observations,
)
from app.services.earth_engine import get_earth_engine_service


def test_gee_service_initialization_state():
    gee = get_earth_engine_service()
    # Without GEE_PROJECT_ID configured, is_live is False and last_error explains why
    assert isinstance(gee.is_live, bool)
    if not gee.is_live:
        assert gee.last_error is not None


def test_sentinel1_discovery_madeira():
    region = validate_and_resolve_region(region_id="madeira")
    result = discover_sentinel1_observations(
        region=region,
        start_date="2024-03-01",
        end_date="2024-04-15",
        polarizations=["VV", "VH"],
    )
    assert result.collection_id == S1_COLLECTION_ID
    assert result.observation_count >= 3
    assert result.consistent_orbit_geometry is True
    assert result.dominant_orbit_pass in ("ASCENDING", "DESCENDING")
    assert result.status in (DataStatus.AVAILABLE, DataStatus.DEMO)
    assert result.metadata.dataset == "sentinel-1"


def test_sentinel1_insufficient_observations():
    region = validate_and_resolve_region(region_id="mamiraua")
    # 1-day window with no pass
    result = discover_sentinel1_observations(
        region=region,
        start_date="2024-03-01",
        end_date="2024-03-02",
        polarizations=["VV", "VH"],
    )
    assert result.observation_count == 0
    assert result.status == DataStatus.INSUFFICIENT_OBSERVATIONS
    assert result.quality.level.value == "insufficient"


def test_sentinel1_rejects_invalid_polarization():
    region = validate_and_resolve_region(region_id="madeira")
    with pytest.raises(ValueError, match="Unsupported Sentinel-1 polarization"):
        discover_sentinel1_observations(
            region=region,
            start_date="2024-03-01",
            end_date="2024-03-30",
            polarizations=["INVALID_POL"],
        )
