"""Phase 1 tests: common models, schemas, metadata, region validation, quality control, and health endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.quality_control import (
    assess_observation_quality,
    validate_and_deduplicate_features,
    validate_date_range,
    validate_raster_array,
)
from app.core.regions import (
    calculate_equal_area_km2,
    calculate_geodesic_area_km2,
    validate_and_resolve_region,
)
from app.main import app
from app.models.layers import GeoJSONFeature, GeoJSONFeatureProperties
from app.models.metadata import DataStatus, DatasetMetadata, QualityLevel

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "Amazon Sentinel" in data["service"]
    assert "timestamp" in data


def test_areas_endpoint():
    resp = client.get("/api/areas")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_region_id"] == "madeira"
    region_ids = {a["region_id"] for a in data["areas"]}
    assert {"madeira", "mamiraua", "xingu", "purus", "amazon_basin"}.issubset(region_ids)
    for area in data["areas"]:
        assert area["area_km2"] > 0
        assert area["crs"] == "EPSG:4326"


def test_datasets_status_endpoint():
    resp = client.get("/api/datasets/status")
    assert resp.status_code == 200
    data = resp.json()
    names = [d["dataset"] for d in data["datasets"]]
    assert names == ["sentinel-1", "gfw-alerts", "nisar", "gpm-imerg"]


def test_region_validation_and_area_calculations():
    madeira = validate_and_resolve_region(region_id="madeira")
    assert madeira["region_id"] == "madeira"
    geod_area = calculate_geodesic_area_km2(madeira["geometry"])
    eq_area = calculate_equal_area_km2(madeira["geometry"])
    # Geodesic and Albers Equal Area should agree within <0.5%
    assert abs(geod_area - eq_area) / geod_area < 0.005

    # Reject geometry outside Amazon Basin (e.g., Europe)
    with pytest.raises(ValueError, match="outside the Amazon Basin"):
        validate_and_resolve_region(bbox=[10.0, 45.0, 12.0, 47.0])


def test_quality_control_functions():
    start, end, days = validate_date_range("2024-03-01", "2024-03-15")
    assert days == 14

    with pytest.raises(ValueError, match="must not be after"):
        validate_date_range("2024-04-01", "2024-03-01")

    status_insuff, q_insuff = assess_observation_quality(observation_count=0, min_required_observations=1)
    assert status_insuff == DataStatus.INSUFFICIENT_OBSERVATIONS
    assert q_insuff.level == QualityLevel.INSUFFICIENT

    # Deduplication and invalid geometry repair
    f1 = GeoJSONFeature(
        geometry={"type": "Polygon", "coordinates": [[[-63.0, -7.5], [-62.9, -7.5], [-62.9, -7.4], [-63.0, -7.4], [-63.0, -7.5]]]},
        properties=GeoJSONFeatureProperties(
            feature_id="feat-1",
            dataset="sentinel-1",
            classification="open_water_inundation",
            source="COPERNICUS/S1_GRD",
        ),
    )
    f_dup = GeoJSONFeature(
        geometry=f1.geometry,
        properties=f1.properties,
    )
    cleaned, warnings = validate_and_deduplicate_features([f1, f_dup])
    assert len(cleaned) == 1
    assert any("duplicate" in w.lower() for w in warnings)

    arr = np.array([[-15.0, -18.0], [-12.0, -22.0]], dtype=np.float32)
    valid, r_warn = validate_raster_array(arr, valid_min=-35.0, valid_max=5.0, parameter_name="VV_dB")
    assert valid is True
    assert len(r_warn) == 0


def test_metadata_schema():
    meta = DatasetMetadata(
        dataset="sentinel-1",
        provider="ESA",
        instrument="Sentinel-1 SAR",
        product_id="COPERNICUS/S1_GRD",
        status=DataStatus.AVAILABLE,
        region_id="madeira",
        method="temporal_backscatter_comparison",
    )
    dumped = meta.model_dump()
    assert dumped["dataset"] == "sentinel-1"
    assert dumped["crs"] == "EPSG:4326"
