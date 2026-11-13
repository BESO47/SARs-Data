"""Phase 4 tests: Sentinel-1 flood map outputs, GeoJSON vectorization, statistics, and FastAPI endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_flood_coverage_endpoint():
    resp = client.get("/api/flood/coverage?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["region_id"] == "madeira"
    assert data["collection_id"] == "COPERNICUS/S1_GRD"
    assert data["observation_count"] >= 3
    assert len(data["scenes"]) == data["observation_count"]


def test_flood_layer_endpoint():
    resp = client.get("/api/flood/layer?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset"] == "sentinel-1"
    assert "{z}/{x}/{y}" in data["tile_url_template"]
    assert data["geojson"]["type"] == "FeatureCollection"
    assert len(data["geojson"]["features"]) > 0
    assert len(data["legend"]["items"]) == 5


def test_flood_statistics_endpoint():
    resp = client.get("/api/flood/statistics?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset"] == "sentinel-1"
    metric_ids = [s["metric_id"] for s in data["statistics"]]
    assert "total_inundated_area_km2" in metric_ids
    assert "open_water_inundation_km2" in metric_ids
    assert "possible_flooded_vegetation_km2" in metric_ids
    assert "permanent_water_km2" in metric_ids
    assert "uncertain_area_km2" in metric_ids
    assert "comparison_with_previous_period" in data["breakdown"]


def test_flood_timeline_endpoint():
    resp = client.get("/api/flood/timeline?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset"] == "sentinel-1"
    assert data["unit"] == "km²"
    assert len(data["series"]) >= 3
    for item in data["series"]:
        assert item["product_id"].startswith("S1A_IW_GRDH")
