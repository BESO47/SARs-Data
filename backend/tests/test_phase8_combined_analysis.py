"""Phase 8 tests: Combined spatial overlap, proximity buffer analysis, rainfall context, and unified timeline."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_combined_overlap_endpoint():
    resp = client.get("/api/analysis/overlap?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["overlap_area_km2"] > 0
    assert 0.0 <= data["pct_of_flood_area"] <= 100.0
    assert 0.0 <= data["pct_of_disturbance_area"] <= 100.0
    assert data["geojson"]["type"] == "FeatureCollection"
    assert len(data["geojson"]["features"]) == data["intersection_count"]


def test_combined_proximity_endpoint():
    resp = client.get("/api/analysis/proximity?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15&distance_km=5.0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["distance_threshold_km"] == 5.0
    assert data["direct_intersection_km2"] > 0
    assert data["proximity_buffer_only_km2"] >= 0
    assert abs(
        (data["direct_intersection_km2"] + data["proximity_buffer_only_km2"])
        - data["total_within_distance_km2"]
    ) < 0.001


def test_combined_summary_and_unified_timeline():
    sum_resp = client.get("/api/analysis/summary?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert sum_resp.status_code == 200
    sdata = sum_resp.json()
    assert set(sdata["dataset_availability"].keys()) == {"sentinel-1", "gfw-alerts", "nisar", "gpm-imerg"}
    assert "before_flood_period" in sdata["rainfall_context"]
    assert "during_flood_period" in sdata["rainfall_context"]
    assert "after_flood_period" in sdata["rainfall_context"]

    tl_resp = client.get("/api/analysis/timeline?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert tl_resp.status_code == 200
    tldata = tl_resp.json()
    datasets_in_tl = {e["dataset"] for e in tldata["series"]}
    assert datasets_in_tl == {"sentinel-1", "gfw-alerts", "nisar", "gpm-imerg"}
    # Verify chronological ordering
    timestamps = [e["timestamp"] for e in tldata["series"]]
    assert timestamps == sorted(timestamps)
