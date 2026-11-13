"""Phase 5 tests: Global Forest Watch (RADD & GLAD) acquisition, normalization, and API routes."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_forest_alerts_endpoint_and_pagination():
    resp = client.get("/api/forest/alerts?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15&page=1&page_size=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "FeatureCollection"
    assert data["total_features"] >= 6
    assert len(data["features"]) == 5
    # Verify separate source identities and confidence levels are preserved
    sources = {f["properties"]["source"] for f in data["features"]}
    assert sources.issubset({"wur_radd_alerts", "umd_glad_sentinel2_alerts", "umd_glad_landsat_alerts"})
    for f in data["features"]:
        assert f["properties"]["area_km2"] > 0
        assert f["properties"]["confidence"] in ("high", "nominal")


def test_forest_filter_by_alert_type_and_confidence():
    resp = client.get(
        "/api/forest/alerts?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15"
        "&alert_type=wur_radd_alerts&confidence=high"
    )
    assert resp.status_code == 200
    data = resp.json()
    for f in data["features"]:
        assert f["properties"]["source"] == "wur_radd_alerts"
        assert f["properties"]["confidence"] == "high"


def test_forest_layer_statistics_and_timeline():
    layer_resp = client.get("/api/forest/layer?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert layer_resp.status_code == 200
    assert layer_resp.json()["dataset"] == "gfw-alerts"

    stats_resp = client.get("/api/forest/statistics?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "counts_by_dataset" in stats_data["breakdown"]
    assert "wur_radd_alerts" in stats_data["breakdown"]["counts_by_dataset"]

    tl_resp = client.get("/api/forest/timeline?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert tl_resp.status_code == 200
    assert len(tl_resp.json()["series"]) > 0
