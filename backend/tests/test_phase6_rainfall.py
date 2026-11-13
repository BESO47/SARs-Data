"""Phase 6 tests: NASA GPM IMERG V07 half-hourly rate integration, monthly summaries, and API endpoints."""

from fastapi.testclient import TestClient

from app.main import app
from app.processing.rainfall.aggregation import integrate_half_hourly_rates_to_mm

client = TestClient(app)


def test_imerg_rate_to_depth_integration_math():
    # Two half-hourly granules at 10 mm/hr each over 1 hour total = 10.0 mm depth (NOT 20.0)
    rates = [10.0, 10.0]
    depth = integrate_half_hourly_rates_to_mm(rates, interval_hours=0.5)
    assert depth == 10.0


def test_rainfall_statistics_and_monthly_summaries():
    resp = client.get("/api/rainfall/statistics?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset"] == "gpm-imerg"
    assert data["metadata"]["product_id"] == "NASA/GPM_L3/IMERG_V07"
    assert data["metadata"]["extra"]["temporal_interval_hours"] == 0.5
    months = [m["month"] for m in data["breakdown"]["monthly_summaries"]]
    assert months == ["2024-03", "2024-04"]


def test_rainfall_layer_and_timeline():
    layer_resp = client.get("/api/rainfall/layer?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15")
    assert layer_resp.status_code == 200
    assert layer_resp.json()["dataset"] == "gpm-imerg"

    tl_resp = client.get("/api/rainfall/timeline?region_id=madeira&start_date=2024-03-01&end_date=2024-03-10")
    assert tl_resp.status_code == 200
    tl = tl_resp.json()
    assert tl["unit"] == "mm"
    assert len(tl["series"]) == 10
    for entry in tl["series"]:
        assert entry["details"]["interval_hours"] == 0.5
        assert entry["details"]["granule_count"] == 48
