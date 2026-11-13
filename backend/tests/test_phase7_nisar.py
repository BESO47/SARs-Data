"""Phase 7 tests: NASA-ISRO NISAR L-band discovery, product-specific preprocessing, and API routes."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_nisar_coverage_and_repeat_time_series():
    # Multi-week window -> >= 2 acquisitions -> has_repeat_time_series is True
    resp = client.get("/api/nisar/coverage?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15&product_type=GCOV")
    assert resp.status_code == 200
    data = resp.json()
    assert data["requested_product_type"] == "GCOV"
    assert data["observation_count"] >= 3
    assert data["has_repeat_time_series"] is True
    for p in data["products"]:
        assert p["hdf5_group_path"] == "/science/LSAR/GCOV/grids/frequencyA"
        assert "s3://" in p["s3_uri"]

    # 8-day window -> only 1 acquisition -> has_repeat_time_series is False
    short_resp = client.get("/api/nisar/coverage?region_id=madeira&start_date=2024-03-05&end_date=2024-03-12&product_type=GCOV")
    assert short_resp.status_code == 200
    short_data = short_resp.json()
    assert short_data["observation_count"] == 1
    assert short_data["has_repeat_time_series"] is False


def test_nisar_statistics_scientific_distinctions():
    resp = client.get("/api/nisar/statistics?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15&product_type=GCOV")
    assert resp.status_code == 200
    data = resp.json()
    breakdown = data["breakdown"]
    assert "direct_measurements" in breakdown
    assert "derived_indicators" in breakdown
    assert "interpretations" in breakdown
    assert "sentinel1_cband_comparison" in breakdown["interpretations"]
    # Ensure no AWS credentials ever appear in response JSON
    raw_str = resp.text.lower()
    assert "secretaccesskey" not in raw_str
    assert "sessiontoken" not in raw_str


def test_nisar_rejects_rslc_for_geocoded_layer():
    resp = client.get("/api/nisar/layer?region_id=madeira&start_date=2024-03-01&end_date=2024-04-15&product_type=RSLC")
    assert resp.status_code == 400
    assert "slant-range" in resp.json()["detail"]
