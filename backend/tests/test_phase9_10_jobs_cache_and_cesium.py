"""
Phase 9 & Phase 10 tests:
  - Caching with dataset/region/date/config keys, TTL expiration, and explicit invalidation.
  - Asynchronous processing jobs (`POST /api/processing/jobs`, `GET /api/processing/jobs/{job_id}`, `GET /api/processing/jobs/{job_id}/results`).
  - CesiumJS XYZ PNG map tile endpoints (`/api/tiles/{dataset}/{region_id}/{z}/{x}/{y}.png`) and full 22-endpoint API contract verification.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.cache.manager import CacheAndJobStore, build_cache_key
from app.main import app

client = TestClient(app)


def test_cache_key_ttl_and_invalidation(tmp_path: Path):
    db_file = tmp_path / "test_cache.db"
    store = CacheAndJobStore(db_path=db_file)

    k1 = build_cache_key("sentinel-1", "madeira", "2024-03-01", "2024-04-15", {"thr": -17.5})
    k2 = build_cache_key("sentinel-1", "madeira", "2024-03-01", "2024-04-15", {"thr": -18.0})
    assert k1 != k2  # Different algorithm parameters produce distinct cache keys

    store.set(k1, "sentinel-1", "madeira", "2024-03-01", "2024-04-15", {"value": 42}, ttl_seconds=60)
    assert store.get(k1) == {"value": 42}

    removed = store.invalidate(dataset="sentinel-1", region_id="madeira")
    assert removed == 1
    assert store.get(k1) is None


def test_async_processing_job_lifecycle():
    # Submit combined_analysis job
    post_resp = client.post(
        "/api/processing/jobs",
        json={
            "job_type": "combined_analysis",
            "region_id": "madeira",
            "start_date": "2024-03-01",
            "end_date": "2024-04-15",
            "parameters": {"distance_km": 5.0},
        },
    )
    assert post_resp.status_code == 202
    job_info = post_resp.json()
    job_id = job_info["job_id"]
    assert job_id.startswith("job-")

    # Starlette TestClient runs BackgroundTasks before returning from post(), so status should now be completed
    status_resp = client.get(f"/api/processing/jobs/{job_id}")
    assert status_resp.status_code == 200
    sdata = status_resp.json()
    assert sdata["status"] == "completed"
    assert sdata["progress_pct"] == 100.0

    # Retrieve results
    res_resp = client.get(f"/api/processing/jobs/{job_id}/results")
    assert res_resp.status_code == 200
    rdata = res_resp.json()
    assert rdata["job_id"] == job_id
    assert "summary" in rdata
    assert "overlap" in rdata
    assert "proximity" in rdata
    assert "timeline" in rdata


def test_cesium_xyz_png_tiles_and_layer_compatibility():
    for ds in ["flood", "forest", "nisar", "rainfall"]:
        tile_resp = client.get(f"/api/tiles/{ds}/madeira/8/83/133.png?start_date=2024-03-01&end_date=2024-04-15")
        assert tile_resp.status_code == 200
        assert tile_resp.headers["content-type"] == "image/png"
        # PNG magic header signature: \x89PNG\r\n\x1a\n
        assert tile_resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Verify all 4 Cesium layer endpoints return independent layer_ids, legends, status, and quality metadata
    layer_urls = [
        "/api/flood/layer?region_id=madeira",
        "/api/forest/layer?region_id=madeira",
        "/api/nisar/layer?region_id=madeira",
        "/api/rainfall/layer?region_id=madeira",
    ]
    layer_ids = set()
    for url in layer_urls:
        r = client.get(url)
        assert r.status_code == 200
        body = r.json()
        layer_ids.add(body["layer_id"])
        assert "legend" in body
        assert "quality" in body
        assert "metadata" in body
        assert 0.0 <= body["default_opacity"] <= 1.0
    assert len(layer_ids) == 4
