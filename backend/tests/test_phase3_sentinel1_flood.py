"""Phase 3 tests: Sentinel-1 SAR preprocessing (Lee speckle filter, linear/dB) and 5-class flood detection."""

import numpy as np

from app.core.regions import validate_and_resolve_region
from app.processing.sentinel1.flood_detection import (
    FLOOD_CLASSES,
    FloodDetectionConfig,
    classify_sentinel1_flood,
)
from app.processing.sentinel1.preprocessing import (
    db_to_linear_power,
    lee_speckle_filter_db,
    linear_power_to_db,
    prepare_sentinel1_observation_grids,
)


def test_db_linear_roundtrip_and_lee_filter():
    arr_db = np.array([[-10.0, -15.0, -20.0], [-8.0, -12.0, -18.0], [-9.0, -14.0, -19.0]], dtype=np.float32)
    lin = db_to_linear_power(arr_db)
    recovered_db = linear_power_to_db(lin)
    assert np.allclose(arr_db, recovered_db, atol=1e-4)

    # Lee filter should reduce variance of noisy speckle array while preserving mean power
    rng = np.random.default_rng(42)
    uniform_lin = np.full((32, 32), 0.1, dtype=np.float32)  # -10 dB
    noisy_lin = uniform_lin * rng.gamma(4.4, 1.0 / 4.4, size=(32, 32)).astype(np.float32)
    noisy_db = linear_power_to_db(noisy_lin)
    filtered_db = lee_speckle_filter_db(noisy_db, window_size=5, enl=4.4)
    assert np.std(filtered_db) < np.std(noisy_db)
    assert abs(float(np.mean(db_to_linear_power(filtered_db))) - 0.1) < 0.015


def test_sentinel1_5class_flood_detection():
    region = validate_and_resolve_region(region_id="madeira")
    prep = prepare_sentinel1_observation_grids(
        region=region,
        start_date="2024-03-01",
        end_date="2024-04-15",
        grid_shape=(64, 64),
    )
    cfg = FloodDetectionConfig()
    res = classify_sentinel1_flood(prep, total_region_area_km2=region["area_km2"], config=cfg)

    # Verify all 5 classes are represented in the output dictionary and sum to total pixels
    assert set(res.pixel_counts.keys()) == set(FLOOD_CLASSES.values())
    assert sum(res.pixel_counts.values()) == 64 * 64
    assert res.area_by_class_km2["permanent_water"] > 0
    assert res.area_by_class_km2["open_water_inundation"] > 0
    assert res.area_by_class_km2["possible_flooded_vegetation"] > 0
    assert res.area_by_class_km2["uncertain"] >= 0
    assert res.area_by_class_km2["not_detected"] > 0
    assert abs(sum(res.area_by_class_km2.values()) - region["area_km2"]) < 1.0
    assert res.total_inundated_km2 == round(
        res.area_by_class_km2["open_water_inundation"] + res.area_by_class_km2["possible_flooded_vegetation"],
        3,
    )
