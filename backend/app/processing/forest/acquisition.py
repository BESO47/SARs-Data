"""Forest disturbance alert acquisition across RADD and GLAD datasets (Section 6.2)."""

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from shapely.geometry import box, mapping

from app.core.quality_control import validate_date_range
from app.services.global_forest_watch import GFW_DATASETS, get_gfw_client


def _generate_reference_forest_alerts(
    region: Dict[str, Any],
    start_dt: date,
    end_dt: date,
    datasets: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate geographically coherent reference RADD and GLAD disturbance records along the
    floodplain/upland transition of the selected Amazon study area when GFW_API_KEY is not configured.
    Preserves separate dataset identities (`wur_radd_alerts`, `umd_glad_sentinel2_alerts`, `umd_glad_landsat_alerts`)
    and original confidence attributes (`high`, `nominal`).
    """
    duration_days = (end_dt - start_dt).days
    if duration_days < 2:
        return []

    min_lon, min_lat, max_lon, max_lat = region["bbox"]
    span_lon = max_lon - min_lon
    span_lat = max_lat - min_lat

    seed = sum(ord(c) for c in region["region_id"]) * 19 + start_dt.toordinal()
    rng = np.random.default_rng(seed=seed)

    records: List[Dict[str, Any]] = []
    # Number of alert patches scales with observation window up to 24 patches
    num_patches = min(max(duration_days // 3, 6), 24)

    for idx in range(num_patches):
        ds_id = datasets[idx % len(datasets)]
        ds_info = GFW_DATASETS[ds_id]

        # Position some patches along riparian floodplain margins (overlapping/proximate to flood zones)
        # and some on upland terra firme ridges
        nx = -0.75 + 1.5 * (idx / max(num_patches - 1, 1))
        river_y = 0.18 * np.sin(2.6 * nx + (sum(ord(c) for c in region["region_id"]) * 31 % 5) * 0.4) - 0.05 * nx

        if idx % 3 == 0:
            # Riparian / floodplain edge disturbance (overlaps flooded vegetation / open flood margin)
            ny = float(river_y + (0.16 if idx % 2 == 0 else -0.16))
        elif idx % 3 == 1:
            # Proximate disturbance within 1-4 km of floodplain
            ny = float(river_y + (0.28 if idx % 2 == 0 else -0.28))
        else:
            # Distant upland disturbance
            ny = float(np.clip(river_y + (0.58 if idx % 2 == 0 else -0.58), -0.88, 0.88))

        cx = min_lon + (nx + 1.0) * 0.5 * span_lon
        cy = min_lat + (ny + 1.0) * 0.5 * span_lat

        # Patch half-width (~0.008 to 0.018 degrees -> ~0.9 to 2.0 km)
        hw = float(0.007 + 0.009 * rng.random())
        hh = float(0.006 + 0.008 * rng.random())
        geom = mapping(box(round(cx - hw, 6), round(cy - hh, 6), round(cx + hw, 6), round(cy + hh, 6)))

        day_offset = int((idx * duration_days) // max(num_patches, 1))
        alert_dt = (start_dt + timedelta(days=day_offset)).isoformat()
        confidence = "high" if (idx % 4 != 0) else "nominal"

        records.append(
            {
                "source_record_id": f"{ds_id.upper()}_{region['region_id'].upper()}_{alert_dt.replace('-', '')}_{idx:03d}",
                "dataset_id": ds_id,
                "alert_family": ds_info["alert_family"],
                "provider": ds_info["provider"],
                "instrument": ds_info["instrument"],
                "resolution_m": ds_info["resolution_m"],
                "alert_date": alert_dt,
                "confidence": confidence,
                "geometry": geom,
            }
        )

    return records


def acquire_forest_disturbance_alerts(
    region: Dict[str, Any],
    start_date: str,
    end_date: str,
    alert_types: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 200,
) -> Dict[str, Any]:
    """
    Acquire forest disturbance alerts from Global Forest Watch across requested alert datasets.
    Prioritizes `wur_radd_alerts` (C-band SAR) and `umd_glad_sentinel2_alerts` / `umd_glad_landsat_alerts`.
    """
    start_dt, end_dt, _ = validate_date_range(start_date, end_date)
    requested_datasets = alert_types or [
        "wur_radd_alerts",
        "umd_glad_sentinel2_alerts",
        "umd_glad_landsat_alerts",
    ]
    for ds in requested_datasets:
        if ds not in GFW_DATASETS:
            raise ValueError(f"Unsupported forest alert dataset '{ds}'. Supported: {list(GFW_DATASETS.keys())}")

    client = get_gfw_client()
    live_records: List[Dict[str, Any]] = []
    is_live = False

    if client.has_api_key:
        for ds_id in requested_datasets:
            rows = client.query_dataset_alerts(
                dataset_id=ds_id,
                geometry_geojson=region["geometry"],
                start_date=start_date,
                end_date=end_date,
                page_number=page,
                page_size=page_size,
            )
            if rows is not None:
                is_live = True
                ds_info = GFW_DATASETS[ds_id]
                for i, r in enumerate(rows):
                    lon = float(r.get("longitude", region["center"][0]))
                    lat = float(r.get("latitude", region["center"][1]))
                    d_deg = 0.0025
                    live_records.append(
                        {
                            "source_record_id": f"{ds_id}_{r.get('alert_date', start_date)}_{i}",
                            "dataset_id": ds_id,
                            "alert_family": ds_info["alert_family"],
                            "provider": ds_info["provider"],
                            "instrument": ds_info["instrument"],
                            "resolution_m": ds_info["resolution_m"],
                            "alert_date": str(r.get("alert_date", start_date))[:10],
                            "confidence": str(r.get("confidence", "nominal")).lower(),
                            "geometry": mapping(box(lon - d_deg, lat - d_deg, lon + d_deg, lat + d_deg)),
                        }
                    )

    if not is_live:
        raw_records = _generate_reference_forest_alerts(region, start_dt, end_dt, requested_datasets)
    else:
        raw_records = live_records

    return {
        "is_live": is_live,
        "requested_datasets": requested_datasets,
        "raw_records": raw_records,
    }
