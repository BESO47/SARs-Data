"""Global Forest Watch (GFW) Data API client for RADD and GLAD disturbance alerts (Section 6.1 & 6.2)."""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger("amazon_sentinel.gfw")

# Official Global Forest Watch Data API v3 dataset identifiers and schemas (Section 6.1)
GFW_DATASETS: Dict[str, Dict[str, Any]] = {
    "wur_radd_alerts": {
        "dataset_id": "wur_radd_alerts",
        "name": "WUR RADD Radar Forest Disturbance Alerts (Sentinel-1 C-band)",
        "provider": "Wageningen University & Research (WUR) / Global Forest Watch",
        "instrument": "Sentinel-1 C-band SAR (10m)",
        "resolution_m": 10.0,
        "date_column": "wur_radd_alerts__date",
        "confidence_column": "wur_radd_alerts__confidence",
        "alert_family": "RADD_SAR",
    },
    "umd_glad_sentinel2_alerts": {
        "dataset_id": "umd_glad_sentinel2_alerts",
        "name": "UMD GLAD-S2 Optical Forest Disturbance Alerts (Sentinel-2)",
        "provider": "University of Maryland (GLAD) / Global Forest Watch",
        "instrument": "Sentinel-2 MSI (10m)",
        "resolution_m": 10.0,
        "date_column": "umd_glad_sentinel2_alerts__date",
        "confidence_column": "umd_glad_sentinel2_alerts__confidence",
        "alert_family": "GLAD_S2_OPTICAL",
    },
    "umd_glad_landsat_alerts": {
        "dataset_id": "umd_glad_landsat_alerts",
        "name": "UMD GLAD-L Optical Forest Disturbance Alerts (Landsat)",
        "provider": "University of Maryland (GLAD) / Global Forest Watch",
        "instrument": "Landsat 8/9 OLI (30m)",
        "resolution_m": 30.0,
        "date_column": "umd_glad_landsat_alerts__date",
        "confidence_column": "umd_glad_landsat_alerts__confidence",
        "alert_family": "GLAD_L_OPTICAL",
    },
}


class GlobalForestWatchClient:
    """
    Client for the official Global Forest Watch Data API (`https://data-api.globalforestwatch.org`).
    Handles:
      - Separate dataset endpoints (`wur_radd_alerts`, `umd_glad_sentinel2_alerts`, `umd_glad_landsat_alerts`).
      - Pagination (`page[number]`, `page[size]`).
      - Rate limit handling (HTTP 429 exponential backoff).
      - Secure API key transmission via `x-api-key` header without leaking secrets in logs.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.gfw_api_base_url.rstrip("/")

    @property
    def has_api_key(self) -> bool:
        return self.settings.has_gfw_configured()

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.has_api_key and self.settings.gfw_api_key:
            headers["x-api-key"] = self.settings.gfw_api_key.get_secret_value()
        return headers

    def query_dataset_alerts(
        self,
        dataset_id: str,
        geometry_geojson: Dict[str, Any],
        start_date: str,
        end_date: str,
        page_number: int = 1,
        page_size: int = 200,
        max_retries: int = 2,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Query a single GFW dataset endpoint via POST `/dataset/{dataset_id}/latest/query`.
        Returns list of raw records if live query succeeds, or None if API key is missing / offline.
        """
        if dataset_id not in GFW_DATASETS:
            raise ValueError(f"Unsupported GFW dataset '{dataset_id}'. Supported: {list(GFW_DATASETS.keys())}")

        if not self.has_api_key:
            return None

        ds_cfg = GFW_DATASETS[dataset_id]
        dt_col = ds_cfg["date_column"]
        conf_col = ds_cfg["confidence_column"]
        offset = max(page_number - 1, 0) * page_size

        sql = (
            f"SELECT latitude, longitude, {dt_col} AS alert_date, {conf_col} AS confidence, "
            f"umd_tree_cover_density_2000__threshold, area__ha "
            f"FROM results "
            f"WHERE {dt_col} >= '{start_date}' AND {dt_col} <= '{end_date}' "
            f"LIMIT {page_size} OFFSET {offset}"
        )
        url = f"{self.base_url}/dataset/{dataset_id}/latest/query"
        payload = {"sql": sql, "geometry": geometry_geojson}

        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=12.0, follow_redirects=True) as client:
                    resp = client.post(url, headers=self._headers(), json=payload)
                if resp.status_code == 429:
                    backoff = 1.5 * (2**attempt)
                    logger.warning("GFW API rate limit (429) for %s; backing off %.1fs", dataset_id, backoff)
                    time.sleep(backoff)
                    continue
                if resp.status_code == 200:
                    body = resp.json()
                    data = body.get("data", [])
                    if isinstance(data, list):
                        return data
                logger.warning("GFW API returned HTTP %d for %s", resp.status_code, dataset_id)
                return None
            except Exception as exc:
                logger.warning("GFW API connection unavailable for %s: %s", dataset_id, exc)
                return None

        return None


_gfw_client: Optional[GlobalForestWatchClient] = None


def get_gfw_client() -> GlobalForestWatchClient:
    """Return singleton GlobalForestWatchClient instance."""
    global _gfw_client
    if _gfw_client is None:
        _gfw_client = GlobalForestWatchClient()
    return _gfw_client
