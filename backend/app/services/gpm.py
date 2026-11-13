"""NASA GPM IMERG V07 service and collection schema (Section 8.1)."""

from typing import Any, Dict

# Official Google Earth Engine collection schema for NASA GPM IMERG V07 (Section 8.1)
IMERG_COLLECTION_ID = "NASA/GPM_L3/IMERG_V07"
IMERG_PRECIPITATION_BAND = "precipitation"  # Calibrated precipitation rate in mm/hr
IMERG_QUALITY_BAND = "probabilityLiquidPrecip"
IMERG_RANDOM_ERROR_BAND = "randomError"  # mm/hr
IMERG_TEMPORAL_INTERVAL_HOURS = 0.5  # 30-minute half-hourly granules
IMERG_RESOLUTION_M = 11132.0  # 0.1 degree (~11.1 km at Equator)


def get_imerg_collection_metadata() -> Dict[str, Any]:
    """Return verified collection metadata for NASA/GPM_L3/IMERG_V07."""
    return {
        "collection_id": IMERG_COLLECTION_ID,
        "provider": "NASA Goddard Earth Sciences Data and Information Services Center (GES DISC)",
        "instrument": "GPM Constellation Passive Microwave + IR (IMERG V07)",
        "primary_band": IMERG_PRECIPITATION_BAND,
        "native_rate_unit": "mm/hr",
        "accumulated_unit": "mm",
        "temporal_interval_hours": IMERG_TEMPORAL_INTERVAL_HOURS,
        "integration_formula": "accumulated_mm = sum(precipitation_mm_per_hr * 0.5_hr)",
        "resolution_m": IMERG_RESOLUTION_M,
    }
