"""Forest disturbance alert normalization, geometry validation, and deduplication (Section 6.3)."""

from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import shape

from app.core.quality_control import validate_and_deduplicate_features
from app.core.regions import calculate_equal_area_km2
from app.models.layers import GeoJSONFeature, GeoJSONFeatureProperties


def normalize_forest_alerts(
    raw_records: List[Dict[str, Any]],
    region: Dict[str, Any],
    confidence_filter: Optional[str] = None,
) -> Tuple[List[GeoJSONFeature], List[str]]:
    """
    Normalize raw GFW alert records into validated GeoJSONFeatures:
      - Clip or verify intersection with study region geometry.
      - Preserve original source record ID, dataset identity (`wur_radd_alerts` vs `umd_glad_*`),
        and confidence level (`high` vs `nominal`).
      - Calculate equal-area km² using South America Albers Equal Area projection.
      - Validate and deduplicate geometries.
    """
    region_geom = shape(region["geometry"])
    features: List[GeoJSONFeature] = []
    warnings: List[str] = []

    for rec in raw_records:
        conf = str(rec.get("confidence", "nominal")).lower()
        if confidence_filter and conf != confidence_filter.lower():
            continue

        try:
            geom = shape(rec["geometry"])
        except Exception as exc:
            warnings.append(f"Invalid geometry in record {rec.get('source_record_id')}: {exc}")
            continue

        if not geom.intersects(region_geom):
            continue

        clipped = geom.intersection(region_geom)
        if clipped.is_empty:
            continue

        area_km2 = calculate_equal_area_km2(clipped)
        ds_id = rec["dataset_id"]
        src_id = rec["source_record_id"]

        features.append(
            GeoJSONFeature(
                id=src_id,
                geometry=rec["geometry"],
                properties=GeoJSONFeatureProperties(
                    feature_id=src_id,
                    dataset="gfw-alerts",
                    classification=rec["alert_family"],
                    observation_date=rec["alert_date"],
                    confidence=conf,
                    area_km2=area_km2,
                    source=ds_id,
                    quality_level="high" if conf == "high" else "medium",
                    extra={
                        "original_source_id": src_id,
                        "gfw_dataset_id": ds_id,
                        "provider": rec["provider"],
                        "instrument": rec["instrument"],
                        "resolution_m": rec["resolution_m"],
                        "scientific_note": (
                            "Disturbance alert indicates canopy loss or structural change; does not automatically prove permanent deforestation."
                        ),
                    },
                ),
            )
        )

    cleaned, dedup_warnings = validate_and_deduplicate_features(features)
    warnings.extend(dedup_warnings)
    return cleaned, warnings
