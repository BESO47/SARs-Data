"""
Proximity analysis between Sentinel-1 flood inundation and Global Forest Watch alerts (Section 9.2).

Clearly distinguishes:
  - Direct intersection (`distance == 0 km`)
  - Proximity buffer (`0 < distance <= distance_km`, default `5.0 km`)
  - Total within distance (`0 <= distance <= distance_km`)
"""

from typing import Any, Dict, List

from shapely.geometry import shape
from shapely.ops import unary_union

from app.core.regions import project_to_equal_area
from app.models.layers import GeoJSONFeature
from app.models.metadata import DataStatus, DatasetMetadata, QualityLevel, QualityMetadata, ResultNature
from app.models.statistics import StatisticItem


def compute_flood_forest_proximity(
    region: Dict[str, Any],
    flood_features: List[GeoJSONFeature],
    forest_features: List[GeoJSONFeature],
    start_date: str,
    end_date: str,
    distance_km: float = 5.0,
    overall_status: DataStatus = DataStatus.DEMO,
) -> Dict[str, Any]:
    """
    Calculate the area of forest disturbance alerts within `distance_km` (default 5.0 km)
    of flood inundation polygons in South America Albers Equal Area projection (meters).
    Explicitly separates direct intersection from non-overlapping buffer proximity.
    """
    if distance_km <= 0.0 or distance_km > 100.0:
        raise ValueError("distance_km must be between 0.1 and 100.0 km.")

    inundated_polys_wgs84 = [
        shape(f.geometry)
        for f in flood_features
        if f.properties.classification in ("open_water_inundation", "possible_flooded_vegetation")
    ]
    forest_polys_wgs84 = [shape(f.geometry) for f in forest_features]

    direct_intersection_km2 = 0.0
    buffer_only_proximity_km2 = 0.0
    total_within_distance_km2 = 0.0
    proximate_alert_count = 0

    if inundated_polys_wgs84 and forest_polys_wgs84:
        flood_eq = project_to_equal_area(unary_union(inundated_polys_wgs84))
        buffer_m = distance_km * 1000.0
        buffered_flood_eq = flood_eq.buffer(buffer_m)
        # Ring outside direct flood polygon (0 < d <= distance_km)
        buffer_ring_eq = buffered_flood_eq.difference(flood_eq)

        for f_geom_wgs in forest_polys_wgs84:
            f_eq = project_to_equal_area(f_geom_wgs)
            if not f_eq.intersects(buffered_flood_eq):
                continue
            proximate_alert_count += 1

            inter_direct = f_eq.intersection(flood_eq)
            inter_ring = f_eq.intersection(buffer_ring_eq)

            direct_intersection_km2 += abs(inter_direct.area) / 1_000_000.0
            buffer_only_proximity_km2 += abs(inter_ring.area) / 1_000_000.0

    direct_intersection_km2 = round(direct_intersection_km2, 4)
    buffer_only_proximity_km2 = round(buffer_only_proximity_km2, 4)
    total_within_distance_km2 = round(direct_intersection_km2 + buffer_only_proximity_km2, 4)

    quality = QualityMetadata(
        level=QualityLevel.MEDIUM,
        observation_count=proximate_alert_count,
        spatial_coverage_pct=100.0,
        confidence_method=f"albers_equal_area_euclidean_buffer_{distance_km}km",
        warnings=[],
        limitations=[
            f"Clearly distinguishes direct intersection (0 km) from adjacent proximity buffer (0 to {distance_km} km).",
            "Proximity in floodplain corridors reflects riparian topography and river access routes; do not interpret spatial proximity alone as causal proof.",
        ],
    )

    metadata = DatasetMetadata(
        dataset="combined-proximity",
        provider="Amazon Sentinel Multi-Sensor Fusion (ESA S1 + WRI GFW)",
        instrument="Sentinel-1 C-band SAR + GFW RADD/GLAD",
        product_id=f"S1_FLOOD_BUFFER_{distance_km}KM_GFW",
        status=overall_status,
        result_nature=ResultNature.DERIVED,
        observation_start=start_date,
        observation_end=end_date,
        region_id=region["region_id"],
        geographic_coverage={"bbox": region["bbox"], "area_km2": region["area_km2"]},
        crs="EPSG:4326",
        area_calculation_crs="ESRI:102033",
        resolution_m=10.0,
        method=f"albers_equal_area_buffer_{distance_km}km_difference_partition",
        quality=quality,
        extra={"distance_threshold_km": distance_km},
    )

    stats = [
        StatisticItem(
            metric_id="disturbance_within_buffer_total_km2",
            label=f"Total Forest Disturbance Area Within {distance_km} km of Flood (Intersection + Proximity)",
            value=total_within_distance_km2,
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source="COPERNICUS/S1_GRD + GFW_ALERTS",
            processing_method=f"Area(ForestDisturbance ∩ Buffer(Flood, {distance_km}km))",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=overall_status,
        ),
        StatisticItem(
            metric_id="disturbance_proximity_buffer_only_km2",
            label=f"Adjacent Proximity Only (0 < distance <= {distance_km} km, Excluding Direct Overlap)",
            value=buffer_only_proximity_km2,
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source="COPERNICUS/S1_GRD + GFW_ALERTS",
            processing_method=f"Area(ForestDisturbance ∩ (Buffer(Flood, {distance_km}km) - Flood))",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=overall_status,
            notes="Strictly excludes direct polygon intersection.",
        ),
        StatisticItem(
            metric_id="disturbance_direct_intersection_km2",
            label="Direct Polygon Intersection Area (distance = 0 km)",
            value=direct_intersection_km2,
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source="COPERNICUS/S1_GRD ∩ GFW_ALERTS",
            processing_method="Area(ForestDisturbance ∩ Flood)",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=overall_status,
        ),
    ]

    return {
        "distance_threshold_km": distance_km,
        "direct_intersection_km2": direct_intersection_km2,
        "proximity_buffer_only_km2": buffer_only_proximity_km2,
        "total_within_distance_km2": total_within_distance_km2,
        "proximate_alert_count": proximate_alert_count,
        "statistics": stats,
        "metadata": metadata,
    }
