"""
Spatial intersection between Sentinel-1 flood inundation and Global Forest Watch disturbance alerts (Section 9.1).

Computes exact polygon intersection in South America Albers Equal Area Conic (`ESRI:102033`):
    Area(FloodGeometry ∩ ForestDisturbanceGeometry)
"""

from typing import Any, Dict, List

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from app.core.regions import project_to_equal_area
from app.models.layers import (
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    GeoJSONFeatureProperties,
)
from app.models.metadata import DataStatus, DatasetMetadata, QualityLevel, QualityMetadata, ResultNature
from app.models.statistics import StatisticItem


def compute_flood_forest_overlap(
    region: Dict[str, Any],
    flood_features: List[GeoJSONFeature],
    forest_features: List[GeoJSONFeature],
    start_date: str,
    end_date: str,
    overall_status: DataStatus = DataStatus.DEMO,
) -> Dict[str, Any]:
    """
    Calculate spatial intersection `Area(FloodGeometry ∩ ForestDisturbanceGeometry)`:
      - Overlapping area in km².
      - Percentage of flood area overlapping disturbance alerts.
      - Percentage of disturbance-alert area overlapping possible inundation.
      - Intersection GeoJSON FeatureCollection for CesiumJS rendering.
    """
    # Filter flood features to active inundation classes (`open_water_inundation` and `possible_flooded_vegetation`)
    inundated_polys_wgs84 = [
        shape(f.geometry)
        for f in flood_features
        if f.properties.classification in ("open_water_inundation", "possible_flooded_vegetation")
    ]
    forest_polys_wgs84 = [shape(f.geometry) for f in forest_features]

    inundated_union_wgs84 = unary_union(inundated_polys_wgs84) if inundated_polys_wgs84 else None
    forest_union_wgs84 = unary_union(forest_polys_wgs84) if forest_polys_wgs84 else None

    flood_area_km2 = 0.0
    if inundated_union_wgs84 and not inundated_union_wgs84.is_empty:
        flood_area_km2 = round(abs(project_to_equal_area(inundated_union_wgs84).area) / 1_000_000.0, 4)

    forest_area_km2 = 0.0
    if forest_union_wgs84 and not forest_union_wgs84.is_empty:
        forest_area_km2 = round(abs(project_to_equal_area(forest_union_wgs84).area) / 1_000_000.0, 4)

    intersection_features: List[GeoJSONFeature] = []
    overlap_area_km2 = 0.0

    if inundated_union_wgs84 and forest_union_wgs84 and not inundated_union_wgs84.is_empty and not forest_union_wgs84.is_empty:
        for idx, f_feat in enumerate(forest_features):
            f_geom = shape(f_feat.geometry)
            if not f_geom.intersects(inundated_union_wgs84):
                continue
            inter_geom = f_geom.intersection(inundated_union_wgs84)
            if inter_geom.is_empty:
                continue
            inter_eq = project_to_equal_area(inter_geom)
            piece_km2 = round(abs(inter_eq.area) / 1_000_000.0, 4)
            if piece_km2 <= 0.0001:
                continue
            overlap_area_km2 += piece_km2
            intersection_features.append(
                GeoJSONFeature(
                    id=f"overlap-{region['region_id']}-{idx:03d}",
                    geometry=mapping(inter_geom),
                    properties=GeoJSONFeatureProperties(
                        feature_id=f"overlap-{region['region_id']}-{idx:03d}",
                        dataset="combined-overlap",
                        classification="flood_forest_disturbance_intersection",
                        observation_date=f_feat.properties.observation_date,
                        confidence=f_feat.properties.confidence,
                        area_km2=piece_km2,
                        source=f"COPERNICUS/S1_GRD ∩ {f_feat.properties.source}",
                        quality_level="medium",
                        extra={
                            "forest_alert_id": f_feat.properties.feature_id,
                            "forest_alert_source": f_feat.properties.source,
                            "spatial_relationship": "direct_polygon_intersection",
                        },
                    ),
                )
            )

    overlap_area_km2 = round(overlap_area_km2, 4)
    pct_of_flood = round((overlap_area_km2 / flood_area_km2) * 100.0, 3) if flood_area_km2 > 0 else 0.0
    pct_of_forest = round((overlap_area_km2 / forest_area_km2) * 100.0, 3) if forest_area_km2 > 0 else 0.0

    quality = QualityMetadata(
        level=QualityLevel.MEDIUM,
        observation_count=len(intersection_features),
        spatial_coverage_pct=100.0,
        confidence_method="albers_equal_area_polygon_intersection",
        warnings=[],
        limitations=[
            "Spatial overlap indicates that forest canopy disturbance alerts and seasonal SAR flood inundation occurred within the same geographic footprint during the selected periods; it does not prove that flooding caused the disturbance or vice versa.",
            "Sentinel-1 flood polygons and GFW disturbance alerts have different sensor revisit intervals.",
        ],
    )

    metadata = DatasetMetadata(
        dataset="combined-overlap",
        provider="Amazon Sentinel Multi-Sensor Fusion (ESA S1 + WRI GFW)",
        instrument="Sentinel-1 C-band SAR + GFW RADD/GLAD",
        product_id="S1_GRD_INTERSECT_GFW_ALERTS",
        status=overall_status,
        result_nature=ResultNature.DERIVED,
        observation_start=start_date,
        observation_end=end_date,
        region_id=region["region_id"],
        geographic_coverage={"bbox": region["bbox"], "area_km2": region["area_km2"]},
        crs="EPSG:4326",
        area_calculation_crs="ESRI:102033",
        resolution_m=10.0,
        method="exact_shapely_polygon_intersection_in_albers_equal_area",
        quality=quality,
    )

    geojson_fc = GeoJSONFeatureCollection(
        metadata=metadata,
        total_features=len(intersection_features),
        bbox=region["bbox"],
        features=intersection_features,
    )

    stats = [
        StatisticItem(
            metric_id="overlap_area_km2",
            label="Direct Spatial Overlap Area (Flood ∩ Forest Disturbance)",
            value=overlap_area_km2,
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source="COPERNICUS/S1_GRD ∩ GFW_ALERTS",
            processing_method="Area(FloodGeometry ∩ ForestDisturbanceGeometry)",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=overall_status,
        ),
        StatisticItem(
            metric_id="pct_flood_overlapping_disturbance",
            label="Percentage of Flood Inundation Area Overlapping Disturbance Alerts",
            value=pct_of_flood,
            unit="%",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source="COPERNICUS/S1_GRD ∩ GFW_ALERTS",
            processing_method="overlap_area_km2 / flood_inundation_polygon_area_km2 * 100",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=overall_status,
        ),
        StatisticItem(
            metric_id="pct_disturbance_overlapping_flood",
            label="Percentage of Disturbance-Alert Area Overlapping Possible Inundation",
            value=pct_of_forest,
            unit="%",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source="COPERNICUS/S1_GRD ∩ GFW_ALERTS",
            processing_method="overlap_area_km2 / forest_disturbance_area_km2 * 100",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=overall_status,
        ),
    ]

    return {
        "overlap_area_km2": overlap_area_km2,
        "flood_polygon_area_km2": flood_area_km2,
        "forest_disturbance_area_km2": forest_area_km2,
        "pct_of_flood_area": pct_of_flood,
        "pct_of_disturbance_area": pct_of_forest,
        "intersection_count": len(intersection_features),
        "geojson": geojson_fc,
        "statistics": stats,
        "metadata": metadata,
    }
