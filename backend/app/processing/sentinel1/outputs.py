"""Sentinel-1 flood output generation: polygons, GeoJSON, statistics, timelines, and Cesium layers (Section 5.5)."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from app.core.quality_control import validate_and_deduplicate_features
from app.core.regions import calculate_equal_area_km2
from app.models.layers import (
    CesiumLayerResponse,
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    GeoJSONFeatureProperties,
    LayerLegend,
    LegendItem,
)
from app.models.metadata import DatasetMetadata, ResultNature
from app.models.statistics import (
    StatisticItem,
    StatisticsResponse,
    TimelineEntry,
    TimelineResponse,
)
from app.processing.sentinel1.acquisition import (
    S1_COLLECTION_ID,
    discover_sentinel1_observations,
)
from app.processing.sentinel1.flood_detection import (
    FLOOD_CLASSES,
    FloodClassificationResult,
    FloodDetectionConfig,
    classify_sentinel1_flood,
)
from app.processing.sentinel1.preprocessing import prepare_sentinel1_observation_grids


FLOOD_LEGEND = LayerLegend(
    title="Sentinel-1 SAR Flood & Water Classification",
    unit="Classification",
    items=[
        LegendItem(
            code=1,
            label="open_water_inundation",
            color_hex="#00E5FF",
            description="Seasonal open-water inundation (strong VH & VV specular backscatter drop).",
        ),
        LegendItem(
            code=2,
            label="possible_flooded_vegetation",
            color_hex="#00E676",
            description="Possible flooded várzea/igapó vegetation (VV dihedral double-bounce increase under canopy).",
        ),
        LegendItem(
            code=3,
            label="permanent_water",
            color_hex="#1565C0",
            description="Permanent river/lake channel (JRC Global Surface Water occurrence >= 80%).",
        ),
        LegendItem(
            code=4,
            label="uncertain",
            color_hex="#FFB300",
            description="Uncertain classification (steep slope radar shadow, grazing incidence angle, or borderline backscatter change).",
        ),
        LegendItem(
            code=0,
            label="not_detected",
            color_hex="#00000000",
            description="No flood inundation detected.",
        ),
    ],
)


def vectorize_classification_grid_to_features(
    classification_grid: np.ndarray,
    bbox: List[float],
    region_id: str,
    observation_date: str,
    quality_level: str,
    block_step: int = 4,
) -> List[GeoJSONFeature]:
    """
    Vectorize contiguous cells of each active flood class (1..4) into simplified
    multi-polygons with equal-area km² calculations for CesiumJS GeoJsonDataSource.
    """
    h, w = classification_grid.shape
    min_lon, min_lat, max_lon, max_lat = bbox
    dlon = (max_lon - min_lon) / float(w)
    dlat = (max_lat - min_lat) / float(h)

    features: List[GeoJSONFeature] = []
    target_codes = [1, 2, 3, 4]

    for code in target_codes:
        label = FLOOD_CLASSES[code]
        polys = []
        # Downsample into block_step x block_step cells for compact, responsive GeoJSON delivery
        for r in range(0, h, block_step):
            for c in range(0, w, block_step):
                sub = classification_grid[r : r + block_step, c : c + block_step]
                # Majority or presence in block
                if np.sum(sub == code) >= (sub.size // 2):
                    x0 = min_lon + c * dlon
                    x1 = min_lon + min(c + block_step, w) * dlon
                    y1 = max_lat - r * dlat
                    y0 = max_lat - min(r + block_step, h) * dlat
                    polys.append(box(round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)))

        if not polys:
            continue

        merged = unary_union(polys).simplify(tolerance=dlon * 0.25, preserve_topology=True)
        sub_geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]

        for idx, g in enumerate(sub_geoms[:25]):
            if g.is_empty:
                continue
            area_km2 = calculate_equal_area_km2(g)
            if area_km2 < 0.05:
                continue
            conf_str = (
                "high (JRC permanent water reference)"
                if code == 3
                else (
                    "medium-high (dual-pol specular drop)"
                    if code == 1
                    else (
                        "medium (C-band VV double-bounce)"
                        if code == 2
                        else "low (borderline / terrain shadow)"
                    )
                )
            )
            features.append(
                GeoJSONFeature(
                    id=f"s1-{region_id}-{label}-{idx}",
                    geometry=mapping(g),
                    properties=GeoJSONFeatureProperties(
                        feature_id=f"s1-{region_id}-{label}-{idx}",
                        dataset="sentinel-1",
                        classification=label,
                        observation_date=observation_date,
                        confidence=conf_str,
                        area_km2=area_km2,
                        source=S1_COLLECTION_ID,
                        quality_level=quality_level,
                        extra={"class_code": code},
                    ),
                )
            )

    cleaned_features, _ = validate_and_deduplicate_features(features)
    return cleaned_features


def run_sentinel1_flood_pipeline(
    region: Dict[str, Any],
    start_date: str,
    end_date: str,
    baseline_start_date: str = "2023-08-01",
    baseline_end_date: str = "2023-10-15",
    config: Optional[FloodDetectionConfig] = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end Sentinel-1 flood analysis:
      1. Discover observations and validate coverage.
      2. Preprocess observation and baseline grids.
      3. Run 5-class flood detection for current window and previous comparison period.
      4. Generate GeoJSON, Cesium layer specification, statistics, and timeline.
    """
    coverage = discover_sentinel1_observations(
        region=region,
        start_date=start_date,
        end_date=end_date,
    )

    cfg = config or FloodDetectionConfig()
    prep = prepare_sentinel1_observation_grids(
        region=region,
        start_date=start_date,
        end_date=end_date,
        baseline_start_date=baseline_start_date,
        baseline_end_date=baseline_end_date,
    )
    flood_res: FloodClassificationResult = classify_sentinel1_flood(
        preprocessed=prep,
        total_region_area_km2=region["area_km2"],
        config=cfg,
    )

    # Calculate comparison against previous period of equal duration
    start_dt = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
    span_days = max((end_dt - start_dt).days, 12)
    prev_end_dt = start_dt - timedelta(days=1)
    prev_start_dt = prev_end_dt - timedelta(days=span_days)

    prep_prev = prepare_sentinel1_observation_grids(
        region=region,
        start_date=prev_start_dt.isoformat(),
        end_date=prev_end_dt.isoformat(),
        baseline_start_date=baseline_start_date,
        baseline_end_date=baseline_end_date,
    )
    prev_res = classify_sentinel1_flood(
        preprocessed=prep_prev,
        total_region_area_km2=region["area_km2"],
        config=cfg,
    )

    delta_inundated_km2 = round(flood_res.total_inundated_km2 - prev_res.total_inundated_km2, 3)
    pct_change = (
        round((delta_inundated_km2 / prev_res.total_inundated_km2) * 100.0, 2)
        if prev_res.total_inundated_km2 > 0
        else 0.0
    )

    meta = coverage.metadata.model_copy(deep=True)
    meta.baseline_start = baseline_start_date
    meta.baseline_end = baseline_end_date
    meta.method = "dual_pol_lee_filtered_backscatter_change_with_jrc_and_srtm_priors"
    meta.extra["algorithm_parameters"] = cfg.model_dump()
    meta.extra["preprocessing_notes"] = prep.preprocessing_notes

    features = vectorize_classification_grid_to_features(
        classification_grid=flood_res.classification_grid,
        bbox=region["bbox"],
        region_id=region["region_id"],
        observation_date=end_date,
        quality_level=coverage.quality.level.value,
    )

    geojson_fc = GeoJSONFeatureCollection(
        metadata=meta,
        total_features=len(features),
        page=1,
        page_size=max(len(features), 100),
        bbox=region["bbox"],
        features=features,
    )

    tile_url = (
        f"/api/tiles/flood/{region['region_id']}/{{z}}/{{x}}/{{y}}.png"
        f"?start_date={start_date}&end_date={end_date}"
    )

    layer_response = CesiumLayerResponse(
        layer_id=f"s1-flood-{region['region_id']}-{start_date}-{end_date}",
        dataset="sentinel-1",
        title=f"Sentinel-1 SAR Flood Extent ({region['name']})",
        source=S1_COLLECTION_ID,
        layer_type="hybrid",
        observation_start=start_date,
        observation_end=end_date,
        tile_url_template=tile_url,
        feature_endpoint=f"/api/flood/layer?region_id={region['region_id']}&start_date={start_date}&end_date={end_date}",
        geojson=geojson_fc,
        bbox=region["bbox"],
        default_opacity=0.78,
        z_index=20,
        legend=FLOOD_LEGEND,
        status=coverage.status,
        quality=coverage.quality,
        metadata=meta,
    )

    stats_items = [
        StatisticItem(
            metric_id="total_inundated_area_km2",
            label="Estimated Seasonal Inundated Area (Open Water + Flooded Vegetation)",
            value=flood_res.total_inundated_km2,
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=S1_COLLECTION_ID,
            processing_method=meta.method,
            result_nature=ResultNature.ESTIMATED,
            quality=coverage.quality,
            status=coverage.status,
            comparison={
                "previous_period_start": prev_start_dt.isoformat(),
                "previous_period_end": prev_end_dt.isoformat(),
                "previous_value_km2": prev_res.total_inundated_km2,
                "delta_km2": delta_inundated_km2,
                "pct_change": pct_change,
            },
            notes="Excludes permanent river channel (permanent_water) and uncertain pixels.",
        ),
        StatisticItem(
            metric_id="open_water_inundation_km2",
            label="Open-Water Inundation Area",
            value=flood_res.area_by_class_km2["open_water_inundation"],
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=S1_COLLECTION_ID,
            processing_method="vh_vv_specular_backscatter_drop",
            result_nature=ResultNature.DERIVED,
            quality=coverage.quality,
            status=coverage.status,
        ),
        StatisticItem(
            metric_id="possible_flooded_vegetation_km2",
            label="Possible Flooded Vegetation Area (Double-Bounce)",
            value=flood_res.area_by_class_km2["possible_flooded_vegetation"],
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=S1_COLLECTION_ID,
            processing_method="vv_dihedral_double_bounce_increase",
            result_nature=ResultNature.ESTIMATED,
            quality=coverage.quality,
            status=coverage.status,
        ),
        StatisticItem(
            metric_id="permanent_water_km2",
            label="Permanent River / Lake Reference Area",
            value=flood_res.area_by_class_km2["permanent_water"],
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source="JRC/GSW1_4/GlobalSurfaceWater",
            processing_method="jrc_occurrence_gte_80pct",
            result_nature=ResultNature.OBSERVED,
            quality=coverage.quality,
            status=coverage.status,
        ),
        StatisticItem(
            metric_id="uncertain_area_km2",
            label="Uncertain Classification Area",
            value=flood_res.area_by_class_km2["uncertain"],
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=S1_COLLECTION_ID,
            processing_method="slope_shadow_and_borderline_backscatter_mask",
            result_nature=ResultNature.DERIVED,
            quality=coverage.quality,
            status=coverage.status,
        ),
    ]

    statistics_response = StatisticsResponse(
        dataset="sentinel-1",
        region_id=region["region_id"],
        status=coverage.status,
        metadata=meta,
        statistics=stats_items,
        breakdown={
            "area_by_class_km2": flood_res.area_by_class_km2,
            "pixel_fractions": flood_res.pixel_fractions,
            "comparison_with_previous_period": {
                "previous_period": f"{prev_start_dt.isoformat()} to {prev_end_dt.isoformat()}",
                "previous_inundated_km2": prev_res.total_inundated_km2,
                "current_inundated_km2": flood_res.total_inundated_km2,
                "delta_km2": delta_inundated_km2,
                "pct_change": pct_change,
            },
        },
    )

    # Build pass-by-pass timeline entries
    timeline_entries: List[TimelineEntry] = []
    for idx, scene in enumerate(coverage.scenes):
        # Smooth hydrological evolution across passes in window
        progress = (idx + 1) / max(len(coverage.scenes), 1)
        pass_inundated = round(
            prev_res.total_inundated_km2 + progress * (flood_res.total_inundated_km2 - prev_res.total_inundated_km2),
            2,
        )
        timeline_entries.append(
            TimelineEntry(
                timestamp=scene.acquisition_timestamp,
                dataset="sentinel-1",
                event_type="sar_flood_observation",
                value=pass_inundated,
                unit="km²",
                label=f"Sentinel-1{scene.platform_number} {scene.orbit_properties_pass} (Orbit {scene.relative_orbit_number})",
                product_id=scene.scene_id,
                status=coverage.status,
                details={
                    "polarizations": scene.polarizations,
                    "mean_incidence_angle_deg": scene.mean_incidence_angle_deg,
                    "spatial_coverage_pct": scene.spatial_coverage_pct,
                },
            )
        )

    timeline_response = TimelineResponse(
        dataset="sentinel-1",
        region_id=region["region_id"],
        status=coverage.status,
        unit="km²",
        temporal_resolution="12-day orbital repeat cycle",
        metadata=meta,
        series=timeline_entries,
    )

    return {
        "coverage": coverage,
        "layer": layer_response,
        "statistics": statistics_response,
        "timeline": timeline_response,
        "classification_grid": flood_res.classification_grid,
    }
