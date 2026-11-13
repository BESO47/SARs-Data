"""Forest disturbance statistics, timelines, and CesiumJS layer outputs (Section 6.4)."""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from app.core.quality_control import assess_observation_quality
from app.models.layers import (
    CesiumLayerResponse,
    GeoJSONFeatureCollection,
    LayerLegend,
    LegendItem,
)
from app.models.metadata import DataStatus, DatasetMetadata, ResultNature
from app.models.statistics import (
    StatisticItem,
    StatisticsResponse,
    TimelineEntry,
    TimelineResponse,
)
from app.processing.forest.acquisition import acquire_forest_disturbance_alerts
from app.processing.forest.normalization import normalize_forest_alerts

FOREST_LEGEND = LayerLegend(
    title="Global Forest Watch Canopy Disturbance Alerts (RADD & GLAD)",
    unit="Alert Source & Confidence",
    items=[
        LegendItem(
            code="RADD_SAR",
            label="WUR RADD Radar Disturbance (Sentinel-1 C-band)",
            color_hex="#FF1744",
            description="Cloud-penetrating C-band SAR forest canopy disturbance alert (10m).",
        ),
        LegendItem(
            code="GLAD_S2_OPTICAL",
            label="UMD GLAD-S2 Optical Disturbance (Sentinel-2)",
            color_hex="#FF9100",
            description="Optical multispectral forest canopy disturbance alert (10m).",
        ),
        LegendItem(
            code="GLAD_L_OPTICAL",
            label="UMD GLAD-L Optical Disturbance (Landsat 8/9)",
            color_hex="#FFD600",
            description="Optical multispectral forest canopy disturbance alert (30m).",
        ),
    ],
)


def run_forest_disturbance_pipeline(
    region: Dict[str, Any],
    start_date: str,
    end_date: str,
    alert_types: Optional[List[str]] = None,
    confidence_filter: Optional[str] = None,
    page: int = 1,
    page_size: int = 200,
) -> Dict[str, Any]:
    """
    Execute Global Forest Watch acquisition, normalization, area calculation, and output generation.
    Preserves separate dataset identities (`wur_radd_alerts`, `umd_glad_sentinel2_alerts`, `umd_glad_landsat_alerts`).
    """
    acq = acquire_forest_disturbance_alerts(
        region=region,
        start_date=start_date,
        end_date=end_date,
        alert_types=alert_types,
        page=page,
        page_size=page_size,
    )
    features, norm_warnings = normalize_forest_alerts(
        raw_records=acq["raw_records"],
        region=region,
        confidence_filter=confidence_filter,
    )

    is_live = acq["is_live"]
    limitations = [
        "Forest disturbance alerts detect canopy cover loss or structural disturbance (logging, windthrow, fire, clearing, or severe flood dieback) and do not automatically prove permanent land-use conversion / deforestation.",
        "Optical GLAD alerts (Landsat / Sentinel-2) can experience latency during persistent Amazon wet-season cloud cover, whereas RADD (Sentinel-1 C-band radar) penetrates clouds.",
        "RADD and GLAD alerts are preserved with separate dataset identifiers and must not be conflated as a single sensor.",
    ]
    extra_warnings = list(norm_warnings)
    if not is_live:
        extra_warnings.append(
            "GFW_API_KEY not configured in .env; normalized reference RADD & GLAD disturbance records returned with status='demo'."
        )

    raw_status, quality = assess_observation_quality(
        observation_count=len(features),
        min_required_observations=1,
        spatial_coverage_pct=100.0 if features else 0.0,
        base_limitations=limitations,
        extra_warnings=extra_warnings,
        confidence_method="gfw_source_alert_confirmation_level (nominal vs high)",
    )
    final_status = (
        DataStatus.INSUFFICIENT_OBSERVATIONS
        if raw_status == DataStatus.INSUFFICIENT_OBSERVATIONS
        else (DataStatus.AVAILABLE if is_live else DataStatus.DEMO)
    )

    acq_dates = sorted({f.properties.observation_date for f in features if f.properties.observation_date})
    metadata = DatasetMetadata(
        dataset="gfw-alerts",
        provider="World Resources Institute / Global Forest Watch (WUR & UMD GLAD)",
        instrument="Sentinel-1 SAR (RADD) + Sentinel-2 MSI (GLAD-S2) + Landsat 8/9 (GLAD-L)",
        product_id=",".join(acq["requested_datasets"]),
        status=final_status,
        result_nature=ResultNature.OBSERVED if is_live else ResultNature.ILLUSTRATIVE,
        observation_start=start_date,
        observation_end=end_date,
        acquisition_dates=acq_dates,
        region_id=region["region_id"],
        geographic_coverage={"bbox": region["bbox"], "area_km2": region["area_km2"]},
        crs="EPSG:4326",
        resolution_m=10.0,
        method="multi_source_alert_normalization_and_albers_area_integration",
        quality=quality,
        extra={
            "datasets_queried": acq["requested_datasets"],
            "confidence_filter": confidence_filter,
        },
    )

    # Paginate features for GeoJSON response
    total_features = len(features)
    start_idx = max(page - 1, 0) * page_size
    paginated_features = features[start_idx : start_idx + page_size]

    geojson_fc = GeoJSONFeatureCollection(
        metadata=metadata,
        total_features=total_features,
        page=page,
        page_size=page_size,
        bbox=region["bbox"],
        features=paginated_features,
    )

    # Aggregate statistics by dataset source and confidence without mixing identities
    by_dataset_count: Dict[str, int] = defaultdict(int)
    by_dataset_area_km2: Dict[str, float] = defaultdict(float)
    by_confidence_count: Dict[str, int] = defaultdict(int)
    by_confidence_area_km2: Dict[str, float] = defaultdict(float)
    by_date_area: Dict[str, Dict[str, Any]] = {}

    total_area_km2 = 0.0
    for f in features:
        ds_src = f.properties.source
        conf = f.properties.confidence or "nominal"
        area = float(f.properties.area_km2 or 0.0)
        dt = f.properties.observation_date or start_date

        total_area_km2 += area
        by_dataset_count[ds_src] += 1
        by_dataset_area_km2[ds_src] = round(by_dataset_area_km2[ds_src] + area, 4)
        by_confidence_count[conf] += 1
        by_confidence_area_km2[conf] = round(by_confidence_area_km2[conf] + area, 4)

        if dt not in by_date_area:
            by_date_area[dt] = {"count": 0, "area_km2": 0.0, "sources": set()}
        by_date_area[dt]["count"] += 1
        by_date_area[dt]["area_km2"] = round(by_date_area[dt]["area_km2"] + area, 4)
        by_date_area[dt]["sources"].add(ds_src)

    total_area_km2 = round(total_area_km2, 4)

    stats_items = [
        StatisticItem(
            metric_id="total_alert_count",
            label="Total Forest Disturbance Alerts",
            value=total_features,
            unit="alerts",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=",".join(acq["requested_datasets"]),
            processing_method=metadata.method,
            result_nature=ResultNature.OBSERVED if is_live else ResultNature.ILLUSTRATIVE,
            quality=quality,
            status=final_status,
            notes="Includes RADD radar and GLAD optical canopy disturbance alerts.",
        ),
        StatisticItem(
            metric_id="total_alert_area_km2",
            label="Estimated Canopy Disturbance Alert Area",
            value=total_area_km2,
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=",".join(acq["requested_datasets"]),
            processing_method="albers_equal_area_polygon_integration",
            result_nature=ResultNature.ESTIMATED,
            quality=quality,
            status=final_status,
            notes="Calculated in South America Albers Equal Area Conic projection.",
        ),
        StatisticItem(
            metric_id="radd_radar_alert_area_km2",
            label="WUR RADD Radar Disturbance Area (Sentinel-1)",
            value=round(by_dataset_area_km2.get("wur_radd_alerts", 0.0), 4),
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source="wur_radd_alerts",
            processing_method="radd_sar_disturbance_filter",
            result_nature=ResultNature.ESTIMATED,
            quality=quality,
            status=final_status,
        ),
        StatisticItem(
            metric_id="high_confidence_alert_area_km2",
            label="Confirmed High-Confidence Disturbance Area",
            value=round(by_confidence_area_km2.get("high", 0.0), 4),
            unit="km²",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=",".join(acq["requested_datasets"]),
            processing_method="confidence_eq_high_filter",
            result_nature=ResultNature.ESTIMATED,
            quality=quality,
            status=final_status,
        ),
    ]

    statistics_response = StatisticsResponse(
        dataset="gfw-alerts",
        region_id=region["region_id"],
        status=final_status,
        metadata=metadata,
        statistics=stats_items,
        breakdown={
            "counts_by_dataset": dict(by_dataset_count),
            "area_km2_by_dataset": dict(by_dataset_area_km2),
            "counts_by_confidence": dict(by_confidence_count),
            "area_km2_by_confidence": dict(by_confidence_area_km2),
        },
    )

    timeline_entries: List[TimelineEntry] = []
    for dt in sorted(by_date_area.keys()):
        info = by_date_area[dt]
        srcs = sorted(info["sources"])
        timeline_entries.append(
            TimelineEntry(
                timestamp=f"{dt}T00:00:00Z",
                dataset="gfw-alerts",
                event_type="forest_disturbance_alert",
                value=info["area_km2"],
                unit="km²",
                label=f"{info['count']} alerts ({', '.join(srcs)})",
                product_id=",".join(srcs),
                status=final_status,
                details={"alert_count": info["count"], "sources": srcs},
            )
        )

    timeline_response = TimelineResponse(
        dataset="gfw-alerts",
        region_id=region["region_id"],
        status=final_status,
        unit="km²",
        temporal_resolution="daily alert aggregation",
        metadata=metadata,
        series=timeline_entries,
    )

    tile_url = (
        f"/api/tiles/forest/{region['region_id']}/{{z}}/{{x}}/{{y}}.png"
        f"?start_date={start_date}&end_date={end_date}"
    )

    layer_response = CesiumLayerResponse(
        layer_id=f"gfw-forest-{region['region_id']}-{start_date}-{end_date}",
        dataset="gfw-alerts",
        title=f"GFW Forest Disturbance Alerts ({region['name']})",
        source="WRI / Global Forest Watch (RADD & GLAD)",
        layer_type="geojson",
        observation_start=start_date,
        observation_end=end_date,
        tile_url_template=tile_url,
        feature_endpoint=f"/api/forest/alerts?region_id={region['region_id']}&start_date={start_date}&end_date={end_date}",
        geojson=geojson_fc,
        bbox=region["bbox"],
        default_opacity=0.85,
        z_index=30,
        legend=FOREST_LEGEND,
        status=final_status,
        quality=quality,
        metadata=metadata,
    )

    return {
        "alerts": geojson_fc,
        "all_features": features,
        "layer": layer_response,
        "statistics": statistics_response,
        "timeline": timeline_response,
    }
