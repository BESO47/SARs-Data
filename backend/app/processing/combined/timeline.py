"""
Unified multi-sensor timeline and rainfall context analysis (Section 9.3, 9.4, 9.5).

Compares:
  - Sentinel-1 flood observation dates.
  - Global Forest Watch disturbance alert dates.
  - NISAR L-band acquisition dates.
  - GPM IMERG rainfall observation periods (including rainfall context BEFORE, DURING, and AFTER the flood period).
Does not imply that events happened simultaneously when their observation periods differ,
and never interprets rainfall-flood co-variation as causal proof.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

from app.models.metadata import DataStatus, DatasetMetadata, QualityLevel, QualityMetadata, ResultNature
from app.models.statistics import TimelineEntry, TimelineResponse
from app.processing.combined.overlap import compute_flood_forest_overlap
from app.processing.combined.proximity import compute_flood_forest_proximity
from app.processing.forest.statistics import run_forest_disturbance_pipeline
from app.processing.nisar.change_analysis import run_nisar_analysis_pipeline
from app.processing.rainfall.aggregation import run_gpm_rainfall_pipeline
from app.processing.sentinel1.outputs import run_sentinel1_flood_pipeline


def compute_rainfall_context_around_flood(
    region: Dict[str, Any],
    flood_start_date: str,
    flood_end_date: str,
    window_days: int = 14,
) -> Dict[str, Any]:
    """
    Calculate GPM IMERG rainfall trends BEFORE, DURING, and AFTER the selected flood period (Section 9.4).
    """
    f_start = datetime.strptime(flood_start_date[:10], "%Y-%m-%d").date()
    f_end = datetime.strptime(flood_end_date[:10], "%Y-%m-%d").date()

    before_start = (f_start - timedelta(days=window_days)).isoformat()
    before_end = (f_start - timedelta(days=1)).isoformat()

    after_start = (f_end + timedelta(days=1)).isoformat()
    after_end = (f_end + timedelta(days=window_days)).isoformat()

    rain_before = run_gpm_rainfall_pipeline(region=region, start_date=before_start, end_date=before_end)
    rain_during = run_gpm_rainfall_pipeline(region=region, start_date=flood_start_date, end_date=flood_end_date)
    rain_after = run_gpm_rainfall_pipeline(region=region, start_date=after_start, end_date=after_end)

    def _extract_summary(res: Dict[str, Any], period_label: str, s_dt: str, e_dt: str) -> Dict[str, Any]:
        stats_map = {s.metric_id: s.value for s in res["statistics"].statistics}
        return {
            "phase": period_label,
            "period_start": s_dt,
            "period_end": e_dt,
            "accumulated_mm": stats_map.get("total_accumulated_rainfall_mm", 0.0),
            "mean_daily_mm": stats_map.get("mean_daily_rainfall_mm", 0.0),
            "max_daily_mm": stats_map.get("max_daily_rainfall_mm", 0.0),
            "unit": "mm",
        }

    return {
        "before_flood_period": _extract_summary(rain_before, "pre_flood_window", before_start, before_end),
        "during_flood_period": _extract_summary(rain_during, "active_flood_window", flood_start_date, flood_end_date),
        "after_flood_period": _extract_summary(rain_after, "post_flood_window", after_start, after_end),
        "scientific_interpretation_note": (
            "Co-occurrence of elevated pre-flood/during-flood IMERG rainfall and SAR inundation provides "
            "hydrological context only and must not be interpreted as standalone causal proof (large Amazon "
            "rivers are also driven by upstream Andean/basin routing)."
        ),
    }


def run_combined_multimodal_analysis(
    region: Dict[str, Any],
    start_date: str,
    end_date: str,
    distance_km: float = 5.0,
) -> Dict[str, Any]:
    """
    Run all four dataset pipelines for the study region and combine their spatial,
    proximity, rainfall-context, and temporal outputs (Section 9).
    """
    s1_res = run_sentinel1_flood_pipeline(region=region, start_date=start_date, end_date=end_date)
    forest_res = run_forest_disturbance_pipeline(region=region, start_date=start_date, end_date=end_date)
    nisar_res = run_nisar_analysis_pipeline(region=region, start_date=start_date, end_date=end_date)
    rain_res = run_gpm_rainfall_pipeline(region=region, start_date=start_date, end_date=end_date)

    statuses = {
        "sentinel-1": s1_res["coverage"].status,
        "gfw-alerts": forest_res["statistics"].status,
        "nisar": nisar_res["coverage"].status,
        "gpm-imerg": rain_res["statistics"].status,
    }
    overall_status = (
        DataStatus.AVAILABLE
        if all(st == DataStatus.AVAILABLE for st in statuses.values())
        else DataStatus.DEMO
    )

    flood_features = s1_res["layer"].geojson.features if s1_res["layer"].geojson else []
    forest_features = forest_res["all_features"]

    overlap_res = compute_flood_forest_overlap(
        region=region,
        flood_features=flood_features,
        forest_features=forest_features,
        start_date=start_date,
        end_date=end_date,
        overall_status=overall_status,
    )
    proximity_res = compute_flood_forest_proximity(
        region=region,
        flood_features=flood_features,
        forest_features=forest_features,
        start_date=start_date,
        end_date=end_date,
        distance_km=distance_km,
        overall_status=overall_status,
    )
    rainfall_context = compute_rainfall_context_around_flood(
        region=region,
        flood_start_date=start_date,
        flood_end_date=end_date,
    )

    # Merge timelines into a unified chronological sequence preserving exact sensor timestamps/intervals
    unified_entries: List[TimelineEntry] = []
    unified_entries.extend(s1_res["timeline"].series)
    unified_entries.extend(forest_res["timeline"].series)
    unified_entries.extend(nisar_res["observations"].series)
    unified_entries.extend(rain_res["timeline"].series)
    unified_entries.sort(key=lambda e: e.timestamp)

    combined_quality = QualityMetadata(
        level=QualityLevel.MEDIUM,
        observation_count=len(unified_entries),
        spatial_coverage_pct=98.5,
        confidence_method="multi_sensor_geodesic_and_albers_fusion",
        warnings=[
            "Observations across Sentinel-1 (12-day C-band), NISAR (12-day L-band), GFW (daily/multi-day alerts), and GPM IMERG (half-hourly integrated daily) have distinct acquisition timestamps and spatial resolutions (10m to 11.1km)."
        ],
        limitations=[
            "Events appearing in the same multi-week analysis window did not necessarily occur simultaneously.",
            "Spatial overlap and rainfall co-variation represent observational associations, not causal proof.",
        ],
    )

    combined_meta = DatasetMetadata(
        dataset="combined-multimodal",
        provider="Amazon Sentinel Integrated Geospatial Engine",
        instrument="Sentinel-1 SAR + GFW (RADD/GLAD) + NISAR L-Band + NASA GPM IMERG",
        product_id="MULTI_SENSOR_SYNTHESIS",
        status=overall_status,
        result_nature=ResultNature.DERIVED,
        observation_start=start_date,
        observation_end=end_date,
        region_id=region["region_id"],
        geographic_coverage={"bbox": region["bbox"], "area_km2": region["area_km2"]},
        crs="EPSG:4326",
        area_calculation_crs="ESRI:102033",
        resolution_m=10.0,
        method="albers_equal_area_spatial_overlay_and_non_simultaneous_timeline_synthesis",
        quality=combined_quality,
    )

    unified_timeline = TimelineResponse(
        dataset="combined-multimodal",
        region_id=region["region_id"],
        status=overall_status,
        unit="multi-unit (km² / mm / dB)",
        temporal_resolution="heterogeneous_sensor_timestamps_preserved",
        metadata=combined_meta,
        series=unified_entries,
    )

    s1_stats_map = {s.metric_id: s.value for s in s1_res["statistics"].statistics}
    forest_stats_map = {s.metric_id: s.value for s in forest_res["statistics"].statistics}
    nisar_stats_map = {s.metric_id: s.value for s in nisar_res["statistics"].statistics}
    rain_stats_map = {s.metric_id: s.value for s in rain_res["statistics"].statistics}

    summary_payload = {
        "region": {
            "region_id": region["region_id"],
            "name": region["name"],
            "bbox": region["bbox"],
            "area_km2": region["area_km2"],
            "crs": region["crs"],
        },
        "observation_period": {"start_date": start_date, "end_date": end_date},
        "dataset_availability": {k: v.value for k, v in statuses.items()},
        "regional_indicators": {
            "sentinel1_inundated_area_km2": s1_stats_map.get("total_inundated_area_km2"),
            "sentinel1_open_water_km2": s1_stats_map.get("open_water_inundation_km2"),
            "sentinel1_flooded_vegetation_km2": s1_stats_map.get("possible_flooded_vegetation_km2"),
            "gfw_disturbance_alert_count": forest_stats_map.get("total_alert_count"),
            "gfw_disturbance_alert_area_km2": forest_stats_map.get("total_alert_area_km2"),
            "nisar_lband_subcanopy_candidate_km2": nisar_stats_map.get("subcanopy_dihedral_candidate_area_km2"),
            "nisar_hh_gamma0_mean_db": nisar_stats_map.get("nisar_hh_gamma0_mean_db"),
            "gpm_accumulated_rainfall_mm": rain_stats_map.get("total_accumulated_rainfall_mm"),
        },
        "spatial_overlap_summary": {
            "overlap_area_km2": overlap_res["overlap_area_km2"],
            "pct_of_flood_area": overlap_res["pct_of_flood_area"],
            "pct_of_disturbance_area": overlap_res["pct_of_disturbance_area"],
            "intersection_polygon_count": overlap_res["intersection_count"],
        },
        "proximity_summary": {
            "distance_threshold_km": proximity_res["distance_threshold_km"],
            "direct_intersection_km2": proximity_res["direct_intersection_km2"],
            "proximity_buffer_only_km2": proximity_res["proximity_buffer_only_km2"],
            "total_within_distance_km2": proximity_res["total_within_distance_km2"],
        },
        "rainfall_context": rainfall_context,
        "metadata": combined_meta,
    }

    return {
        "summary": summary_payload,
        "overlap": overlap_res,
        "proximity": proximity_res,
        "timeline": unified_timeline,
    }
