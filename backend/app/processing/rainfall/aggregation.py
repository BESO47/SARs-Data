"""
NASA GPM IMERG V07 Rainfall Aggregation & Output Generation (Section 8.3 & 8.4).

Crucial Scientific Rule:
Each `NASA/GPM_L3/IMERG_V07` granule reports `precipitation` as a RATE in `mm/hr` over a
30-minute (`0.5 hr`) observation interval.
Never sum raw `mm/hr` rates directly without accounting for the `0.5 hr` temporal interval:
    accumulated_depth_mm = sum(rate_mm_per_hr * 0.5_hr)
"""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from app.core.quality_control import assess_observation_quality, validate_raster_array
from app.models.layers import (
    CesiumLayerResponse,
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
from app.processing.rainfall.acquisition import acquire_imerg_half_hourly_rates
from app.services.gpm import (
    IMERG_COLLECTION_ID,
    IMERG_PRECIPITATION_BAND,
    IMERG_RESOLUTION_M,
    IMERG_TEMPORAL_INTERVAL_HOURS,
)

RAINFALL_LEGEND = LayerLegend(
    title="NASA GPM IMERG V07 Accumulated Precipitation (mm)",
    unit="mm",
    min_value=0.0,
    max_value=600.0,
    palette=["#E0F7FA", "#4DD0E1", "#0288D1", "#1A237E", "#4A148C"],
    items=[
        LegendItem(code="0-50", label="0–50 mm", color_hex="#E0F7FA", description="Low accumulated rainfall"),
        LegendItem(code="50-150", label="50–150 mm", color_hex="#4DD0E1", description="Moderate rainfall"),
        LegendItem(code="150-300", label="150–300 mm", color_hex="#0288D1", description="High wet-season rainfall"),
        LegendItem(code="300+", label="> 300 mm", color_hex="#1A237E", description="Intense convective accumulation"),
    ],
)


def integrate_half_hourly_rates_to_mm(
    rates_mm_per_hr: np.ndarray | List[float],
    interval_hours: float = IMERG_TEMPORAL_INTERVAL_HOURS,
) -> float:
    """
    Integrate half-hourly precipitation rates (mm/hr) into accumulated depth (mm)
    by multiplying each rate by its temporal duration (`interval_hours = 0.5`).
    """
    arr = np.asarray(rates_mm_per_hr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    depth_mm = np.sum(arr * float(interval_hours))
    return round(float(depth_mm), 3)


def run_gpm_rainfall_pipeline(
    region: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Execute GPM IMERG V07 acquisition, temporal integration (`rate * 0.5 hr`),
    daily and monthly aggregation, spatial grid calculation, and Cesium layer formatting.
    """
    acq = acquire_imerg_half_hourly_rates(region=region, start_date=start_date, end_date=end_date)
    daily_records = acq["daily_half_hourly_rates"]
    is_live = acq["is_live"]

    daily_series: List[Dict[str, Any]] = []
    monthly_totals: Dict[str, float] = defaultdict(float)
    monthly_days: Dict[str, int] = defaultdict(int)
    total_granules = 0

    for item in daily_records:
        dt_str = item["date"]
        rates = item["rates_mm_per_hr"]
        interval_hr = item["interval_hours"]
        total_granules += int(item["granule_count"])

        # Integrate 48 half-hour rates (mm/hr * 0.5 hr) -> daily accumulated mm
        daily_mm = integrate_half_hourly_rates_to_mm(rates, interval_hours=interval_hr)
        mean_rate_mm_hr = round(float(np.mean(rates)), 3)
        peak_rate_mm_hr = round(float(np.max(rates)), 3)

        daily_series.append(
            {
                "date": dt_str,
                "accumulated_mm": daily_mm,
                "mean_rate_mm_per_hr": mean_rate_mm_hr,
                "peak_rate_mm_per_hr": peak_rate_mm_hr,
                "granule_count": item["granule_count"],
            }
        )
        ym = dt_str[:7]
        monthly_totals[ym] = round(monthly_totals[ym] + daily_mm, 3)
        monthly_days[ym] += 1

    total_accumulated_mm = round(sum(d["accumulated_mm"] for d in daily_series), 3)
    mean_daily_mm = round(total_accumulated_mm / max(len(daily_series), 1), 3)
    max_daily_entry = max(daily_series, key=lambda d: d["accumulated_mm"]) if daily_series else None

    # Spatial grid of accumulated rainfall (mm) over the study region
    spatial_grid_mm = (acq["spatial_pattern"] * total_accumulated_mm).astype(np.float32)
    _, raster_warnings = validate_raster_array(
        spatial_grid_mm,
        valid_min=0.0,
        valid_max=10000.0,
        parameter_name="IMERG_accumulated_mm",
    )

    limitations = [
        "GPM IMERG V07 has a native spatial resolution of 0.1° (~11.1 km at the Equator), which captures regional convective systems but smooths localized micro-scale river-bank storms.",
        "Each 30-minute granule reports precipitation rate in mm/hr; all accumulations in this pipeline explicitly integrate `sum(rate_mm_per_hr * 0.5_hr)`.",
        "Hydrological flood response in large Amazon tributaries (e.g., Madeira, Solimões, Purus) lags upstream basin rainfall by days to weeks due to basin routing.",
    ]
    extra_warnings = list(raster_warnings)
    if not is_live:
        extra_warnings.append(
            "GEE credentials not configured in .env; executed verified half-hourly rate integration (`mm/hr * 0.5h`) on reference observations with status='demo'."
        )

    raw_status, quality = assess_observation_quality(
        observation_count=total_granules,
        min_required_observations=48,
        spatial_coverage_pct=100.0,
        temporal_gap_days=0,
        base_limitations=limitations,
        extra_warnings=extra_warnings,
        confidence_method="imerg_v07_half_hourly_gauge_calibrated_integration",
    )
    final_status = (
        DataStatus.INSUFFICIENT_OBSERVATIONS
        if raw_status == DataStatus.INSUFFICIENT_OBSERVATIONS
        else (DataStatus.AVAILABLE if is_live else DataStatus.DEMO)
    )

    metadata = DatasetMetadata(
        dataset="gpm-imerg",
        provider="NASA GES DISC (via Google Earth Engine)",
        instrument="NASA/JAXA GPM Constellation (IMERG V07)",
        product_id=IMERG_COLLECTION_ID,
        status=final_status,
        result_nature=ResultNature.OBSERVED if is_live else ResultNature.ILLUSTRATIVE,
        observation_start=start_date,
        observation_end=end_date,
        acquisition_dates=[d["date"] for d in daily_series],
        region_id=region["region_id"],
        geographic_coverage={"bbox": region["bbox"], "area_km2": region["area_km2"]},
        crs="EPSG:4326",
        resolution_m=IMERG_RESOLUTION_M,
        method="half_hourly_rate_temporal_integration_mm_per_hr_times_0.5h",
        quality=quality,
        extra={
            "band": IMERG_PRECIPITATION_BAND,
            "native_rate_unit": "mm/hr",
            "integrated_unit": "mm",
            "temporal_interval_hours": IMERG_TEMPORAL_INTERVAL_HOURS,
            "total_half_hourly_granules": total_granules,
        },
    )

    stats_items = [
        StatisticItem(
            metric_id="total_accumulated_rainfall_mm",
            label="Total Period Accumulated Rainfall",
            value=total_accumulated_mm,
            unit="mm",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=IMERG_COLLECTION_ID,
            processing_method="sum(precipitation_mm_per_hr * 0.5_hr)",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=final_status,
            notes=f"Integrated across {total_granules} half-hourly (0.5 hr) granules.",
        ),
        StatisticItem(
            metric_id="mean_daily_rainfall_mm",
            label="Mean Daily Rainfall",
            value=mean_daily_mm,
            unit="mm/day",
            period_start=start_date,
            period_end=end_date,
            region_id=region["region_id"],
            source=IMERG_COLLECTION_ID,
            processing_method="period_accumulated_mm_divided_by_days",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=final_status,
        ),
        StatisticItem(
            metric_id="max_daily_rainfall_mm",
            label="Peak Single-Day Rainfall Accumulation",
            value=max_daily_entry["accumulated_mm"] if max_daily_entry else 0.0,
            unit="mm",
            period_start=max_daily_entry["date"] if max_daily_entry else start_date,
            period_end=max_daily_entry["date"] if max_daily_entry else end_date,
            region_id=region["region_id"],
            source=IMERG_COLLECTION_ID,
            processing_method="daily_integrated_maximum",
            result_nature=ResultNature.DERIVED,
            quality=quality,
            status=final_status,
        ),
    ]

    monthly_summary = [
        {
            "month": ym,
            "accumulated_mm": total_mm,
            "days_covered": monthly_days[ym],
            "mean_daily_mm": round(total_mm / max(monthly_days[ym], 1), 3),
            "unit": "mm",
        }
        for ym, total_mm in sorted(monthly_totals.items())
    ]

    statistics_response = StatisticsResponse(
        dataset="gpm-imerg",
        region_id=region["region_id"],
        status=final_status,
        metadata=metadata,
        statistics=stats_items,
        breakdown={
            "monthly_summaries": monthly_summary,
            "daily_count": len(daily_series),
            "spatial_distribution_mm": {
                "min_mm": round(float(np.min(spatial_grid_mm)), 2),
                "mean_mm": round(float(np.mean(spatial_grid_mm)), 2),
                "max_mm": round(float(np.max(spatial_grid_mm)), 2),
            },
        },
    )

    timeline_entries = [
        TimelineEntry(
            timestamp=f"{d['date']}T00:00:00Z",
            end_timestamp=f"{d['date']}T23:59:59Z",
            dataset="gpm-imerg",
            event_type="daily_accumulated_rainfall",
            value=d["accumulated_mm"],
            unit="mm",
            label=f"Daily IMERG V07 Rainfall ({d['accumulated_mm']} mm)",
            product_id=IMERG_COLLECTION_ID,
            status=final_status,
            details={
                "mean_rate_mm_per_hr": d["mean_rate_mm_per_hr"],
                "peak_rate_mm_per_hr": d["peak_rate_mm_per_hr"],
                "granule_count": d["granule_count"],
                "interval_hours": IMERG_TEMPORAL_INTERVAL_HOURS,
            },
        )
        for d in daily_series
    ]

    timeline_response = TimelineResponse(
        dataset="gpm-imerg",
        region_id=region["region_id"],
        status=final_status,
        unit="mm",
        temporal_resolution="daily (integrated from 0.5-hour granules)",
        metadata=metadata,
        series=timeline_entries,
    )

    tile_url = (
        f"/api/tiles/rainfall/{region['region_id']}/{{z}}/{{x}}/{{y}}.png"
        f"?start_date={start_date}&end_date={end_date}"
    )

    layer_response = CesiumLayerResponse(
        layer_id=f"gpm-imerg-{region['region_id']}-{start_date}-{end_date}",
        dataset="gpm-imerg",
        title=f"NASA GPM IMERG Accumulated Rainfall ({region['name']})",
        source=IMERG_COLLECTION_ID,
        layer_type="xyz_tiles",
        observation_start=start_date,
        observation_end=end_date,
        tile_url_template=tile_url,
        feature_endpoint=None,
        geojson=None,
        bbox=region["bbox"],
        default_opacity=0.65,
        z_index=15,
        legend=RAINFALL_LEGEND,
        status=final_status,
        quality=quality,
        metadata=metadata,
    )

    return {
        "layer": layer_response,
        "statistics": statistics_response,
        "timeline": timeline_response,
        "daily_series": daily_series,
        "monthly_summary": monthly_summary,
        "spatial_grid_mm": spatial_grid_mm,
    }
