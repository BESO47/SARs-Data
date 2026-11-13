"""Dedicated quality-control and scientific validation system (Section 14)."""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import mapping, shape
from shapely.validation import explain_validity, make_valid

from app.models.layers import GeoJSONFeature
from app.models.metadata import DataStatus, QualityLevel, QualityMetadata


def validate_date_range(
    start_date_str: str,
    end_date_str: str,
    allow_single_day: bool = True,
) -> Tuple[date, date, int]:
    """
    Validate ISO 8601 date strings (YYYY-MM-DD), ensure start <= end,
    and return (start_date, end_date, duration_days).
    """
    try:
        start_dt = datetime.strptime(start_date_str[:10], "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date_str[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date format ({start_date_str}, {end_date_str}). Expected YYYY-MM-DD.") from exc

    if start_dt > end_dt:
        raise ValueError(f"start_date ({start_date_str}) must not be after end_date ({end_date_str}).")
    duration = (end_dt - start_dt).days
    if not allow_single_day and duration < 1:
        raise ValueError("Observation period must span at least 1 day.")
    return start_dt, end_dt, duration


def validate_and_deduplicate_features(
    features: List[GeoJSONFeature],
) -> Tuple[List[GeoJSONFeature], List[str]]:
    """
    Validate GeoJSON features:
    - Repair or discard invalid Shapely geometries.
    - Verify coordinate bounds are valid WGS84 (EPSG:4326).
    - Deduplicate features by feature_id and geometry WKT signature.
    """
    cleaned: List[GeoJSONFeature] = []
    warnings: List[str] = []
    seen_ids: set[str] = set()
    seen_wkt: set[Tuple[str, str]] = set()

    for feat in features:
        fid = feat.properties.feature_id
        if fid in seen_ids:
            warnings.append(f"Removed duplicate feature_id '{fid}'.")
            continue

        try:
            geom = shape(feat.geometry)
        except Exception as exc:
            warnings.append(f"Dropped feature '{fid}' with unparseable geometry: {exc}")
            continue

        if geom.is_empty:
            warnings.append(f"Dropped empty geometry for feature '{fid}'.")
            continue

        if not geom.is_valid:
            reason = explain_validity(geom)
            geom = make_valid(geom)
            warnings.append(f"Repaired invalid geometry for feature '{fid}' ({reason}).")

        minx, miny, maxx, maxy = geom.bounds
        if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
            warnings.append(f"Dropped feature '{fid}' with out-of-range EPSG:4326 coordinates: {geom.bounds}")
            continue

        sig = (feat.properties.classification, geom.wkt)
        if sig in seen_wkt:
            warnings.append(f"Removed duplicate geometry for feature '{fid}'.")
            continue

        seen_ids.add(fid)
        seen_wkt.add(sig)
        feat.geometry = mapping(geom)
        cleaned.append(feat)

    return cleaned, warnings


def validate_raster_array(
    arr: np.ndarray,
    valid_min: float,
    valid_max: float,
    parameter_name: str,
) -> Tuple[bool, List[str]]:
    """
    Check a raster array for unexpected values, NaN saturation, or empty coverage.
    """
    warnings: List[str] = []
    if arr.size == 0:
        return False, [f"Raster array for {parameter_name} is empty."]

    finite_mask = np.isfinite(arr)
    finite_ratio = float(np.sum(finite_mask)) / float(arr.size)
    if finite_ratio == 0.0:
        return False, [f"Raster array for {parameter_name} contains 100% NaN/nodata values."]

    if finite_ratio < 0.5:
        warnings.append(
            f"Spatial coverage warning: only {finite_ratio * 100:.1f}% of pixels in {parameter_name} contain valid values."
        )

    finite_vals = arr[finite_mask]
    actual_min = float(np.min(finite_vals))
    actual_max = float(np.max(finite_vals))
    if actual_min < valid_min or actual_max > valid_max:
        warnings.append(
            f"Unexpected raster value range in {parameter_name}: [{actual_min:.3f}, {actual_max:.3f}] "
            f"outside expected physical bounds [{valid_min}, {valid_max}]."
        )

    return True, warnings


def assess_observation_quality(
    observation_count: int,
    min_required_observations: int = 1,
    spatial_coverage_pct: Optional[float] = None,
    temporal_gap_days: Optional[int] = None,
    consistent_orbit_geometry: Optional[bool] = None,
    base_limitations: Optional[List[str]] = None,
    extra_warnings: Optional[List[str]] = None,
    confidence_method: Optional[str] = None,
) -> Tuple[DataStatus, QualityMetadata]:
    """
    Evaluate scientific quality and determine DataStatus and QualityMetadata.
    Never fabricates a confidence score.
    """
    warnings = list(extra_warnings or [])
    limitations = list(base_limitations or [])

    if observation_count < min_required_observations:
        warnings.append(
            f"Insufficient observations ({observation_count} found, minimum {min_required_observations} required)."
        )
        return DataStatus.INSUFFICIENT_OBSERVATIONS, QualityMetadata(
            level=QualityLevel.INSUFFICIENT,
            observation_count=observation_count,
            spatial_coverage_pct=spatial_coverage_pct,
            temporal_gap_days=temporal_gap_days,
            consistent_orbit_geometry=consistent_orbit_geometry,
            confidence_method=None,
            warnings=warnings,
            limitations=limitations,
        )

    score_penalties = 0
    if spatial_coverage_pct is not None and spatial_coverage_pct < 75.0:
        score_penalties += 1
        warnings.append(f"Partial spatial coverage ({spatial_coverage_pct:.1f}%) over study region.")

    if temporal_gap_days is not None and temporal_gap_days > 16:
        score_penalties += 1
        warnings.append(f"Large temporal gap ({temporal_gap_days} days) between satellite acquisitions.")

    if consistent_orbit_geometry is False:
        score_penalties += 1
        warnings.append(
            "Mixed ascending/descending or relative orbit geometries detected; incidence angle differences increase uncertainty."
        )

    if observation_count == 1:
        score_penalties += 1
        warnings.append("Single acquisition in window; speckle and transient effects cannot be temporally averaged.")

    if score_penalties == 0:
        level = QualityLevel.HIGH
    elif score_penalties == 1:
        level = QualityLevel.MEDIUM
    else:
        level = QualityLevel.LOW

    return DataStatus.AVAILABLE, QualityMetadata(
        level=level,
        observation_count=observation_count,
        spatial_coverage_pct=spatial_coverage_pct,
        temporal_gap_days=temporal_gap_days,
        consistent_orbit_geometry=consistent_orbit_geometry,
        confidence_method=confidence_method,
        warnings=warnings,
        limitations=limitations,
    )
