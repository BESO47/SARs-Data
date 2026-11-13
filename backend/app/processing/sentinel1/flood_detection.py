"""
Configurable Sentinel-1 SAR Flood Detection Algorithm (Section 5.4).

Classifies each pixel into one of five scientifically explicit categories without forcing
borderline or terrain-occluded pixels into a binary flood/no-flood decision:
  0 -> `not_detected`
  1 -> `open_water_inundation`
  2 -> `possible_flooded_vegetation`
  3 -> `permanent_water`
  4 -> `uncertain`
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from pydantic import BaseModel, Field

from app.processing.sentinel1.preprocessing import PreprocessedS1Observation

FLOOD_CLASSES: Dict[int, str] = {
    0: "not_detected",
    1: "open_water_inundation",
    2: "possible_flooded_vegetation",
    3: "permanent_water",
    4: "uncertain",
}

FLOOD_CLASS_CODES: Dict[str, int] = {v: k for k, v in FLOOD_CLASSES.items()}


class FloodDetectionConfig(BaseModel):
    """
    Documented, configurable parameters for dual-polarization SAR flood detection (Section 5.4).
    Supports separate tuning for open-water specular reduction and flooded-vegetation double-bounce.
    """

    open_water_vh_max_db: float = Field(
        default=-17.5,
        description="Maximum VH backscatter (dB) for specular open-water surface scattering.",
    )
    open_water_vv_max_db: float = Field(
        default=-12.0,
        description="Maximum VV backscatter (dB) for open-water inundation.",
    )
    open_water_delta_vh_db: float = Field(
        default=-2.8,
        description="Minimum VH backscatter drop (obs - baseline, in dB) required to classify seasonal open-water flood.",
    )
    open_water_delta_vv_db: float = Field(
        default=-2.5,
        description="Minimum VV backscatter drop (obs - baseline, in dB) required for open-water flood.",
    )
    flooded_veg_delta_vv_min_db: float = Field(
        default=1.4,
        description="Minimum VV backscatter increase (obs - baseline, in dB) indicating dihedral trunk-water double-bounce in flooded canopy.",
    )
    flooded_veg_vv_min_db: float = Field(
        default=-7.2,
        description="Minimum observation VV backscatter (dB) for flooded vegetation double-bounce.",
    )
    flooded_veg_max_vv_vh_ratio_db: float = Field(
        default=9.5,
        description="Maximum polarization difference (VV - VH, in dB) consistent with dihedral/canopy scattering.",
    )
    permanent_water_jrc_min_pct: float = Field(
        default=80.0,
        description="JRC Global Surface Water occurrence threshold (%) above which pixels are classified as permanent_water.",
    )
    max_slope_deg: float = Field(
        default=5.0,
        description="Maximum SRTM topographic slope (degrees) for plausible floodplain inundation; steeper pixels with low backscatter are flagged as uncertain (radar shadow).",
    )
    min_incidence_angle_deg: float = Field(
        default=30.0,
        description="Minimum radar incidence angle (degrees) before steep-angle foreshortening uncertainty.",
    )
    max_incidence_angle_deg: float = Field(
        default=45.0,
        description="Maximum radar incidence angle (degrees) before far-range grazing uncertainty.",
    )


@dataclass
class FloodClassificationResult:
    """Output of the Sentinel-1 flood classification algorithm."""

    classification_grid: np.ndarray  # uint8 codes 0..4
    delta_vv_db: np.ndarray
    delta_vh_db: np.ndarray
    pixel_counts: Dict[str, int]
    pixel_fractions: Dict[str, float]
    area_by_class_km2: Dict[str, float]
    total_inundated_km2: float  # open_water_inundation + possible_flooded_vegetation
    config_used: FloodDetectionConfig


def classify_sentinel1_flood(
    preprocessed: PreprocessedS1Observation,
    total_region_area_km2: float,
    config: FloodDetectionConfig | None = None,
) -> FloodClassificationResult:
    """
    Execute the multi-class Sentinel-1 flood detection algorithm on preprocessed grids.
    """
    cfg = config or FloodDetectionConfig()

    vv_obs = preprocessed.vv_obs_db
    vh_obs = preprocessed.vh_obs_db
    vv_base = preprocessed.vv_base_db
    vh_base = preprocessed.vh_base_db
    jrc = preprocessed.jrc_water_occurrence_pct
    slope = preprocessed.slope_deg
    inc = preprocessed.incidence_angle_deg
    canopy = preprocessed.canopy_mask

    delta_vv = (vv_obs - vv_base).astype(np.float32)
    delta_vh = (vh_obs - vh_base).astype(np.float32)
    pol_diff_vv_vh = (vv_obs - vh_obs).astype(np.float32)

    classes = np.zeros(vv_obs.shape, dtype=np.uint8)  # Default 0: not_detected

    # 1. Permanent water reference (JRC >= 80%)
    perm_mask = jrc >= cfg.permanent_water_jrc_min_pct
    classes[perm_mask] = FLOOD_CLASS_CODES["permanent_water"]

    non_perm = ~perm_mask
    flat_terrain = slope <= cfg.max_slope_deg
    valid_incidence = (inc >= cfg.min_incidence_angle_deg) & (inc <= cfg.max_incidence_angle_deg)

    # 2. Open water inundation (strong specular drop in both VH and VV on flat floodplain)
    open_water_mask = (
        non_perm
        & flat_terrain
        & valid_incidence
        & (vh_obs <= cfg.open_water_vh_max_db)
        & (vv_obs <= cfg.open_water_vv_max_db)
        & (delta_vh <= cfg.open_water_delta_vh_db)
        & (delta_vv <= cfg.open_water_delta_vv_db)
    )
    classes[open_water_mask] = FLOOD_CLASS_CODES["open_water_inundation"]

    # 3. Possible flooded vegetation (double-bounce VV enhancement under forest canopy on flat floodplain)
    flooded_veg_mask = (
        non_perm
        & (~open_water_mask)
        & flat_terrain
        & valid_incidence
        & canopy
        & (delta_vv >= cfg.flooded_veg_delta_vv_min_db)
        & (vv_obs >= cfg.flooded_veg_vv_min_db)
        & (pol_diff_vv_vh <= cfg.flooded_veg_max_vv_vh_ratio_db)
    )
    classes[flooded_veg_mask] = FLOOD_CLASS_CODES["possible_flooded_vegetation"]

    # 4. Uncertain pixels (do not force into definitive flood/no-flood):
    #    - Borderline backscatter drops (-2.8 < delta_vh <= -1.4 dB)
    #    - Low backscatter on steep slopes (> max_slope_deg -> radar shadow)
    #    - Out-of-bounds incidence angles
    remaining = non_perm & (~open_water_mask) & (~flooded_veg_mask)
    borderline_drop = (delta_vh <= -1.4) | (delta_vv <= -1.4)
    borderline_rise = canopy & (delta_vv >= 0.85) & (delta_vv < cfg.flooded_veg_delta_vv_min_db)
    steep_shadow = (~flat_terrain) & ((vh_obs <= cfg.open_water_vh_max_db) | (delta_vh <= -2.0))
    bad_inc = ~valid_incidence

    uncertain_mask = remaining & (borderline_drop | borderline_rise | steep_shadow | bad_inc)
    classes[uncertain_mask] = FLOOD_CLASS_CODES["uncertain"]

    total_pixels = float(classes.size)
    pixel_counts: Dict[str, int] = {}
    pixel_fractions: Dict[str, float] = {}
    area_by_class_km2: Dict[str, float] = {}

    for code, label in FLOOD_CLASSES.items():
        cnt = int(np.sum(classes == code))
        frac = cnt / total_pixels if total_pixels > 0 else 0.0
        pixel_counts[label] = cnt
        pixel_fractions[label] = round(frac, 6)
        area_by_class_km2[label] = round(frac * total_region_area_km2, 3)

    total_inundated = round(
        area_by_class_km2["open_water_inundation"] + area_by_class_km2["possible_flooded_vegetation"],
        3,
    )

    return FloodClassificationResult(
        classification_grid=classes,
        delta_vv_db=delta_vv,
        delta_vh_db=delta_vh,
        pixel_counts=pixel_counts,
        pixel_fractions=pixel_fractions,
        area_by_class_km2=area_by_class_km2,
        total_inundated_km2=total_inundated,
        config_used=cfg,
    )
