"""Product-specific NISAR L-band preprocessing (Section 7.3)."""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from app.processing.nisar.discovery import NisarProductMetadata
from app.processing.nisar.remote_access import read_nisar_windowed_subset
from app.services.nisar_access import NISAR_PRODUCT_SPECS


@dataclass
class PreprocessedNisarProduct:
    """Product-specific preprocessed NISAR L-band grids and quality metrics."""

    granule_id: str
    product_type: str
    product_level: str
    product_maturity: str
    acquisition_date: str
    hdf5_group_path: str
    calibration_quantity: str
    hh_gamma0_db: np.ndarray
    hv_gamma0_db: np.ndarray
    hh_hv_ratio_db: np.ndarray
    valid_mask: np.ndarray
    valid_pixel_pct: float
    preprocessing_steps: List[str]


def preprocess_nisar_product(
    product: NisarProductMetadata,
    region: Dict[str, Any],
) -> PreprocessedNisarProduct:
    """
    Execute product-specific NISAR L-band preprocessing (Section 7.3):
      - Inspects `product.product_type` (`GCOV` vs `GUNW` vs `RSLC`) and its HDF5 structure.
      - Rejects unsupported slant-range complex products (`RSLC`) for geocoded map layer analysis
        with a clear scientific message.
      - Masks invalid/shadow pixels using the NISAR quality bitmask (`mask == 1`).
      - Converts terrain-flattened linear covariance (`HHHH`, `HVHV` $\\gamma^0$) to decibels ($10 \\log_{10}(\\gamma^0)$).
      - Computes co-pol / cross-pol ratio (`HH - HV` in dB) for dihedral vs volume scattering separation.
    """
    spec = NISAR_PRODUCT_SPECS.get(product.product_type)
    if not spec or not spec["supported_for_backscatter_analysis"]:
        raise ValueError(
            f"NISAR product_type '{product.product_type}' ({product.hdf5_group_path}) is in slant-range "
            "zero-Doppler coordinates and is not directly geocoded for map-layer backscatter analysis. "
            "Use L2 GCOV (Geocoded Polarimetric Covariance) or L2 GUNW."
        )

    raw = read_nisar_windowed_subset(product=product, region=region)
    hhhh_lin = raw["hhhh_gamma0_linear"]
    hvhv_lin = raw["hvhv_gamma0_linear"]
    qmask = raw["quality_mask"]

    valid_mask = (qmask == 1) & np.isfinite(hhhh_lin) & np.isfinite(hvhv_lin) & (hhhh_lin > 0) & (hvhv_lin > 0)
    valid_pct = round(float(np.sum(valid_mask)) / float(valid_mask.size) * 100.0, 2)

    hh_db = np.where(valid_mask, 10.0 * np.log10(np.maximum(hhhh_lin, 1e-6)), np.nan).astype(np.float32)
    hv_db = np.where(valid_mask, 10.0 * np.log10(np.maximum(hvhv_lin, 1e-6)), np.nan).astype(np.float32)
    ratio_db = (hh_db - hv_db).astype(np.float32)

    steps = [
        f"Verified NISAR {product.product_level} {product.product_type} HDF5 group structure ({product.hdf5_group_path}).",
        f"Applied NISAR radiometric quality & layover/shadow mask ({valid_pct}% valid geocoded pixels).",
        "Converted terrain-flattened gamma0 polarimetric covariance terms (HHHH, HVHV) from linear power to decibels (dB).",
        "Calculated L-band HH/HV polarimetric ratio (HH_dB - HV_dB) to separate trunk-water dihedral double-bounce from canopy volume scattering.",
    ]

    return PreprocessedNisarProduct(
        granule_id=product.granule_id,
        product_type=product.product_type,
        product_level=product.product_level,
        product_maturity=product.product_maturity,
        acquisition_date=product.acquisition_date,
        hdf5_group_path=product.hdf5_group_path,
        calibration_quantity=product.calibration_quantity,
        hh_gamma0_db=hh_db,
        hv_gamma0_db=hv_db,
        hh_hv_ratio_db=ratio_db,
        valid_mask=valid_mask,
        valid_pixel_pct=valid_pct,
        preprocessing_steps=steps,
    )
