"""Artifact storage and dynamic XYZ PNG map-tile generator for CesiumJS (Section 11, 13, 15)."""

import io
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from app.core.config import get_settings

# RGBA color palettes for each dataset's XYZ map tiles
FLOOD_RGBA_MAP: Dict[int, Tuple[int, int, int, int]] = {
    0: (0, 0, 0, 0),          # not_detected -> transparent
    1: (0, 229, 255, 215),    # open_water_inundation -> cyan
    2: (0, 230, 118, 210),    # possible_flooded_vegetation -> vibrant green
    3: (21, 101, 192, 220),   # permanent_water -> deep blue
    4: (255, 179, 0, 190),    # uncertain -> amber
}


def render_classification_png_tile(
    grid: np.ndarray,
    tile_size: int = 256,
) -> bytes:
    """
    Render a 2D classification grid (codes 0..4) into a 256x256 RGBA PNG byte buffer
    suitable for CesiumJS `UrlTemplateImageryProvider`.
    """
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for code, color in FLOOD_RGBA_MAP.items():
        mask = grid == code
        rgba[mask] = color

    img = Image.fromarray(rgba, mode="RGBA")
    if (w, h) != (tile_size, tile_size):
        img = img.resize((tile_size, tile_size), resample=Image.Resampling.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_continuous_png_tile(
    grid: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: str = "rainfall",
    tile_size: int = 256,
) -> bytes:
    """
    Render a continuous 2D scientific grid (e.g., GPM rainfall mm or NISAR L-band dB)
    into a 256x256 RGBA PNG tile for CesiumJS.
    """
    h, w = grid.shape
    valid = np.isfinite(grid)
    safe_grid = np.where(valid, grid, vmin)
    norm = np.clip((safe_grid - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if colormap == "nisar":
        # Purple-cyan-emerald gradient for L-band gamma0 / dihedral enhancement
        rgba[..., 0] = np.where(valid, (40 + 180 * norm).astype(np.uint8), 0)
        rgba[..., 1] = np.where(valid, (180 - 90 * np.abs(norm - 0.5)).astype(np.uint8), 0)
        rgba[..., 2] = np.where(valid, (120 + 135 * norm).astype(np.uint8), 0)
        rgba[..., 3] = np.where(valid, 200, 0)
    else:
        # Cyan-blue-indigo rainfall intensity gradient
        rgba[..., 0] = np.where(valid, (224 * (1.0 - norm) + 26 * norm).astype(np.uint8), 0)
        rgba[..., 1] = np.where(valid, (247 * (1.0 - norm) + 35 * norm).astype(np.uint8), 0)
        rgba[..., 2] = np.where(valid, (250 * (1.0 - norm * 0.4)).astype(np.uint8), 0)
        rgba[..., 3] = np.where(valid & (norm > 0.02), 195, 0)

    img = Image.fromarray(rgba, mode="RGBA")
    if (w, h) != (tile_size, tile_size):
        img = img.resize((tile_size, tile_size), resample=Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def save_processed_geojson(name: str, payload: Dict[str, Any]) -> Path:
    """Persist processed GeoJSON output to storage directory without duplicating raw satellite scenes."""
    import json

    settings = get_settings()
    out_dir = settings.storage_dir / "geojson"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.geojson"
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path
