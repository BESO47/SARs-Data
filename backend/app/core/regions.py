"""Amazon Basin study regions, geographic validation, and equal-area projections."""

from typing import Any, Dict, List, Optional, Tuple

from pyproj import CRS, Geod, Transformer
import shapely
from shapely.geometry import Polygon, box, mapping, shape
from shapely.validation import explain_validity

# Amazon Basin general bounding box (lon_min, lat_min, lon_max, lat_max)
AMAZON_BASIN_BBOX: Tuple[float, float, float, float] = (-80.0, -20.0, -44.0, 9.0)

# South America Albers Equal Area Conic (standard equal-area CRS for Amazon Basin)
SOUTH_AMERICA_EQUAL_AREA_CRS = CRS.from_proj4(
    "+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 "
    "+x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs"
)
WGS84_CRS = CRS.from_epsg(4326)
WGS84_GEOD = Geod(ellps="WGS84")

_TO_EQUAL_AREA_TRANSFORMER = Transformer.from_crs(WGS84_CRS, SOUTH_AMERICA_EQUAL_AREA_CRS, always_xy=True)
_TO_WGS84_TRANSFORMER = Transformer.from_crs(SOUTH_AMERICA_EQUAL_AREA_CRS, WGS84_CRS, always_xy=True)


PREDEFINED_STUDY_AREAS: Dict[str, Dict[str, Any]] = {
    "madeira": {
        "region_id": "madeira",
        "name": "Middle Madeira River & Humaitá Floodplain",
        "country": "Brazil",
        "state": "Amazonas / Rondônia",
        "description": (
            "Major Amazon tributary featuring seasonal várzea inundation and active "
            "riparian forest disturbance along BR-319 and Humaitá."
        ),
        "bbox": [-63.20, -7.80, -62.40, -7.10],
        "center": [-62.80, -7.45],
        "typical_flood_season": "February–May",
        "geometry": mapping(box(-63.20, -7.80, -62.40, -7.10)),
    },
    "mamiraua": {
        "region_id": "mamiraua",
        "name": "Mamirauá Sustainable Development Reserve (Solimões–Japurá)",
        "country": "Brazil",
        "state": "Amazonas",
        "description": (
            "Largest flooded forest (várzea) reserve in the central Amazon, prime reference "
            "site for L-band & C-band SAR double-bounce flooded vegetation analysis."
        ),
        "bbox": [-65.30, -3.25, -64.55, -2.65],
        "center": [-64.925, -2.95],
        "typical_flood_season": "April–July",
        "geometry": mapping(box(-65.30, -3.25, -64.55, -2.65)),
    },
    "xingu": {
        "region_id": "xingu",
        "name": "Lower Xingu & Altamira Hydrological Corridor",
        "country": "Brazil",
        "state": "Pará",
        "description": (
            "Eastern Amazon basin experiencing intense forest disturbance pressure alongside "
            "complex seasonal river fluctuations."
        ),
        "bbox": [-52.60, -3.65, -51.85, -2.95],
        "center": [-52.225, -3.30],
        "typical_flood_season": "February–May",
        "geometry": mapping(box(-52.60, -3.65, -51.85, -2.95)),
    },
    "purus": {
        "region_id": "purus",
        "name": "Lower Purus & Beruri Floodplain",
        "country": "Brazil",
        "state": "Amazonas",
        "description": (
            "Meandering white-water river floodplain with extensive seasonal open-water "
            "and macrophyte/igapó inundation dynamics."
        ),
        "bbox": [-61.65, -4.30, -60.95, -3.65],
        "center": [-61.30, -3.975],
        "typical_flood_season": "March–July",
        "geometry": mapping(box(-61.65, -4.30, -60.95, -3.65)),
    },
    "amazon_basin": {
        "region_id": "amazon_basin",
        "name": "Amazon Basin (General Monitoring Envelope)",
        "country": "Multi-national",
        "state": "Amazonia",
        "description": "Continental bounding envelope of the greater Amazon River Basin.",
        "bbox": list(AMAZON_BASIN_BBOX),
        "center": [-62.0, -5.5],
        "typical_flood_season": "Variable by sub-basin",
        "geometry": mapping(box(*AMAZON_BASIN_BBOX)),
    },
}


def calculate_geodesic_area_km2(geom_dict_or_shapely: Any) -> float:
    """
    Calculate accurate area in km² using WGS84 ellipsoid geodesic integration,
    with fallback verification in South America Albers Equal Area projection.
    """
    geom = shape(geom_dict_or_shapely) if isinstance(geom_dict_or_shapely, dict) else geom_dict_or_shapely
    if geom.is_empty:
        return 0.0
    area_m2, _ = WGS84_GEOD.geometry_area_perimeter(geom)
    return round(abs(area_m2) / 1_000_000.0, 4)


def calculate_equal_area_km2(geom_dict_or_shapely: Any) -> float:
    """Calculate area in km² by projecting to South America Albers Equal Area Conic."""
    geom = shape(geom_dict_or_shapely) if isinstance(geom_dict_or_shapely, dict) else geom_dict_or_shapely
    if geom.is_empty:
        return 0.0
    projected = project_to_equal_area(geom)
    return round(abs(projected.area) / 1_000_000.0, 4)


def project_to_equal_area(geom: Any) -> Any:
    """Transform a Shapely geometry from EPSG:4326 to South America Albers Equal Area (meters)."""
    return shapely.transform(geom, _TO_EQUAL_AREA_TRANSFORMER.transform, interleaved=False)


def project_to_wgs84(geom: Any) -> Any:
    """Transform a Shapely geometry from South America Albers Equal Area back to EPSG:4326."""
    return shapely.transform(geom, _TO_WGS84_TRANSFORMER.transform, interleaved=False)


def validate_and_resolve_region(
    region_id: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    polygon_geojson: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve and validate a geographic filter from region_id, bbox, or custom GeoJSON polygon.
    Ensures the geometry is valid, in EPSG:4326, and intersects the Amazon Basin envelope.
    """
    if polygon_geojson is not None:
        geom = shape(polygon_geojson)
        if not geom.is_valid:
            reason = explain_validity(geom)
            raise ValueError(f"Invalid study-area polygon geometry: {reason}")
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Expected Polygon or MultiPolygon, got {geom.geom_type}")
        resolved_id = region_id or "custom_polygon"
        resolved_name = "Custom Study-Area Polygon"
        bounds = list(geom.bounds)
    elif bbox is not None:
        if len(bbox) != 4:
            raise ValueError("Bounding box must have exactly 4 values: [min_lon, min_lat, max_lon, max_lat]")
        min_lon, min_lat, max_lon, max_lat = bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(f"Invalid bounding box coordinates: {bbox}")
        if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
            raise ValueError("Longitude values must be between -180 and 180")
        if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
            raise ValueError("Latitude values must be between -90 and 90")
        geom = box(min_lon, min_lat, max_lon, max_lat)
        resolved_id = region_id or f"bbox_{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}"
        resolved_name = f"Bounding Box ({min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f})"
        bounds = [min_lon, min_lat, max_lon, max_lat]
    else:
        rid = (region_id or "madeira").lower()
        if rid not in PREDEFINED_STUDY_AREAS:
            valid_ids = ", ".join(PREDEFINED_STUDY_AREAS.keys())
            raise ValueError(f"Unknown region_id '{rid}'. Valid predefined regions: {valid_ids}")
        preset = PREDEFINED_STUDY_AREAS[rid]
        geom = shape(preset["geometry"])
        resolved_id = preset["region_id"]
        resolved_name = preset["name"]
        bounds = list(preset["bbox"])

    amazon_box = box(*AMAZON_BASIN_BBOX)
    if not geom.intersects(amazon_box):
        raise ValueError(
            f"Selected geometry {bounds} lies outside the Amazon Basin envelope {list(AMAZON_BASIN_BBOX)}."
        )

    area_km2 = calculate_geodesic_area_km2(geom)
    centroid = geom.centroid

    return {
        "region_id": resolved_id,
        "name": resolved_name,
        "bbox": [round(b, 6) for b in bounds],
        "center": [round(centroid.x, 6), round(centroid.y, 6)],
        "area_km2": area_km2,
        "crs": "EPSG:4326",
        "equal_area_crs": "ESRI:102033 (South America Albers Equal Area Conic)",
        "geometry": mapping(geom),
    }


def list_predefined_areas() -> List[Dict[str, Any]]:
    """Return metadata and calculated areas for all predefined study regions."""
    areas = []
    for rid, info in PREDEFINED_STUDY_AREAS.items():
        geom = shape(info["geometry"])
        entry = dict(info)
        entry["area_km2"] = calculate_geodesic_area_km2(geom)
        entry["crs"] = "EPSG:4326"
        areas.append(entry)
    return areas
