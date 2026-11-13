"""Models for CesiumJS-compatible layers, GeoJSON FeatureCollections, and legends (Section 10.2 & 13)."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.metadata import DataStatus, DatasetMetadata, QualityMetadata


class LegendItem(BaseModel):
    """Single classification entry in a map layer legend."""

    code: int | str
    label: str
    color_hex: str
    description: str


class LayerLegend(BaseModel):
    """Legend definition for CesiumJS and React UI rendering."""

    title: str
    unit: Optional[str] = None
    items: List[LegendItem] = Field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    palette: Optional[List[str]] = None


class GeoJSONFeatureProperties(BaseModel):
    """Standardized GeoJSON feature properties (Section 10.2)."""

    feature_id: str
    dataset: str
    classification: str
    observation_date: Optional[str] = None
    confidence: Optional[str] = Field(
        default=None,
        description="Source-provided or method-backed confidence level; never fabricated.",
    )
    area_km2: Optional[float] = None
    source: str
    quality_level: str = "medium"
    extra: Dict[str, Any] = Field(default_factory=dict)


class GeoJSONFeature(BaseModel):
    """Validated GeoJSON Feature."""

    type: Literal["Feature"] = "Feature"
    id: Optional[str] = None
    geometry: Dict[str, Any]
    properties: GeoJSONFeatureProperties


class GeoJSONFeatureCollection(BaseModel):
    """Validated GeoJSON FeatureCollection with embedded dataset metadata and pagination."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    metadata: DatasetMetadata
    total_features: int = 0
    page: int = 1
    page_size: int = 500
    bbox: Optional[List[float]] = None
    features: List[GeoJSONFeature] = Field(default_factory=list)


class CesiumLayerResponse(BaseModel):
    """
    CesiumJS-compatible layer response (Section 13).
    Supports XYZ raster tiles, GEE map tiles, COG tile endpoints, and GeoJSON vector layers.
    """

    layer_id: str
    dataset: str
    title: str
    source: str
    layer_type: Literal["xyz_tiles", "geojson", "cog_tiles", "hybrid"]
    observation_start: Optional[str] = None
    observation_end: Optional[str] = None
    tile_url_template: Optional[str] = Field(
        default=None,
        description="XYZ tile template URL ({z}/{x}/{y}) with no private credentials exposed.",
    )
    feature_endpoint: Optional[str] = Field(
        default=None,
        description="GeoJSON or vector feature endpoint URL for Cesium GeoJsonDataSource.",
    )
    geojson: Optional[GeoJSONFeatureCollection] = Field(
        default=None,
        description="Inline GeoJSON FeatureCollection when manageable in size.",
    )
    bbox: List[float]
    crs: str = "EPSG:4326"
    default_opacity: float = Field(default=0.75, ge=0.0, le=1.0)
    z_index: int = Field(default=10)
    legend: LayerLegend
    status: DataStatus
    quality: QualityMetadata
    metadata: DatasetMetadata
