import shapely
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from loguru import logger
from blocksnet.utils.validation import GdfSchema, ensure_crs


class BoundariesSchema(GdfSchema):
    """Schema validating boundary polygons for block cutting."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon, shapely.MultiPolygon}


class LineObjectsSchema(GdfSchema):
    """Schema validating line obstacles involved in block cutting."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.LineString, shapely.MultiLineString}


class PolygonObjectsSchema(GdfSchema):
    """Schema validating polygon obstacles used in block cutting."""

    @classmethod
    def _geometry_types(cls):
        return {shapely.Polygon, shapely.MultiPolygon}


class BuildingsSchema(GdfSchema):
    """Schema validating building geometries for block splitting."""

    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        return {shapely.Point}

    @classmethod
    def _before_validate(cls, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if "geometry" in gdf.columns:
            gdf.geometry = gdf.geometry.representative_point()
        return gdf


class BlocksSchema(GdfSchema):
    """Schema describing blocks produced by the cutting workflow."""

    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        return {shapely.Polygon}


def validate_and_preprocess_gdfs(
    boundaries_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame | None,
    polygons_gdf: gpd.GeoDataFrame | None,
    buildings_gdf: gpd.GeoDataFrame | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Validate and harmonise inputs for the block cutting pipeline.

    Parameters
    ----------
    boundaries_gdf : geopandas.GeoDataFrame
        Territory boundaries to partition.
    lines_gdf, polygons_gdf, buildings_gdf : geopandas.GeoDataFrame or None
        Optional datasets of linear, polygonal and point obstacles. Missing or
        empty inputs are replaced with empty GeoDataFrames in the same CRS.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame, geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        Validated GeoDataFrames for boundaries, line objects, polygon objects
        and buildings respectively.
    """

    logger.info("Checking boundaries schema")
    boundaries_gdf = BoundariesSchema(boundaries_gdf)
    crs = boundaries_gdf.crs

    logger.info("Checking line objects schema")
    if lines_gdf is None or lines_gdf.empty:
        logger.warning("Creating empty line objects")
        lines_gdf = LineObjectsSchema.create_empty(crs)
    else:
        lines_gdf = LineObjectsSchema(lines_gdf, allow_empty=True).explode("geometry", ignore_index=True)

    logger.info("Checking polygon objects schema")
    if polygons_gdf is None or polygons_gdf.empty:
        logger.warning("Creating empty polygon objects")
        polygons_gdf = PolygonObjectsSchema.create_empty(crs)
    else:
        polygons_gdf = PolygonObjectsSchema(polygons_gdf, allow_empty=True).explode("geometry", ignore_index=True)

    logger.info("Checking buildings schema")
    if buildings_gdf is None or buildings_gdf.empty:
        logger.warning("Creating empty buildings")
        buildings_gdf = BuildingsSchema.create_empty(crs)
    else:
        buildings_gdf = BuildingsSchema(buildings_gdf, allow_empty=True)

    ensure_crs(boundaries_gdf, lines_gdf, polygons_gdf, buildings_gdf)

    return boundaries_gdf, lines_gdf, polygons_gdf, buildings_gdf
