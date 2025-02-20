import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from pandera import dataframe_parser
from loguru import logger
from shapely.ops import polygonize
from .schemas import BoundariesSchema, LineObjectsSchema, PolygonObjectsSchema


def _validate_and_prepare_geometries(
    boundaries_gdf: gpd.GeoDataFrame, lines_gdf: gpd.GeoDataFrame | None, polygons_gdf: gpd.GeoDataFrame | None
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    logger.info("Checking boundaries schema")
    boundaries_gdf = BoundariesSchema(boundaries_gdf)
    crs = boundaries_gdf.crs

    logger.info("Checking line objects schema")
    if lines_gdf is None:
        lines_gdf = LineObjectsSchema.create_empty().to_crs(crs)
    else:
        lines_gdf = LineObjectsSchema(lines_gdf)

    logger.info("Checking polygon objects schema")
    if polygons_gdf is None:
        polygons_gdf = PolygonObjectsSchema.create_empty().to_crs(crs)
    else:
        polygons_gdf = PolygonObjectsSchema(polygons_gdf)

    for gdf in [lines_gdf, polygons_gdf]:
        assert gdf.crs == crs, "CRS must match for all geodataframes"

    return boundaries_gdf, lines_gdf, polygons_gdf


def _exclude_polygons(boundaries_gdf: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame):
    logger.info("Excluding polygon objects from blocks")
    polygons_gdf = gpd.GeoDataFrame(geometry=[polygons_gdf.unary_union], crs=polygons_gdf.crs)
    return boundaries_gdf.overlay(polygons_gdf, how="difference")


def _get_enclosures(boundaries_gdf: gpd.GeoDataFrame, lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Setting up enclosures")
    barriers = (
        pd.concat([lines_gdf.geometry, boundaries_gdf.boundary]).explode(ignore_index=True).reset_index(drop=True)
    )

    unioned = barriers.unary_union
    polygons = polygonize(unioned)
    # return gpd.GeoDataFrame(geometry=list(polygons), crs=boundaries_gdf.crs)
    enclosures = gpd.GeoSeries(list(polygons), crs=lines_gdf.crs)
    _, enclosure_idxs = enclosures.representative_point().sindex.query(boundaries_gdf.geometry, predicate="contains")
    enclosures = enclosures.iloc[np.unique(enclosure_idxs)]
    enclosures = enclosures.rename("geometry").reset_index()

    return enclosures


def _fill_holes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Filling holes inside the blocks")
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.boundary
    gdf = gdf.explode(index_parts=False)
    gdf.geometry = gdf.geometry.map(
        lambda g: shapely.Polygon(g) if not isinstance(g.geom_type, shapely.Point) else np.nan
    )
    gdf = gdf.dropna(subset="geometry").reset_index(drop=True)
    return gdf


def _filter_overlapping(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Filtering overlapping geometries")
    gdf = gdf.copy()
    overlaps = gdf.geometry.sindex.query(gdf.geometry, predicate="contains")
    contains_dict = {x: [] for x in overlaps[0]}
    for x, y in zip(overlaps[0], overlaps[1]):
        if x != y:
            contains_dict[x].append(y)
    contained_geoms_idxs = list({x for v in contains_dict.values() for x in v})
    gdf = gdf.drop(contained_geoms_idxs).reset_index(drop=True)
    return gdf


def _filter_bottlenecks(gdf: gpd.GeoDataFrame, min_block_width: float):
    logger.info("Filtering bottlenecks and small blocks")

    def _filter_bottlenecks_helper(poly):
        try:
            return poly.intersection(poly.buffer(-min_block_width / 2).buffer(min_block_width / 2, join_style=2))
        except:
            return poly

    gdf.geometry = gdf["geometry"].apply(_filter_bottlenecks_helper)
    gdf = gdf[gdf["geometry"] != shapely.Polygon()]
    gdf = gdf.explode(ignore_index=True)
    gdf = gdf[gdf.type == "Polygon"]

    return gdf


def cut_urban_blocks(
    boundaries_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame | None,
    polygons_gdf: gpd.GeoDataFrame | None,
    fill_holes: bool = True,
    min_block_width: float | None = None,
):
    boundaries_gdf, lines_gdf, polygons_gdf = _validate_and_prepare_geometries(boundaries_gdf, lines_gdf, polygons_gdf)
    boundaries_gdf = _exclude_polygons(boundaries_gdf, polygons_gdf)
    blocks_gdf = _get_enclosures(boundaries_gdf, lines_gdf)
    if fill_holes:
        blocks_gdf = _fill_holes(blocks_gdf)

    blocks_gdf = _filter_overlapping(blocks_gdf)
    if min_block_width is not None:
        blocks_gdf = _filter_bottlenecks(blocks_gdf, min_block_width)
    logger.success("Blocks are successfully cut!")
    return blocks_gdf.reset_index(drop=True)[["geometry"]].copy()
