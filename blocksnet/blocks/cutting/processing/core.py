from functools import wraps
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from loguru import logger
from shapely.ops import polygonize
from .schemas import BoundariesSchema, LineObjectsSchema, PolygonObjectsSchema
from blocksnet.utils.validation import ensure_crs


def _validate_and_process_gdfs(func):
    @wraps(func)
    def wrapper(
        boundaries_gdf: gpd.GeoDataFrame,
        lines_gdf: gpd.GeoDataFrame | None,
        polygons_gdf: gpd.GeoDataFrame | None,
        *args,
        **kwargs,
    ):
        logger.info("Checking boundaries schema")
        boundaries_gdf = BoundariesSchema(boundaries_gdf)
        crs = boundaries_gdf.crs

        logger.info("Checking line objects schema")
        if lines_gdf is None:
            lines_gdf = LineObjectsSchema.create_empty(crs)
        else:
            lines_gdf = LineObjectsSchema(lines_gdf).explode("geometry", ignore_index=True)

        logger.info("Checking polygon objects schema")
        if polygons_gdf is None:
            polygons_gdf = PolygonObjectsSchema.create_empty(crs)
        else:
            polygons_gdf = PolygonObjectsSchema(polygons_gdf).explode("geometry", ignore_index=True)

        ensure_crs(boundaries_gdf, lines_gdf, polygons_gdf)

        return func(boundaries_gdf, lines_gdf, polygons_gdf, *args, **kwargs)

    return wrapper


def _exclude_polygons(boundaries_gdf: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame):
    logger.info("Excluding polygon objects from blocks")
    polygons_gdf = gpd.GeoDataFrame(geometry=[polygons_gdf.union_all()], crs=polygons_gdf.crs)
    return boundaries_gdf.overlay(polygons_gdf, how="difference")


def _get_enclosures(boundaries_gdf: gpd.GeoDataFrame, lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Setting up enclosures")
    barriers = (
        pd.concat([lines_gdf.geometry, boundaries_gdf.boundary]).explode(ignore_index=True).reset_index(drop=True)
    )

    unioned = barriers.union_all()
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
    gdf.geometry = gdf.geometry.map(lambda g: shapely.Polygon(g) if g.geom_type != "Point" else np.nan)
    gdf = gdf.dropna(subset="geometry").reset_index(drop=True)
    return gdf


def _filter_overlapping(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Filtering overlapping geometries")
    gdf = gdf.copy()
    overlaps = gdf.geometry.sindex.query(gdf.geometry, predicate="contains")
    contained_geoms_idxs = {y for x, y in zip(overlaps[0], overlaps[1]) if x != y}
    gdf = gdf.drop(list(contained_geoms_idxs)).reset_index(drop=True)
    return gdf


def _filter_bottlenecks(gdf: gpd.GeoDataFrame, min_block_width: float) -> gpd.GeoDataFrame:
    logger.info("Filtering bottlenecks and small blocks")

    def _filter_bottlenecks_helper(poly):
        try:
            return poly.intersection(poly.buffer(-min_block_width / 2).buffer(min_block_width / 2, join_style=2))
        except:
            return poly

    gdf.geometry = gdf["geometry"].apply(_filter_bottlenecks_helper)
    gdf = gdf[~gdf["geometry"].is_empty]
    gdf = gdf.explode(ignore_index=True)
    gdf = gdf[gdf.type == "Polygon"]

    return gdf


@_validate_and_process_gdfs
def cut_urban_blocks(
    boundaries_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame | None,
    polygons_gdf: gpd.GeoDataFrame | None,
    fill_holes: bool = True,
    min_block_width: float | None = None,
) -> gpd.GeoDataFrame:
    boundaries_gdf = _exclude_polygons(boundaries_gdf, polygons_gdf)
    blocks_gdf = _get_enclosures(boundaries_gdf, lines_gdf)
    if fill_holes:
        blocks_gdf = _fill_holes(blocks_gdf)
    blocks_gdf = _filter_overlapping(blocks_gdf)
    if min_block_width is not None:
        blocks_gdf = _filter_bottlenecks(blocks_gdf, min_block_width)
    logger.success("Blocks are successfully cut!")
    return gpd.GeoDataFrame(blocks_gdf.reset_index(drop=True)[["geometry"]])
