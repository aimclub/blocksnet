import pandas as pd
import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.ops import polygonize


def _exclude_polygons(boundaries_gdf: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame):
    # logger.info("Excluding polygon objects from blocks")
    polygons_gdf = gpd.GeoDataFrame(geometry=[polygons_gdf.union_all()], crs=polygons_gdf.crs)
    return boundaries_gdf.overlay(polygons_gdf, how="difference")


def _get_enclosures(boundaries_gdf: gpd.GeoDataFrame, lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # logger.info("Setting up enclosures")
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


def cut_blocks(
    boundaries_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    boundaries_gdf = _exclude_polygons(boundaries_gdf, polygons_gdf)
    blocks_gdf = _get_enclosures(boundaries_gdf, lines_gdf)
    return gpd.GeoDataFrame(blocks_gdf.reset_index(drop=True)[["geometry"]])
