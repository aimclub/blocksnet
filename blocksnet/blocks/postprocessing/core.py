import pandas as pd
import geopandas as gpd
from typing import cast
from .schemas import BlocksSchema


def _separate_blocks(blocks: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    valid_blocks = blocks[blocks.is_valid].copy()
    invalid_blocks = blocks[~blocks.is_valid].copy()
    return valid_blocks, invalid_blocks


def _concat_blocks(valid_blocks: gpd.GeoDataFrame, invalid_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = pd.concat([valid_blocks, invalid_blocks])
    blocks = blocks.reset_index(drop=True)
    return cast(gpd.GeoDataFrame, blocks)


def _make_valid(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geometry = blocks.make_valid()
    geometry = geometry.explode(ignore_index=True)
    blocks = gpd.GeoDataFrame(geometry=geometry, crs=blocks.crs)
    return blocks


def _explode_and_filter(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = blocks.explode(ignore_index=True)
    blocks = blocks[blocks.geom_type.isin(["Polygon"])]
    return blocks.reset_index(drop=True)


def postprocess_urban_blocks(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clean and validate blocks produced by the cutting pipeline.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Raw blocks GeoDataFrame to postprocess.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing only valid polygon geometries.
    """

    blocks = BlocksSchema(blocks)
    valid_blocks, invalid_blocks = _separate_blocks(blocks)
    invalid_blocks = _make_valid(invalid_blocks)
    blocks = _concat_blocks(valid_blocks, invalid_blocks)
    return _explode_and_filter(blocks)
