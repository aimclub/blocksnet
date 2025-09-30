import pandas as pd
import geopandas as gpd
from typing import cast
from .schemas import BlocksSchema


def _separate_blocks(blocks: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Separate blocks.

    Parameters
    ----------
    blocks : gpd.GeoDataFrame
        Description.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        Description.

    """
    valid_blocks = blocks[blocks.is_valid].copy()
    invalid_blocks = blocks[~blocks.is_valid].copy()
    return valid_blocks, invalid_blocks


def _concat_blocks(valid_blocks: gpd.GeoDataFrame, invalid_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Concat blocks.

    Parameters
    ----------
    valid_blocks : gpd.GeoDataFrame
        Description.
    invalid_blocks : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    blocks = pd.concat([valid_blocks, invalid_blocks])
    blocks = blocks.reset_index(drop=True)
    return cast(gpd.GeoDataFrame, blocks)


def _make_valid(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make valid.

    Parameters
    ----------
    blocks : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    geometry = blocks.make_valid()
    geometry = geometry.explode(ignore_index=True)
    blocks = gpd.GeoDataFrame(geometry=geometry, crs=blocks.crs)
    return blocks


def _explode_and_filter(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode and filter.

    Parameters
    ----------
    blocks : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    blocks = blocks.explode(ignore_index=True)
    blocks = blocks[blocks.geom_type.isin(["Polygon"])]
    return blocks.reset_index(drop=True)


def postprocess_urban_blocks(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Postprocess urban blocks.

    Parameters
    ----------
    blocks : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    blocks = BlocksSchema(blocks)
    valid_blocks, invalid_blocks = _separate_blocks(blocks)
    invalid_blocks = _make_valid(invalid_blocks)
    blocks = _concat_blocks(valid_blocks, invalid_blocks)
    return _explode_and_filter(blocks)
