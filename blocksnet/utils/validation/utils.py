import networkx as nx
import geopandas as gpd
import pandas as pd
from loguru import logger
from blocksnet.relations.accessibility import validate_accessibility_matrix
from blocksnet.relations.adjacency import validate_adjacency_graph


def ensure_crs(gdf: gpd.GeoDataFrame, *args):
    """Ensure GeoDataFrames share the CRS of the first argument.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Reference GeoDataFrame providing the target CRS.
    *args
        Additional GeoDataFrames to compare and, if necessary, reproject in
        place.
    """

    for arg in args:
        if arg.crs != gdf.crs:
            logger.warning("CRS of GeoDataFrame do not match first provided one. Reprojecting")
            arg.to_crs(gdf.crs, inplace=True)


def validate_matrix(*args, **kwargs):
    """Deprecated wrapper for accessibility matrix validation.

    Raises
    ------
    ValueError
        Propagated from :func:`blocksnet.relations.accessibility.validate_accessibility_matrix`.
    """

    logger.warning("Deprecated. Try relations/accessibility/validate_accessibility_matrix")
    validate_accessibility_matrix(*args, **kwargs)


def validate_graph(*args, **kwargs):
    """Deprecated wrapper for adjacency graph validation.

    Raises
    ------
    ValueError
        Propagated from :func:`blocksnet.relations.adjacency.validate_adjacency_graph`.
    """

    logger.warning("Deprecated. Try relations/adjacency/validate_adjacency_graph")
    validate_adjacency_graph(*args, **kwargs)
