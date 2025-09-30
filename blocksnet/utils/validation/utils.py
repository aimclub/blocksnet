import networkx as nx
import geopandas as gpd
import pandas as pd
from loguru import logger
from blocksnet.relations.accessibility import validate_accessibility_matrix
from blocksnet.relations.adjacency import validate_adjacency_graph


def ensure_crs(gdf: gpd.GeoDataFrame, *args):
    """Ensure crs.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Description.
    *args : tuple
        Description.

    """
    for arg in args:
        if arg.crs != gdf.crs:
            logger.warning("CRS of GeoDataFrame do not match first provided one. Reprojecting")
            arg.to_crs(gdf.crs, inplace=True)


def validate_matrix(*args, **kwargs):
    """Validate matrix.

    Parameters
    ----------
    *args : tuple
        Description.
    **kwargs : dict
        Description.

    """
    logger.warning("Deprecated. Try relations/accessibility/validate_accessibility_matrix")
    validate_accessibility_matrix(*args, **kwargs)


def validate_graph(*args, **kwargs):
    """Validate graph.

    Parameters
    ----------
    *args : tuple
        Description.
    **kwargs : dict
        Description.

    """
    logger.warning("Deprecated. Try relations/adjacency/validate_adjacency_graph")
    validate_adjacency_graph(*args, **kwargs)
