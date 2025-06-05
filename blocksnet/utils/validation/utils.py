import networkx as nx
import geopandas as gpd
import pandas as pd
from loguru import logger
from blocksnet.relations.accessibility import validate_accessibility_matrix
from blocksnet.relations.adjacency import validate_adjacency_graph


def ensure_crs(gdf: gpd.GeoDataFrame, *args):
    for arg in args:
        if arg.crs != gdf.crs:
            logger.warning("CRS of geodataframes do not match. Reprojecting")
            arg.to_crs(gdf.crs, inplace=True)


def validate_matrix(*args, **kwargs):
    logger.warning("Deprecated. Try relations/accessibility/validate_accessibility_matrix")
    validate_accessibility_matrix(*args, **kwargs)


def validate_graph(*args, **kwargs):
    logger.warning("Deprecated. Try relations/adjacency/validate_adjacency_graph")
    validate_adjacency_graph(*args, **kwargs)
