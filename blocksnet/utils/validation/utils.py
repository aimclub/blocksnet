import networkx as nx
import geopandas as gpd
import pandas as pd
from loguru import logger


def ensure_crs(gdf: gpd.GeoDataFrame, *args):
    for arg in args:
        if arg.crs != gdf.crs:
            logger.warning("CRS of geodataframes do not match. Reprojecting")
            arg.to_crs(gdf.crs, inplace=True)
