import networkx as nx
import geopandas as gpd
import pandas as pd
from loguru import logger


def ensure_crs(gdf: gpd.GeoDataFrame, *args):
    for arg in args:
        if arg.crs != gdf.crs:
            logger.warning("CRS of geodataframes do not match. Reprojecting.")
            arg.to_crs(gdf.crs, inplace=True)


def validate_matrix(matrix: pd.DataFrame, blocks_df: pd.DataFrame | None = None):
    if not all(matrix.index == matrix.columns):
        raise ValueError("Matrix index and columns must match")
    if blocks_df is not None and isinstance(blocks_df, pd.DataFrame):
        if not blocks_df.index.isin(matrix.index).all():
            raise ValueError("Block index must be in matrix index")
