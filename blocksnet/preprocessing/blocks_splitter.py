import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from longsgis import voronoiDiagram4plg
from tqdm import tqdm
from ..models import BaseSchema


class BlocksSchema(BaseSchema):
    """
    Schema for validating blocks GeoDataFrame.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types for the blocks, default is [shapely.Polygon].
    """

    _geom_types = [shapely.Polygon]


class BuildingsSchema(BaseSchema):
    """
    Schema for validating buildings GeoDataFrame.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types for the buildings, default is [shapely.Point].
    """

    _geom_types = [shapely.Point]


class BlocksSplitter:
    """
    Splits blocks based on the distribution of buildings within them.

    Parameters
    ----------
    blocks : gpd.GeoDataFrame
        GeoDataFrame containing block data. Must contain the following columns:
        - index : int
        - geometry : Polygon

    buildings : gpd.GeoDataFrame
        GeoDataFrame containing building data. Must contain the following columns:
        - index : int
        - geometry : Point

    Raises
    ------
    AssertionError
        If the Coordinate Reference Systems (CRS) of `blocks` and `buildings` do not match.

    Methods
    -------
    run(n_clusters: int = 4, points_quantile: float = 0.98, area_quantile: float = 0.95) -> gpd.GeoDataFrame
        Splits blocks based on buildings distribution.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form within each block, default is 4.

        points_quantile : float
            Quantile value to filter blocks by the number of points, default is 0.98.

        area_quantile : float
            Quantile value to filter blocks by area, default is 0.95.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing all the blocks.
    """

    def __init__(self, blocks: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame):
        blocks = BlocksSchema(blocks)
        buildings = BuildingsSchema(buildings)
        assert blocks.crs == buildings.crs, "Blocks CRS must match buildings CRS"
        self.blocks = blocks
        self.buildings = buildings

    @staticmethod
    def _drop_index_columns(gdf) -> None:
        """
        Drops index columns from a GeoDataFrame if they exist.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame from which to drop index columns.
        """
        if "index_left" in gdf.columns:
            gdf.drop(columns=["index_left"], inplace=True)
        if "index_right" in gdf.columns:
            gdf.drop(columns=["index_right"], inplace=True)

    @staticmethod
    def _split_block(block: shapely.Polygon, buildings: gpd.GeoDataFrame, n_clusters: int) -> gpd.GeoDataFrame:
        """
        Splits a block into smaller regions based on building locations using Voronoi diagrams and K-Means clustering.

        Parameters
        ----------
        block : shapely.Polygon
            The geometry of the block to be split.

        buildings : gpd.GeoDataFrame
            GeoDataFrame containing the buildings within the block.

        n_clusters : int
            Number of clusters to form within the block.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the split regions.
        """
        vd = voronoiDiagram4plg(buildings, block)
        vd = vd.explode(index_parts=True)

        X = vd["geometry"].apply(lambda geom: geom.bounds).tolist()
        X_flat = [coord for rect in X for coord in rect]
        X = np.array(X_flat).reshape(-1, 4)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        vd["cluster"] = kmeans.fit_predict(X_scaled)
        vd = vd.dissolve(by="cluster")
        vd = vd.explode(index_parts=False)

        return vd.reset_index(drop=True)

    def run(self, n_clusters: int = 4, points_quantile: float = 0.98, area_quantile: float = 0.95) -> gpd.GeoDataFrame:
        """
        Splits blocks based on the distribution of buildings.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form within each block, default is 4.

        points_quantile : float
            Quantile value to filter blocks by the number of points, default is 0.98.

        area_quantile : float
            Quantile value to filter blocks by area, default is 0.95.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the split blocks.
        """
        get_geom_coords = lambda geom: len(geom.exterior.coords)

        logger.info("Joining buildings and blocks to exclude duplicates")
        blocks = self.blocks.copy()
        buildings = self.buildings.copy()
        sjoin = buildings.sjoin(blocks)
        sjoin = sjoin.rename(columns={"index_right": "block_id"})
        sjoin["building_id"] = sjoin.index
        # sjoin['intersection_area'] = sjoin.apply(lambda s : blocks.loc[s.block_id,'geometry'].intersection(s.geometry), axis=1).area
        # sjoin = sjoin.sort_values("intersection_area").drop_duplicates(subset="building_id", keep="last")

        logger.info("Choosing blocks to be splitted")
        quantile_num_points = blocks.geometry.apply(get_geom_coords).quantile(points_quantile)
        quantile_area = blocks.area.quantile(area_quantile)
        filtered_blocks = blocks[
            (blocks.geometry.apply(get_geom_coords) > quantile_num_points) & (blocks.area > quantile_area)
        ]
        filtered_blocks = filtered_blocks[filtered_blocks.index.isin(sjoin.block_id)]
        sjoin = sjoin[sjoin.block_id.isin(filtered_blocks.index)]

        logger.info("Splitting filtered blocks")
        gdfs = []
        for block_id, buildings_gdf in tqdm(sjoin.groupby("block_id")):
            try:
                block_geometry = blocks.loc[block_id, "geometry"]
                gdfs.append(self._split_block(block_geometry, buildings_gdf, n_clusters))
            except:
                gdfs.append(filtered_blocks[filtered_blocks.index == block_id])

        new_blocks = pd.concat(gdfs)
        old_blocks = blocks[~blocks.index.isin(filtered_blocks.index)]
        return BlocksSchema(pd.concat([new_blocks, old_blocks]).reset_index())
