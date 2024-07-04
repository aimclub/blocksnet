import pandas as pd
import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.ops import polygonize
from pyproj import CRS
from . import utils

BOUNDARIES_COLUMNS = ["geometry"]
BOUNDARIES_GEOM_TYPES = ["Polygon", "MultiPolygon"]

ROADS_COLUMNS = ["geometry"]
ROADS_GEOM_TYPES = ["LineString"]

RAILWAYS_COLUMNS = ["geometry"]
RAILWAYS_GEOM_TYPES = ["LineString"]

WATER_COLUMNS = ["geometry"]
WATER_GEOM_TYPES = ["LineString", "Polygon", "MultiPolygon"]

BLOCKS_COLUMNS = ["geometry"]


def validate_gdf(gdf: gpd.GeoDataFrame, columns: list[str], geom_types: list[str]) -> gpd.GeoDataFrame:
    """
    Validate provided GeoDataFrame with columns and geometry types and return validated copy.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to validate.
    columns : list[str]
        List of columns, that the GeoDataFrame must contain and filter others out.
    geom_types : list[str]
        Geometry types allowed in ``geometry`` column.

    Returns
    -------
    gpd.GeoDataFrame
        Validated GeoDataFrame.
    """
    assert isinstance(gdf, gpd.GeoDataFrame), "Must be instance of gpd.GeoDataFrame"
    gdf = gdf[columns].copy()
    assert all(gdf.geom_type.isin(geom_types)), f"Geometry must be in {geom_types}"
    return gdf


class BlocksGenerator:
    """
    Generates blocks (land parcels) based on boundaries, roads, railways, and water objects.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        Boundaries of a city or a territory. Must contain ``geometry`` column of ``Polygon`` or ``MultiPolygon`` geometries.
    roads : gpd.GeoDataFrame | None, optional
        Roads geometries. Can be obtained via OSMnx and must contain ``geometry`` column of ``LineString`` geometries. By default None.
    railways : gpd.GeoDataFrame | None, optional
        Railways geometries. Can be obtained via OSM tags ``railway==rail`` and must contain ``geometry`` column of ``LineString`` geometries. By default None.
    water : gpd.GeoDataFrame | None, optional
        Water objects geometries. Can be obtained via OSM tags like ``riverbank==*``, ``pond==*``, etc. Must contain ``geometry`` column of ``LineString``, ``Polygon`` or ``MultiPolygon``. By default None.

    Methods
    -------
    generate_blocks(min_block_width=None)
        Generates blocks based on the provided boundaries, roads, railways, and water objects.
    """

    def __init__(
        self,
        boundaries: gpd.GeoDataFrame,
        roads: gpd.GeoDataFrame | None = None,
        railways: gpd.GeoDataFrame | None = None,
        water: gpd.GeoDataFrame | None = None,
    ):
        """
        Initializes the BlocksGenerator with the specified boundaries, roads, railways, and water objects.

        Parameters
        ----------
        boundaries : gpd.GeoDataFrame
            Boundaries of a city or a territory. Must contain a ``geometry`` column with ``Polygon`` or ``MultiPolygon`` geometries.
        roads : gpd.GeoDataFrame, optional
            Roads geometries. Can be obtained via OSMnx and must contain a ``geometry`` column with ``LineString`` geometries. By default None.
        railways : gpd.GeoDataFrame, optional
            Railways geometries. Can be obtained via OSM tags ``railway==rail`` and must contain a ``geometry`` column with ``LineString`` geometries. By default None.
        water : gpd.GeoDataFrame, optional
            Water objects geometries. Can be obtained via OSM tags like ``riverbank==*``, ``pond==*``, etc. Must contain a ``geometry`` column with ``LineString``, ``Polygon``, or ``MultiPolygon`` geometries. By default None.
        """

        boundaries = validate_gdf(boundaries, BOUNDARIES_COLUMNS, BOUNDARIES_GEOM_TYPES)
        crs = boundaries.crs

        if roads is None:
            roads = gpd.GeoDataFrame(data=[], columns=ROADS_COLUMNS, crs=crs)
        else:
            roads = validate_gdf(roads, ROADS_COLUMNS, ROADS_GEOM_TYPES)

        if railways is None:
            railways = gpd.GeoDataFrame(data=[], columns=RAILWAYS_COLUMNS, crs=crs)
        else:
            railways = validate_gdf(railways, RAILWAYS_COLUMNS, RAILWAYS_GEOM_TYPES)

        if water is None:
            water = gpd.GeoDataFrame(data=[], columns=WATER_COLUMNS, crs=crs)
        else:
            water = validate_gdf(water, WATER_COLUMNS, WATER_GEOM_TYPES)

        for gdf in [roads, railways, water]:
            assert gdf.crs == crs, "All CRS must match"
            boundaries = boundaries.overlay(water[water.geom_type != "LineString"], how="difference")
            water["geometry"] = water["geometry"].apply(lambda x: x if x.geom_type == "LineString" else x.boundary)

        self.boundaries = boundaries
        self.roads = roads
        self.railways = railways
        self.water = water

    @property
    def local_crs(self) -> CRS:
        """
        Local CRS, defined by boundaries geometry.

        Returns
        -------
        CRS
            ``pyproj.CRS`` object
        """
        return self.boundaries.crs

    def generate_blocks(self, min_block_width: float | None = None) -> gpd.GeoDataFrame:
        """
        Generates blocks based on the provided boundaries, roads, railways, and water bodies.

        Parameters
        ----------
        min_block_width : float, optional
            Minimum width for the blocks. If specified, small blocks and bottlenecks will be filtered out. By default None.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the generated blocks with ``geometry`` column.
        """

        logger.info("Generating blocks")

        # create a GeoDataFrame with barriers
        barriers = gpd.GeoDataFrame(
            geometry=pd.concat([self.roads.geometry, self.water.geometry, self.railways.geometry]), crs=self.local_crs
        ).reset_index(drop=True)
        barriers = barriers.explode(index_parts=True).reset_index(drop=True).geometry

        # transform enclosed barriers to polygons
        logger.info("Setting up enclosures")
        blocks = self._get_enclosures(barriers, self.boundaries.geometry)

        # fill everything within blocks' boundaries
        logger.info("Filling holes")
        blocks = utils.fill_holes(blocks)

        # cleanup after filling holes
        logger.info("Dropping overlapping blocks")
        blocks = utils.drop_contained_geometries(blocks)
        blocks = blocks.explode(index_parts=False).reset_index(drop=True)

        blocks = blocks.rename(columns={"index": "block_id"})

        # apply negative and positive buffers consecutively to remove small blocks
        # and divide them on bottlenecks
        if min_block_width is not None:
            logger.info("Filtering bottlenecks and small blocks")
            blocks = utils.filter_bottlenecks(blocks, self.local_crs, min_block_width)
            blocks = self._reindex_blocks(blocks)

        # calculate blocks' area in local projected CRS
        logger.info("Calculating blocks area")
        blocks["area"] = blocks.to_crs(self.local_crs).area
        blocks = blocks[blocks["area"] > 1]

        # fix blocks' indices
        blocks = self._reindex_blocks(blocks)

        logger.info("Blocks generated")

        return blocks.to_crs(self.local_crs)[BLOCKS_COLUMNS]

    @staticmethod
    def _get_enclosures(barriers: gpd.GeoDataFrame, limit: gpd.GeoDataFrame):
        """
        Identifies enclosures formed by the barriers and within the limit.

        Parameters
        ----------
        barriers : gpd.GeoDataFrame
            GeoDataFrame containing barrier geometries.
        limit : gpd.GeoDataFrame
            GeoDataFrame containing the limit within which enclosures are to be found.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the identified enclosures.
        """
        barriers = pd.concat([barriers, limit.boundary]).reset_index(drop=True)

        unioned = barriers.unary_union
        polygons = polygonize(unioned)
        enclosures = gpd.GeoSeries(list(polygons), crs=barriers.crs)
        _, enclosure_idxs = enclosures.representative_point().sindex.query(limit.geometry, predicate="contains")
        enclosures = enclosures.iloc[np.unique(enclosure_idxs)]
        enclosures = enclosures.rename("geometry").reset_index()

        return enclosures

    @staticmethod
    def _reindex_blocks(blocks: gpd.GeoDataFrame):
        """
        Reindexes the blocks, ensuring each block has a unique block_id.

        Parameters
        ----------
        blocks : gpd.GeoDataFrame
            GeoDataFrame containing the blocks to be reindexed.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with reindexed blocks.
        """
        if "block_id" in blocks.columns:
            blocks = blocks.drop("block_id", axis=1).reset_index().rename(columns={"index": "block_id"})
        return blocks
