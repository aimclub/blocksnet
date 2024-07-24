import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from ..models import BaseSchema
from loguru import logger
from shapely.ops import polygonize
from pyproj import CRS
from .. import utils


class BoundariesSchema(BaseSchema):
    _geom_types = [shapely.Polygon, shapely.MultiPolygon]


class RoadsSchema(BaseSchema):
    _geom_types = [shapely.LineString]


class RailwaysSchema(BaseSchema):
    _geom_types = [shapely.LineString]


class WaterSchema(BaseSchema):
    _geom_types = [shapely.LineString, shapely.Polygon, shapely.MultiPolygon]


class BlocksSchema(BaseSchema):
    _geom_types = [shapely.Polygon]


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
    run(min_block_width=None)
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
            Roads geometries. Must contain a ``geometry`` column with ``LineString`` geometries. By default None.

            Possible OSM tags:
            - ``highway`` : construction, crossing, living_street, motorway, motorway_link, motorway_junction, pedestrian, primary, primary_link, raceway, residential, road, secondary, secondary_link, services, tertiary, tertiary_link, track, trunk, trunk_link, turning_circle, turning_loop, unclassified
            - ``service`` : living_street, emergency_access

        railways : gpd.GeoDataFrame, optional
            Railways geometries. Must contain a ``geometry`` column with ``LineString`` geometries. By default None.

            Possible OSM tags:
            - ``railway`` : rail

        water : gpd.GeoDataFrame, optional
            Water objects geometries. Must contain a ``geometry`` column with ``LineString``, ``Polygon``, or ``MultiPolygon`` geometries. By default None.

            Possible OSM tags:
            - ``riverbank``
            - ``reservoir``
            - ``basin``
            - ``dock``
            - ``canal``
            - ``pond``
            - ``natural`` : water, bay
            - ``waterway`` : river, canal, ditch
            - ``landuse`` : basin
        """

        logger.info("Check boundaries schema")
        boundaries = BoundariesSchema(boundaries)
        crs = boundaries.crs

        logger.info("Check roads schema")
        if roads is None:
            roads = RoadsSchema.to_gdf().to_crs(crs)
        else:
            roads = RoadsSchema(roads)

        logger.info("Check railways schema")
        if railways is None:
            railways = RailwaysSchema.to_gdf().to_crs(crs)
        else:
            railways = RailwaysSchema(railways)

        logger.info("Check water schema")
        if water is None:
            water = WaterSchema.to_gdf().to_crs(crs)
        else:
            water = WaterSchema(water)

        for gdf in [roads, railways, water]:
            assert gdf.crs == crs, "All CRS must match"

        logger.info("Exclude water objects")
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

    def run(self, min_block_width: float | None = None) -> gpd.GeoDataFrame:
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

        return BlocksSchema(blocks.to_crs(self.local_crs))

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
