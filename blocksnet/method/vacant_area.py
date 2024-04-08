"""
Identify vacant areas within specified blocks.
"""
import geopandas as gpd
import osmnx as ox
import pandas as pd
import asyncio
import nest_asyncio
from pydantic import Field

from ..models import Block
from .base_method import BaseMethod

# For the async work
nest_asyncio.apply()


class VacantArea(BaseMethod):
    """
    A class for identifying and calculating vacant areas within specified city blocks.

    This class extends `BaseMethod` and includes methods to identify and calculate vacant areas in city blocks
    by considering various geographical features such as buildings, roads, natural areas, and amenities. The vacant
    areas are filtered based on their size, shape, and proximity to existing features.

    Attributes
    ----------
    area_to_length_min : float
        Minimum ratio of area to length for considering an area as vacant.
    area_min : float
        Minimum area size for considering an area as vacant.
    area_to_mrr_area_min : float
        Minimum ratio of area to minimum rotated rectangle (MRR) area for considering an area as vacant.
    path_buffer : float
        Buffer distance around path and footway geometries.
    roads_buffer : float
        Buffer distance around road geometries.
    buildings_buffer : float
        Buffer distance around building geometries.
    blocks_buffer_min : float
        Minimum buffer distance around blocks.
    blocks_buffer_max : float
        Maximum buffer distance around blocks.
    """

    area_to_length_min: float = Field(ge=0, default=4)
    area_min: float = Field(ge=0, default=100)
    area_to_mrr_area_min: float = Field(ge=0, default=0.5)

    path_buffer: float = Field(ge=0, default=1)
    roads_buffer: float = Field(ge=0, default=10)
    buildings_buffer: float = Field(ge=0, default=10)
    blocks_buffer_min: float = Field(ge=0, default=20)
    blocks_buffer_max: float = Field(ge=0, default=40)

    async def _dwn_other(self, geometry) -> gpd.GeoDataFrame:
        """
        Download non-standard areas within a block.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the geometries of non-standard areas.
        """
        try:
            other = ox.features_from_polygon(geometry, tags={"man_made": True, "aeroway": True, "military": True})
            other = other.to_crs(self.city_model.crs)
            return other[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoDataFrame()

    async def _dwn_leisure(self, geometry) -> gpd.GeoSeries:
        """
        Download leisure areas within a block.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoSeries
            A GeoSeries with the geometries of leisure areas.
        """
        try:
            leisure = ox.features_from_polygon(geometry, tags={"leisure": True})
            leisure = leisure.to_crs(self.city_model.crs)
            return leisure[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoDataFrame()

    async def _dwn_landuse(self, geometry) -> gpd.GeoDataFrame:
        """
        Download land use areas within a block, excluding residential areas.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the geometries of non-residential land use areas.
        """
        try:
            landuse = ox.features_from_polygon(geometry, tags={"landuse": True})
            if not landuse.empty:
                landuse = landuse[landuse["landuse"] != "residential"]
            landuse = landuse.to_crs(self.city_model.crs)
            return landuse[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoDataFrame()

    async def _dwn_amenity(self, geometry) -> gpd.GeoDataFrame:
        """
        Download amenity areas within a block.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the geometries of amenity areas.
        """
        try:
            amenity = ox.features_from_polygon(geometry, tags={"amenity": True})
            amenity = amenity.to_crs(self.city_model.crs)
            return amenity[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    async def _dwn_buildings(self, geometry) -> gpd.GeoDataFrame:
        """
        Download building areas within a block and apply a buffer.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with buffered building geometries.
        """
        try:
            buildings = ox.features_from_polygon(geometry, tags={"building": True})
            buildings = buildings.to_crs(self.city_model.crs)
            buildings["geometry"] = buildings["geometry"].buffer(self.buildings_buffer)
            return buildings[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    async def _dwn_natural(self, geometry) -> gpd.GeoDataFrame:
        """
        Download natural feature areas within a block, excluding bays.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the geometries of natural feature areas.
        """
        try:
            natural = ox.features_from_polygon(geometry, tags={"natural": True})
            if not natural.empty:
                natural = natural[natural["natural"] != "bay"]
            natural = natural.to_crs(self.city_model.crs)
            return natural[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoDataFrame()

    async def _dwn_waterway(self, geometry) -> gpd.GeoDataFrame:
        """
        Download waterway areas within a block.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the geometries of waterway areas.
        """
        try:
            waterway = ox.features_from_polygon(geometry, tags={"waterway": True})
            waterway = waterway.to_crs(self.city_model.crs)
            return waterway[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoDataFrame()

    async def _dwn_highway(self, block) -> gpd.GeoDataFrame:
        """
        Download highway areas within a block, applying a buffer and excluding certain types of highways.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with buffered highway geometries.
        """
        try:
            highway = ox.features_from_polygon(block, tags={"highway": True})
            condition = (
                (highway["highway"] != "path")
                & (highway["highway"] != "footway")
                & (highway["highway"] != "pedestrian")
            )
            filtered_highway = highway[condition]
            filtered_highway = filtered_highway.to_crs(self.city_model.crs)
            filtered_highway["geometry"] = filtered_highway["geometry"].buffer(self.roads_buffer)
            return filtered_highway[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoDataFrame()

    async def _dwn_path(self, geometry) -> gpd.GeoDataFrame:
        """
        Download path and footway areas within a block and apply a buffer.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with buffered path and footway geometries.
        """
        try:
            tags = {"highway": "path", "highway": "footway"}
            path = ox.features_from_polygon(geometry, tags=tags)
            path = path.to_crs(self.city_model.crs)
            path["geometry"] = path["geometry"].buffer(self.path_buffer)
            return path[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoDataFrame()

    async def _dwn_railway(self, geometry) -> gpd.GeoDataFrame:
        """
        Download railway areas within a block, excluding subways.

        Parameters
        ----------
        geometry : gpd.GeoSeries
            The polygon geometry of the block.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame with the geometries of railway areas.
        """
        try:
            railway = ox.features_from_polygon(geometry, tags={"railway": True})
            if not railway.empty:
                railway = railway[railway["railway"] != "subway"]
            railway = railway.to_crs(self.city_model.crs)
            return railway[["geometry"]]
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoDataFrame()

    async def _calculate(self, blocks: list[int] | list[Block] = []) -> gpd.GeoDataFrame:
        """
        Calculate vacant areas within specified blocks.

        Parameters
        ----------
        blocks : list[int] | list[Block]
            List of block identifiers or `Block` objects representing the areas to analyze. If not provided, the calculation will be done for the whole city.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the vacant areas within the specified blocks, including their area and length.
        """

        blocks_gdf = self.city_model.get_blocks_gdf()[["geometry"]]
        buildings_gdf = self.city_model.get_buildings_gdf()
        services_gdf = self.city_model.get_services_gdf()

        # filter gdf if blocks are provided
        if len(blocks) > 0:
            blocks = [b.id if isinstance(b, Block) else b for b in blocks]
            blocks_gdf = blocks_gdf.loc[blocks]
            buildings_gdf = buildings_gdf.loc[buildings_gdf["block_id"].isin(blocks)]
            services_gdf = services_gdf.loc[services_gdf["block_id"].isin(blocks)]

        buildings_gdf["geometry"] = buildings_gdf["geometry"].buffer(self.buildings_buffer)
        blocks_buffer = blocks_gdf.buffer(self.blocks_buffer_min).to_crs(epsg=4326).unary_union

        # setting occupied area with all the buffers possible
        occupied_areas = [
            asyncio.create_task(self._dwn_other(blocks_buffer)),
            asyncio.create_task(self._dwn_landuse(blocks_buffer)),
            asyncio.create_task(self._dwn_natural(blocks_buffer)),
            asyncio.create_task(self._dwn_waterway(blocks_buffer)),
            asyncio.create_task(self._dwn_highway(blocks_buffer)),
            asyncio.create_task(self._dwn_railway(blocks_buffer)),
            asyncio.create_task(self._dwn_path(blocks_buffer)),
            asyncio.create_task(self._dwn_leisure(blocks_buffer)),
            asyncio.create_task(self._dwn_amenity(blocks_buffer)),
            asyncio.create_task(self._dwn_buildings(blocks_buffer)),
        ]
        occupied_areas = await asyncio.gather(*occupied_areas)
        occupied_area = pd.concat([buildings_gdf, services_gdf, *occupied_areas])[["geometry"]]
        occupied_area = occupied_area.loc[occupied_area.geom_type.isin(["Polygon", "MultiPolygon"])]
        buffered_blocks_gdf = blocks_gdf.copy()
        buffered_blocks_gdf["geometry"] = buffered_blocks_gdf["geometry"].buffer(self.blocks_buffer_max)

        vacant_gdf = blocks_gdf.overlay(occupied_area, how="difference")
        # vacant_gdf = blocks_gdf.overlay(vacant_gdf, how="intersection")
        unified_geometry = vacant_gdf["geometry"].buffer(1.1).unary_union
        vacant_gdf = gpd.GeoDataFrame(geometry=[unified_geometry], crs=self.city_model.crs)
        vacant_gdf = vacant_gdf.explode(index_parts=True).reset_index(drop=True)

        # calculate filtering indicators
        vacant_gdf["area"] = vacant_gdf["geometry"].area
        vacant_gdf["mrr_area"] = vacant_gdf["geometry"].apply(lambda g: g.minimum_rotated_rectangle.area)
        vacant_gdf["length"] = vacant_gdf["geometry"].apply(lambda g: g.length)
        vacant_gdf["area_to_length"] = vacant_gdf["area"] / vacant_gdf["length"]
        vacant_gdf["area_to_mrr_area"] = vacant_gdf["area"] / vacant_gdf["mrr_area"]

        result_gdf = vacant_gdf.loc[vacant_gdf["area"] >= self.area_min]
        result_gdf = result_gdf.loc[vacant_gdf["area_to_mrr_area"] >= self.area_to_mrr_area_min]
        result_gdf = result_gdf.loc[vacant_gdf["area_to_length"] >= self.area_to_length_min]

        result_gdf = result_gdf.sjoin(blocks_gdf, how="left", predicate="within")
        result_gdf = result_gdf.rename(columns={"index_right": "block_id"})

        return result_gdf.reset_index(drop=True)

    def calculate(self, blocks: list[int] | list[Block] = []) -> gpd.GeoDataFrame:
        """
        Calculate vacant areas within specified blocks.

        Parameters
        ----------
        blocks : list[int] | list[Block]
            List of block identifiers or `Block` objects representing the areas to analyze. If not provided, the calculation will be done for the whole city.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the vacant areas within the specified blocks, including their area and length.
        """

        return asyncio.run(self._calculate(blocks))
