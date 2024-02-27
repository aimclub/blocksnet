from typing import ClassVar

import geopandas as gpd
import osmnx as ox
import pandas as pd

from ...models import Block, ServiceType
from ..base_method import BaseMethod


class VacantArea(BaseMethod):
    """
    A class to identify and calculate vacant areas within a specified block or blocks in a city model,
    taking into account various geographical features like buildings, highways, natural areas, etc.

    Attributes:
    - local_crs (ClassVar[int]): The local coordinate reference system (CRS) code used for spatial operations.
    - roads_buffer (ClassVar[int]): The buffer distance to apply around roads when identifying vacant areas.
    - buildings_buffer (ClassVar[int]): The buffer distance to apply around buildings when identifying vacant areas.
    - min_length (ClassVar[int]): The minimum length threshold for considered vacant areas.
    - min_area (ClassVar[int]): The minimum area threshold for considered vacant areas.
    - area_attitude (ClassVar[int]): The ratio used to filter areas based on their minimum bounding rectangle.
    """
    local_crs: ClassVar[int] = 32636

    min_lenght: ClassVar[int] = 4
    min_area: ClassVar[int] = 100
    area_attitude: ClassVar[int] = 2

    roads_buffer: ClassVar[int] = 10
    buildings_buffer: ClassVar[int] = 10
    buffer_min: ClassVar[int] = 20
    buffer_max: ClassVar[int] = 40

    @staticmethod
    def _dwn_other(block, local_crs) -> gpd.GeoSeries:
        """
        Download other non-standard areas within a block.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.

        Returns:
        - A GeoSeries of geometries representing other non-standard areas.
        """
        try:
            other = ox.features_from_polygon(block, tags={"man_made": True, "aeroway": True, "military": True})
            other["geometry"] = other["geometry"].to_crs(local_crs)
            return other.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_leisure(block, local_crs) -> gpd.GeoSeries:
        """
        Download leisure areas within a block.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.

        Returns:
        - A GeoSeries of geometries representing leisure areas.
        """
        try:
            leisure = ox.features_from_polygon(block, tags={"leisure": True})
            leisure["geometry"] = leisure["geometry"].to_crs(local_crs)
            return leisure.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_landuse(block, local_crs) -> gpd.GeoSeries:
        """
        Download land use areas within a block, excluding residential.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.

        Returns:
        - A GeoSeries of geometries representing non-residential land use areas.
        """
        try:
            landuse = ox.features_from_polygon(block, tags={"landuse": True})
            if not landuse.empty:
                landuse = landuse[landuse["landuse"] != "residential"]
            landuse["geometry"] = landuse["geometry"].to_crs(local_crs)
            return landuse.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_amenity(block, local_crs) -> gpd.GeoSeries:
        """
        Download amenity areas within a block.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.

        Returns:
        - A GeoSeries of geometries representing amenity areas.
        """
        try:
            amenity = ox.features_from_polygon(block, tags={"amenity": True})
            amenity["geometry"] = amenity["geometry"].to_crs(local_crs)
            return amenity.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_buildings(block, local_crs, buildings_buffer) -> gpd.GeoSeries:
        """
        Download building areas within a block and apply a buffer.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.
        - buildings_buffer: Buffer distance to apply to building geometries.

        Returns:
        - A GeoSeries of buffered building geometries.
        """
        try:
            buildings = ox.features_from_polygon(block, tags={"building": True})
            if buildings_buffer:
                buildings["geometry"] = buildings["geometry"].to_crs(local_crs).buffer(buildings_buffer)
            return buildings.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_natural(block, local_crs) -> gpd.GeoSeries:
        """
        Download natural feature areas within a block, excluding bays.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.

        Returns:
        - A GeoSeries of geometries representing natural feature areas.
        """
        try:
            natural = ox.features_from_polygon(block, tags={"natural": True})
            if not natural.empty:
                natural = natural[natural["natural"] != "bay"]
            natural["geometry"] = natural["geometry"].to_crs(local_crs)
            return natural.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_waterway(block, local_crs) -> gpd.GeoSeries:
        """
        Download waterway areas within a block.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.

        Returns:
        - A GeoSeries of geometries representing waterway areas.
        """
        try:
            waterway = ox.features_from_polygon(block, tags={"waterway": True})
            waterway["geometry"] = waterway["geometry"].to_crs(local_crs)
            return waterway.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_highway(block, local_crs, roads_buffer) -> gpd.GeoSeries:
        """
        Download highway areas within a block, excluding paths, footways, and pedestrian areas, and apply a buffer.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.
        - roads_buffer: Buffer distance to apply to highway geometries.

        Returns:
        - A GeoSeries of buffered highway geometries.
        """
        try:
            highway = ox.features_from_polygon(block, tags={"highway": True})
            condition = (
                (highway["highway"] != "path")
                & (highway["highway"] != "footway")
                & (highway["highway"] != "pedestrian")
            )
            filtered_highway = highway[condition]
            if roads_buffer:
                filtered_highway["geometry"] = filtered_highway["geometry"].to_crs(local_crs).buffer(roads_buffer)
            return filtered_highway.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_path(block, local_crs) -> gpd.GeoSeries:
        """
        Download path and footway areas within a block and apply a buffer.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.

        Returns:
        - A GeoSeries of buffered path and footway geometries.
        """
        try:
            tags = {"highway": "path", "highway": "footway"}
            path = ox.features_from_polygon(block, tags=tags)
            path["geometry"] = path["geometry"].to_crs(local_crs).buffer(1)
            return path.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _dwn_railway(block, local_crs) -> gpd.GeoSeries:
        """
        Download railway areas within a block, excluding subways.

        Parameters:
        - block: The block's polygon geometry.
        - local_crs: The local coordinate reference system to project the geometries.

        Returns:
        - A GeoSeries of geometries representing railway areas.
        """
        try:
            railway = ox.features_from_polygon(block, tags={"railway": True})
            if not railway.empty:
                railway = railway[railway["railway"] != "subway"]
            railway["geometry"] = railway["geometry"].to_crs(local_crs)
            return railway.geometry
        except (ValueError, AttributeError) as exc:
            print(f"Error encountered: {exc}")
            return gpd.GeoSeries()

    @staticmethod
    def _create_minimum_bounding_rectangle(polygon) -> gpd.GeoSeries:
        """
        Create a minimum bounding rectangle for a given polygon.

        Parameters:
        - polygon: A polygon geometry for which to create the bounding rectangle.

        Returns:
        - A GeoSeries containing the minimum bounding rectangle of the input polygon.
        """
        return polygon.minimum_rotated_rectangle

    @staticmethod
    def _buffer_and_union(row, buffer_distance=1.1) -> gpd.GeoSeries:
        """
        Apply a buffer to a geometry and return the buffered geometry.

        Parameters:
        - row: A DataFrame row containing a geometry.
        - buffer_distance: The distance by which to buffer the geometry.

        Returns:
        - A GeoSeries containing the buffered geometry.
        """
        polygon = row["geometry"]
        buffer_polygon = polygon.buffer(buffer_distance)
        return buffer_polygon



    def get_vacant_area(self, block_id: int | Block) -> gpd.GeoDataFrame:
        """
        Calculate the vacant area within a given block or blocks.

        Parameters:
        - block: Either a single block identifier or a Block object representing the area to analyze.

        Returns:
        - A GeoDataFrame containing the vacant areas within the specified block(s), along with their area and length.
        """

        blocks_gdf = self.city_model.get_blocks_gdf()
        if block_id:
            blocks_gdf = blocks_gdf.iloc[[block_id]]
        block_buffer = blocks_gdf.buffer(self.buffer_min).to_crs(epsg=4326).unary_union

        # Define a list of functions to download data. Functions that require additional arguments
        # are included as lambda functions to defer their execution until later.
        occupied_area_sources = [
            lambda: self._dwn_leisure(block_buffer, self.local_crs),
            lambda: self._dwn_landuse(block_buffer, self.local_crs),
            lambda: self._dwn_other(block_buffer, self.local_crs),
            lambda: self._dwn_amenity(block_buffer, self.local_crs),
            lambda: self._dwn_buildings(block_buffer, self.local_crs, self.buildings_buffer),
            lambda: self._dwn_natural(block_buffer, self.local_crs),
            lambda: self._dwn_waterway(block_buffer, self.local_crs),
            lambda: self._dwn_highway(block_buffer, self.local_crs, self.roads_buffer),
            lambda: self._dwn_railway(block_buffer, self.local_crs),
            lambda: self._dwn_path(block_buffer, self.local_crs)
        ]

        # Execute each function in the list to obtain the GeoDataFrames and concatenate them
        occupied_area = pd.concat([func() for func in occupied_area_sources])
        
        occupied_area = gpd.GeoDataFrame(geometry=gpd.GeoSeries(occupied_area), crs=self.local_crs)
        occupied_area = occupied_area[(occupied_area.geometry.geom_type == "Polygon") |
                                    (occupied_area.geometry.geom_type == "MultiPolygon")]
        
        block_buffer2 = gpd.GeoDataFrame(geometry=blocks_gdf.buffer(self.buffer_max))
        blocks_new = gpd.overlay(block_buffer2, occupied_area, how="difference")
        blocks_new = gpd.overlay(blocks_gdf, blocks_new, how="intersection").explode(index_parts=True)

        blocks_new.reset_index(drop=True, inplace=True)
        blocks_new["buffered_geometry"] = blocks_new.apply(self._buffer_and_union, axis=1)
        unified_geometry = blocks_new["buffered_geometry"].unary_union
        result_gdf = gpd.GeoDataFrame(geometry=[unified_geometry], crs=self.local_crs).explode(index_parts=True)
        result_gdf["area"] = result_gdf.geometry.area
        result_gdf.reset_index(drop=True, inplace=True)

        blocks_filtered_area = result_gdf[result_gdf["area"] >= self.min_area]
        if blocks_filtered_area.empty:
            print("The block has no vacant area")
            return gpd.GeoDataFrame()

        # # Filtering based on minimum bounding rectangle condition
        blocks_filtered_area = blocks_filtered_area[
            blocks_filtered_area.apply(
                lambda row: row["geometry"].area * self.area_attitude > self._create_minimum_bounding_rectangle(row["geometry"]).area, axis=1
            )
        ]
        # # Vectorized operation for filtering based on area and length
        blocks_filtered_area["length"] = blocks_filtered_area.geometry.length
        blocks_filtered_area = blocks_filtered_area[(blocks_filtered_area["area"] / blocks_filtered_area["length"] >= self.min_lenght)]

        blocks_filtered_area.reset_index(drop=True, inplace=True)
        return blocks_filtered_area
