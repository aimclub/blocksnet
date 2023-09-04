"""
This module gets data from the OSM. The module also implements several methods of data processing.
These methods allow you to connect parts of the data processing pipeline.
"""

import geopandas as gpd
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm
from typing import Literal
from pydantic import BaseModel, field_validator

from .aggregate_parameters import AggregateParameters
from ..models.city_model import CityBlockFeature, AccessibilityMatrix
from ..models import PolygonGeoJSON
from ..method.blocks.blocks_cutter import BlocksCutterFeatureProperties
from .accs_matrix_calculator import Accessibility

tqdm.pandas()


class DataGetter(BaseModel):
    """
    This class is used to get and pre-process data to be used in calculations in other modules.
    """

    blocks: PolygonGeoJSON[BlocksCutterFeatureProperties]

    @field_validator("blocks", mode="before")
    def validate_blocks(value):
        if isinstance(value, gpd.GeoDataFrame):
            return PolygonGeoJSON[BlocksCutterFeatureProperties].from_gdf(value)
        return value

    def get_accessibility_matrix(self, graph: nx.Graph) -> AccessibilityMatrix:
        """
        This function returns an accessibility matrix for a city. The matrix is calculated using
        the `Accessibility` class.

        Args:
            blocks (GeoDataFrame, optional): A GeoDataFrame containing information about the blocks in the city.
            Defaults to None.
            graph (Graph, optional): A networkx graph representing the city's road network. Defaults to None.

        Returns:
            np.ndarray: An accessibility matrix for the city.
        """

        accessibility = Accessibility(self.blocks.to_gdf(), graph)
        return AccessibilityMatrix(df=accessibility.get_matrix())

    @staticmethod
    def _get_living_area(row) -> float:
        """
        This function calculates the living area of a building based on the data in the given row.

        Args:
            row (pd.Series): A row of data containing information about a building.

        Returns:
            float: The calculated living area of the building.
        """

        if row["living_area"]:
            return float(row["living_area"])
        if row["is_living"]:
            if row["storeys_count"]:
                if row["building_area"]:
                    living_area = row["building_area"] * row["storeys_count"] * 0.7

                    return living_area
                return 0.0
            return 0.0
        return 0.0

    @staticmethod
    def _get_living_area_pyatno(row) -> float:
        """
        This function calculates the living area of a building based on the data in the given row.
        If the `living_area` attribute is not available, the function returns 0.

        Args:
            row (pd.Series): A row of data containing information about a building.

        Returns:
            float: The calculated living area of the building.
        """

        if row["living_area"]:
            return float(row["building_area"])
        return 0.0

    def aggregate_blocks_info(self, params: AggregateParameters) -> "PolygonGeoJSON[CityBlockFeature]":
        """
        This function aggregates information about blocks in a city. The information includes data about buildings,
        green spaces, and parking spaces.

        Args:
            blocks (gpd.GeoDataFrame): A GeoDataFrame containing information about the blocks in the city.
            buildings (gpd.GeoDataFrame): A GeoDataFrame containing information about buildings in the city.
            greenings (gpd.GeoDataFrame): A GeoDataFrame containing information about green spaces in the city.
            parkings (gpd.GeoDataFrame): A GeoDataFrame containing information about parking spaces in the city.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing aggregated information about blocks in the city.
        """

        blocks = self.blocks.to_gdf()
        buildings = params.buildings.to_gdf()
        greenings = params.greenings.to_gdf()
        parkings = params.parkings.to_gdf()

        buildings["living_area"].fillna(0, inplace=True)
        buildings["storeys_count"].fillna(0, inplace=True)
        tqdm.pandas(desc="Restoring living area")
        # TODO: is tqdm really necessary?
        buildings["living_area"] = buildings.progress_apply(self._get_living_area, axis=1)
        tqdm.pandas(desc="Restoring living area squash")
        # TODO: is tqdm really necessary?
        buildings["living_area_pyatno"] = buildings.progress_apply(self._get_living_area_pyatno, axis=1)
        buildings["total_area"] = buildings["building_area"] * buildings["storeys_count"]

        blocks_and_greens = (
            gpd.sjoin(blocks, greenings, predicate="intersects", how="left")
            .groupby("id")
            .agg(
                {
                    "current_green_capacity": "sum",
                    "current_green_area": "sum",
                }
            )
        )
        blocks_and_greens = (
            blocks_and_greens.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks_and_parkings = (
            gpd.sjoin(blocks, parkings, predicate="intersects", how="left")
            .groupby("id")
            .agg({"current_parking_capacity": "sum"})
        )
        blocks_and_parkings = (
            blocks_and_parkings.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks_and_buildings = (
            gpd.sjoin(blocks, buildings, predicate="intersects", how="left")
            .drop(columns=["index_right"])
            .groupby("id")
            .agg(
                {
                    "population_balanced": "sum",
                    "building_area": "sum",
                    "storeys_count": "median",
                    "total_area": "sum",
                    "living_area": "sum",
                    "living_area_pyatno": "sum",
                }
            )
        )
        blocks_and_buildings = (
            blocks_and_buildings.reset_index(drop=True).reset_index(drop=False).rename(columns={"index": "block_id"})
        )

        blocks.reset_index(drop=False, inplace=True)

        blocks_info_aggregated = pd.merge(blocks_and_buildings, blocks_and_greens)
        blocks_info_aggregated = pd.merge(blocks_info_aggregated, blocks_and_parkings)

        blocks_info_aggregated = gpd.GeoDataFrame(
            pd.merge(blocks, blocks_info_aggregated, left_on="index", right_on="block_id").drop(
                columns=["index", "id"]
            ),
            geometry="geometry",
        )
        blocks_info_aggregated.rename(
            columns={"building_area": "building_area_pyatno", "total_area": "building_area"}, inplace=True
        )

        blocks_info_aggregated["current_industrial_area"] = (
            blocks_info_aggregated["building_area_pyatno"] - blocks_info_aggregated["living_area_pyatno"]
        )
        blocks_info_aggregated.rename(
            columns={
                "population_balanced": "current_population",
                "storeys_count": "floors",
                "living_area_pyatno": "current_living_area",
            },
            inplace=True,
        )
        blocks_info_aggregated["area"] = blocks_info_aggregated["geometry"].area
        blocks_info_aggregated.drop(columns=["building_area_pyatno", "building_area", "living_area"], inplace=True)
        blocks_info_aggregated["is_living"] = blocks_info_aggregated["current_population"].apply(lambda x: x > 0)
        return PolygonGeoJSON[CityBlockFeature].from_gdf(blocks_info_aggregated)
