import geopandas as gpd
from tqdm import tqdm  # pylint: disable=import-error
import pandas as pd

tqdm.pandas()


class DataPreparation:
    """
    TODO: add docstring
    """

    def __init__(self, buildings: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame):
        self.buildings = buildings
        self.blocks = blocks

    def _get_living_area(self, row):
        """
        TODO: add docstring
        """
        if row["living_area"]:
            return row["living_area"]
        else:
            if row["is_living"]:
                if row["storeys_count"]:
                    if row["building_area"]:
                        living_area = row["building_area"] * row["storeys_count"] * 0.7

                        return living_area
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0

    def _get_living_area_pyatno(self, row):
        """
        TODO: add docstring
        """
        if row["living_area"]:
            return row["building_area"]
        else:
            return 0

    def run(self):
        """
        TODO: add docstring
        """
        self.buildings["living_area"].fillna(0, inplace=True)
        self.buildings["storeys_count"].fillna(0, inplace=True)
        self.buildings["living_area"] = self.buildings.progress_apply(self._get_living_area, axis=1)
        self.buildings["living_area_pyatno"] = self.buildings.progress_apply(self._get_living_area_pyatno, axis=1)
        self.buildings["total_area"] = self.buildings["building_area"] * self.buildings["storeys_count"]

        blocks_with_buildings_info = (
            gpd.sjoin(self.blocks, self.buildings, predicate="intersects", how="left")
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

        blocks_with_buildings_info = (
            blocks_with_buildings_info.reset_index(drop=True)
            .reset_index(drop=False)
            .rename(columns={"index": "block_id"})
        )

        blocks_with_buildings_info = gpd.GeoDataFrame(
            pd.merge(self.blocks, blocks_with_buildings_info, left_on="id", right_on="block_id").drop(columns=["id"]),
            geometry="geometry",
        )
        blocks_with_buildings_info.rename(
            columns={"building_area": "building_area_pyatno", "total_area": "building_area"}, inplace=True
        )

        blocks_with_buildings_info["current_industrial_area"] = (
            blocks_with_buildings_info["building_area_pyatno"] - blocks_with_buildings_info["living_area_pyatno"]
        )
        blocks_with_buildings_info.rename(
            columns={
                "population_balanced": "current_population",
                "storeys_count": "floors",
                "living_area_pyatno": "current_living_area",
            },
            inplace=True,
        )
        blocks_with_buildings_info["area"] = blocks_with_buildings_info["geometry"].area
        blocks_with_buildings_info.drop(columns=["building_area_pyatno", "building_area", "living_area"], inplace=True)

        return blocks_with_buildings_info
