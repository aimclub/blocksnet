import geopandas as gpd
from .schemas import BuildingsSchema


def _impute_footprint_area(buildings_gdf: gpd.GeoDataFrame):
    """Impute footprint area.

    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Description.

    """
    sub_gdf = buildings_gdf[(buildings_gdf.footprint_area == 0) | (buildings_gdf.footprint_area.isna())]
    buildings_gdf.loc[sub_gdf.index, "footprint_area"] = buildings_gdf.area


def _impute_number_of_floors(buildings_gdf: gpd.GeoDataFrame, default_number_of_floors: int):
    """Impute number of floors.

    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Description.
    default_number_of_floors : int
        Description.

    """
    sub_gdf = buildings_gdf[(buildings_gdf.number_of_floors == 0) | (buildings_gdf.number_of_floors.isna())]
    buildings_gdf.loc[sub_gdf.index, "number_of_floors"] = default_number_of_floors


def _impute_build_floor_area(buildings_gdf: gpd.GeoDataFrame):
    """Impute build floor area.

    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Description.

    """
    sub_gdf = buildings_gdf[(buildings_gdf.build_floor_area == 0) | (buildings_gdf.build_floor_area.isna())]
    buildings_gdf.loc[sub_gdf.index, "build_floor_area"] = buildings_gdf.number_of_floors * buildings_gdf.footprint_area


def _impute_living_area(buildings_gdf: gpd.GeoDataFrame, living_area_coefficient: float):
    """Impute living area.

    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Description.
    living_area_coefficient : float
        Description.

    """
    sub_gdf = buildings_gdf[(buildings_gdf.living_area == 0) | (buildings_gdf.living_area.isna())]
    buildings_gdf.loc[sub_gdf.index, "living_area"] = (
        buildings_gdf.is_living * buildings_gdf.build_floor_area * living_area_coefficient
    )


def _impute_non_living_area(buildings_gdf: gpd.GeoDataFrame):
    """Impute non living area.

    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Description.

    """
    sub_gdf = buildings_gdf[(buildings_gdf.non_living_area == 0) | (buildings_gdf.non_living_area.isna())]
    buildings_gdf.loc[sub_gdf.index, "non_living_area"] = buildings_gdf.build_floor_area - buildings_gdf.living_area


def _impute_population(buildings_gdf: gpd.GeoDataFrame, living_demand: float):
    """Impute population.

    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Description.
    living_demand : float
        Description.

    """
    sub_gdf = buildings_gdf[
        ((buildings_gdf.population == 0) | (buildings_gdf.population.isna())) & buildings_gdf.is_living
    ]
    population = buildings_gdf.living_area // living_demand
    population.loc[population == 0] = 1
    buildings_gdf.loc[sub_gdf.index, "population"] = population


def impute_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    default_number_of_floors: int = 1,
    living_area_coefficient: float = 0.9,
    default_living_demand: float = 20,
) -> gpd.GeoDataFrame:
    """Impute buildings.

    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Description.
    default_number_of_floors : int, default: 1
        Description.
    living_area_coefficient : float, default: 0.9
        Description.
    default_living_demand : float, default: 20
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    buildings_gdf = BuildingsSchema(buildings_gdf)

    _impute_footprint_area(buildings_gdf)
    _impute_number_of_floors(buildings_gdf, default_number_of_floors)
    _impute_build_floor_area(buildings_gdf)
    _impute_living_area(buildings_gdf, living_area_coefficient)
    _impute_non_living_area(buildings_gdf)
    _impute_population(buildings_gdf, default_living_demand)

    return buildings_gdf
