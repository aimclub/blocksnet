"""
This module contains methods for calculating a new feasibility profile for the selected area.

The maximisation parameter is the number of inhabitants.
"""
import itertools
from math import ceil

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SQUARE_METERS_IN_HECTARE = 10_000


def kindergarten_area_ranges(children_number: int) -> tuple[float, int]:
    children_number = ceil(children_number)

    conditions = [
        140 < children_number <= 180,
        180 < children_number <= 250,
        250 < children_number <= 281,
    ]

    choices = [(0.72, 180), (1.44, 250), (1.1, 280)]
    return tuple(np.select(conditions, choices, default=(0, 0)))


def kindergarten_area(children_number) -> tuple[float, int]:
    if children_number >= 280:
        return tuple(map(sum, zip(kindergarten_area_ranges(280), kindergarten_area(children_number - 280))))
    return kindergarten_area_ranges(children_number)


def school_area_ranges(schoolkids: int) -> tuple[float, int]:
    schoolkids = ceil(schoolkids)

    conditions = [
        100 < schoolkids <= 250,
        250 < schoolkids <= 300,
        300 < schoolkids <= 600,
        600 < schoolkids <= 800,
        800 < schoolkids <= 1101,
    ]

    choices = [(1.2, 250), (1.1, 300), (1.3, 600), (1.5, 800), (1.8, 1100)]
    return tuple(np.select(conditions, choices, default=(0, 0)))


def school_area(schoolkids) -> tuple[float, int]:
    if schoolkids >= 1100:
        return tuple(map(sum, zip(school_area_ranges(1100), school_area(schoolkids - 1100))))
    return school_area_ranges(schoolkids)


def balance_data(gdf, polygon, services_prov):  # pylint: disable=too-many-arguments
    """
    This function balances data about blocks in a city by intersecting the given GeoDataFrame with a polygon
    and calculating various statistics.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame containing information about blocks in the city.
        polygon (gpd.GeoSeries): A polygon representing the area to intersect with the blocks.
        school (GeoDataFrame): A GeoDataFrame containing information about schools in the city.
        kindergarten (GeoDataFrame): A GeoDataFrame containing information about kindergartens in the city.
        greening (GeoDataFrame): A GeoDataFrame containing information about green spaces in the city.

    Returns:
        dict: A dictionary containing balanced data about blocks in the city.
    """

    intersecting_blocks = gpd.overlay(gdf, polygon, how="intersection").drop(columns=["id"])
    intersecting_blocks.rename(columns={"id_1": "id"}, inplace=True)
    gdf = intersecting_blocks

    gdf["current_building_area"] = gdf["current_living_area"] + gdf["current_industrial_area"]
    gdf_ = gdf[
        [
            "block_id",
            "area",
            "current_living_area",
            "current_industrial_area",
            "current_population",
            "current_green_area",
            "floors",
        ]
    ]

    for service_type in services_prov.keys():
        gdf_ = gdf_.merge(
            services_prov[service_type][["id", f"population_unprov_{service_type}"]], left_on="block_id", right_on="id"
        )
    gdf_ = gdf_.drop(columns=["id_x", "id_y"])

    gdf_["area"] = gdf_["area"] / SQUARE_METERS_IN_HECTARE
    gdf_["current_living_area"] = gdf_["current_living_area"] / SQUARE_METERS_IN_HECTARE
    gdf_["current_industrial_area"] = gdf_["current_industrial_area"] / SQUARE_METERS_IN_HECTARE
    gdf_["current_green_area"] = gdf_["current_green_area"] / SQUARE_METERS_IN_HECTARE

    df_sum = gdf_.sum()
    df_sum["floors"] = gdf_["floors"].mean()
    df_new = pd.DataFrame(df_sum).T

    sample = df_new[df_new["area"] > 7].sample()
    sample = sample.to_dict("records")
    block = sample[0].copy()

    return block


# TODO? Maybe MasterPlan should be moved to a separate file
class MasterPlan:  # pylint: disable=too-many-instance-attributes,invalid-name
    """
    This class is aimed to calculate balanced parameters for masterplanning for the specified area
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        area,  # total area in hectares
        current_living_area=None,  # the current living area in hectares
        current_industrial_area=None,  # the current industrial area in hectares
        current_population=None,  # the current number of people
        current_green_area=None,  # the current green area in hectares
        current_op_area=None,  # the current public spaces area in hectares
        current_unprov_schoolkids=None,  # the current number of students who do not have enough places in
        # schools
        current_unprov_kids=None,  # the current number of kids who do not have enough places in kindergartens
        current_unprov_green_population=None,  # the current number of people who do not have enough green areas
    ):
        self.current_living_area = current_living_area if current_living_area is not None else 0
        self.current_industrial_area = current_industrial_area if current_industrial_area is not None else 0
        self.current_population = current_population if current_population is not None else 0
        self.current_green_area = current_green_area if current_green_area is not None else 0
        self.current_op_area = current_op_area if current_op_area is not None else 0

        self.current_unprov_schoolkids = current_unprov_schoolkids if current_unprov_schoolkids is not None else 0
        self.current_unprov_kids = current_unprov_kids if current_unprov_kids is not None else 0
        self.current_unprov_green_population = (
            current_unprov_green_population if current_unprov_green_population is not None else 0
        )

        self.cons = None
        self.bnds = None
        self.x0s = None
        self.results = pd.DataFrame()

        self.area = area

        self.BA_coef = 0.8
        self.FA_coef = 0.2
        self.LA_coef = 0.7
        self.IA_coef = 0.3

        self.F_max = 9
        self.b_min, self.b_max = 18 / SQUARE_METERS_IN_HECTARE, 30 / SQUARE_METERS_IN_HECTARE
        self.G_min, self.G_max = 6 / SQUARE_METERS_IN_HECTARE, 12 / SQUARE_METERS_IN_HECTARE

        self.SC_coef = 0.12
        self.KG_coef = 0.061
        self.OP_coef = 0.03 / SQUARE_METERS_IN_HECTARE

        self.P1_coef = 0.42 * 0.15 * 0.012
        self.P2_coef = 0.42 * 0.35 * 0.005

        self.max_building_area = self.area * self.BA_coef - self.current_living_area - self.current_industrial_area
        self.max_free_area = self.area * self.FA_coef - self.current_green_area - self.current_op_area

        self.max_living_area = self.max_building_area * self.LA_coef
        self.max_industrial_area = self.max_building_area * self.IA_coef

        self.max_living_area_full = self.max_living_area * self.F_max
        self.max_population = ceil(self.max_living_area_full / self.b_max)

    def sc_area(self, population: int) -> float:
        return school_area(self.SC_coef * population)[0]

    def kg_area(self, population: int) -> float:
        return kindergarten_area(self.KG_coef * population)[0]

    def op_area(self, population: int) -> float:
        return self.OP_coef * population

    def parking1_area(self, population: int) -> float:
        return self.P1_coef * population

    def parking2_area(self, population: int) -> float:
        return self.P2_coef * population

    @staticmethod
    def living_area(population: int, b):
        return population * b

    @staticmethod
    def green_area(population, G):
        return population * G

    def fun(self, x):
        return (
            self.area
            - self.living_area(x[0], x[1])
            - self.sc_area(x[0] + self.current_unprov_schoolkids)
            - self.kg_area(x[0] + self.current_unprov_kids)
            - self.green_area(x[0] + self.current_unprov_green_population, x[1])
            - self.op_area(x[0])
            - self.parking1_area(x[0])
            - self.parking2_area(x[0])
        )

    def bnds_and_cons(self):
        self.cons = (
            {"type": "ineq", "fun": lambda x: self.max_population - x[0]},
            {
                "type": "ineq",
                "fun": lambda x: self.max_living_area - self.living_area(x[0], x[1]) - self.parking1_area(x[0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: self.max_free_area
                - self.green_area(x[0] + self.current_unprov_green_population, x[2])
                - self.op_area(x[0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: self.max_industrial_area
                - self.sc_area(x[0] + self.current_unprov_schoolkids)
                - self.kg_area(x[0] + self.current_unprov_kids)
                - self.parking2_area(x[0]),
            },
            {"type": "ineq", "fun": self.fun},
        )

        self.bnds = ((0, self.max_population), (self.b_min, self.b_max), (self.G_min, self.G_max))

    def make_x0s(self):
        self.x0s = [
            (0, 0, 0),
            (1 / self.max_population, self.b_min, self.G_min),
            (self.max_population / 32, (self.b_min + self.b_max) / 2, (self.G_min + self.G_max) / 2),
            (self.max_population / 16, (self.b_min + self.b_max) / 2, (self.G_min + self.G_max) / 2),
            (self.max_population / 8, self.b_max, self.G_max),
        ]

    def find_optimal_solutions(self):
        """Find optimal solutions"""

        self.bnds_and_cons()
        self.make_x0s()

        results: list[pd.DataFrame] = []
        for x0 in self.x0s:
            results.append(
                pd.DataFrame([minimize(self.fun, x0, method="SLSQP", bounds=self.bnds, constraints=self.cons)])
            )
        self.results = pd.concat(itertools.chain([self.results], results)).reset_index(drop=True)

    def select_one_optimal(self):
        """Select ine optimal solution"""

        return self.results["x"][self.results[self.results["fun"] > 0]["fun"].idxmin()]

    def recalculate_indicators(self, population, b, G) -> dict:
        population = ceil(population)
        green = self.green_area(population + self.current_unprov_green_population, G) + self.current_green_area
        sc = school_area(self.SC_coef * (population + self.current_unprov_schoolkids))
        kg = kindergarten_area(self.KG_coef * (population + self.current_unprov_kids))

        return {
            "area": self.area,
            "population": population + self.current_population,
            "b": b * SQUARE_METERS_IN_HECTARE,
            "green_coef_G": G * SQUARE_METERS_IN_HECTARE,
            "living_area": self.living_area(population, b) + self.current_living_area,
            "schools_area": sc[0],
            "schools_capacity": sc[1],
            "kindergartens_area": kg[0],
            "kindergartens_capacity": kg[1],
            "green_area": green,
            "G_min_capacity": green / self.G_min,
            "G_max_capacity": green / self.G_max,
            "green_coef_G_capacity": green / G,
            "op_area": self.op_area(population),
            "parking1_area": self.parking1_area(population),
            "parking2_area": self.parking2_area(population),
        }

    def optimal_solution_indicators(self) -> dict:
        """
        This method selects optimal parameters for the specified area

        Returns
        -------
        Parameters: dict
        """

        self.find_optimal_solutions()
        return self.recalculate_indicators(*self.select_one_optimal())
