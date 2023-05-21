"""
This module contains methods 
for calculating a new feasibility profile for the selected area. 
The maximisation parameter is the number of inhabitants.
"""


from math import ceil
import pandas as pd
import numpy as np
from scipy.optimize import minimize


def kindergarten_area_ranges(kids) -> tuple:
    """
    This method select weight coefficient according to the kindergarten area

    Attributes
    ----------
    kids: int
        The number of kids in the school

    Returns
    -------
    Weight coefficient
    """

    conditions = [140 < kids < 180, 250 < kids < 280]

    choices = [(0.72, 180), (1.1, 280)]
    return np.select(conditions, choices, default=(0, 0))


def kindergarten_area(kids) -> tuple:
    """
    This method select weight coefficient according to the number of children in kindergarten

    Attributes
    ----------
    kids: int
        The number of kids in the school

    Returns
    -------
    weights: tuple
        The weight coefficient for the specified number of kids
    """

    if kids >= 280:
        return tuple(map(sum, zip(kindergarten_area_ranges(280), kindergarten_area(kids - 280))))
    else:
        return kindergarten_area_ranges(kids)


def school_area_ranges(schoolkids) -> tuple:
    """
    This method select weight coefficient according to the number of schoolchildren

    Attributes
    ----------
    schoolkids: int
        The number of schoolkids in the school

    Returns
    -------
    Weight coefficient
    """

    schoolkids = ceil(schoolkids)

    conditions = [
        100 < schoolkids < 250,
        250 < schoolkids < 300,
        300 < schoolkids < 600,
        600 < schoolkids < 800,
        800 < schoolkids < 1101,
    ]

    choices = [(1.2, 250), (1.1, 300), (1.3, 600), (1.5, 800), (1.8, 1100)]
    return np.select(conditions, choices, default=(0, 0))


def school_area(schoolkids) -> tuple:
    """
    This method select weight coefficient according to the number of schoolkids in school

    Attributes
    ----------
    schoolkids: int
        The number of kids in the school

    Returns
    -------
    weights: tuple
        The weight coefficient for the specified number of kids
    """

    if schoolkids >= 1100:
        return tuple(map(sum, zip(school_area_ranges(1100), school_area(schoolkids - 1100))))
    else:
        return school_area_ranges(schoolkids)


class MasterPlan:
    """
    class MasterPlan is aimed to calculate balanced parameters for masterplanning for the specified area
    """

    def __init__(
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
        self.Hectare = 10000

        self.area = area

        self.BA_coef = 0.8
        self.FA_coef = 0.2
        self.LA_coef = 0.7
        self.IA_coef = 0.3

        self.F_max = 9
        self.b_min, self.b_max = 18 / self.Hectare, 30 / self.Hectare
        self.G_min, self.G_max = 6 / self.Hectare, 12 / self.Hectare

        self.SC_coef = 0.12
        self.KG_coef = 0.061
        self.OP_coef = 0.03 / self.Hectare

        self.P1_coef = 0.42 * 0.15 * 0.012
        self.P2_coef = 0.42 * 0.35 * 0.005

        self.max_building_area = self.area * self.BA_coef - self.current_living_area - self.current_industrial_area
        self.max_free_area = self.area * self.FA_coef - self.current_green_area - self.current_op_area

        self.max_living_area = self.max_building_area * self.LA_coef
        self.max_industrial_area = self.max_building_area * self.IA_coef

        self.max_living_area_full = self.max_living_area * self.F_max
        self.max_population = ceil(self.max_living_area_full / self.b_max)

    def sc_area(self, population) -> float:
        """
        Method calculates coefficients for school area based on the number of inhabitants

        Attributes
        ----------
        population: int
            maximized number of inhabitants
        
        Returns
        -------
        Weight coefficient
        """

        return school_area(self.SC_coef * population)[0]

    def kg_area(self, population) -> float:
        """
        Method calculates coefficients for kindergarten area based on the number of inhabitants

        Attributes
        ----------
        population: int
            maximized number of inhabitants
        
        Returns
        -------
        Weight coefficient
        """

        return kindergarten_area(self.KG_coef * population)[0]

    def op_area(self, population) -> float:
        """
        Method calculates coefficients for public spaces based on the number of inhabitants

        Attributes
        ----------
        population: int
            maximized number of inhabitants
        
        Returns
        -------
        Weight coefficient
        """

        return self.OP_coef * population

    def parking1_area(self, population) -> float:
        """
        Method calculates coefficients №1 for parking area based on the number of inhabitants

        Attributes
        ----------
        population: int
            maximized number of inhabitants
        
        Returns
        -------
        Weight coefficient
        """
                
        return self.P1_coef * population

    def parking2_area(self, population):
        """
        Method calculates coefficients №2 for parking area based on the number of inhabitants

        Attributes
        ----------
        population: int
            maximized number of inhabitants
        
        Returns
        -------
        Weight coefficient
        """

        return self.P2_coef * population

    @staticmethod
    def living_area(population, b):
        """
        Method calculates coefficients №1 for parking area based on the number of inhabitants

        Attributes
        ----------
        population: int
            maximized number of inhabitants
        
        Returns
        -------
        Weight coefficient
        """

        return population * b

    @staticmethod
    def green_area(population, G):
        """
        Method calculates coefficients for green area based on the number of inhabitants

        Attributes
        ----------
        population: int
            maximized number of inhabitants
        
        Returns
        -------
        Weight coefficient
        """
                
        return population * G

    def fun(self, x):
        """
        Method calculates coefficients №1 for parking area based on the number of inhabitants

        Attributes
        ----------
        Parameters for the selected variable: tuple
            population and 
        
        Returns
        -------
        Integral weight coefficient
        """

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
        """Information adout this method will be provided later"""
        
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
            {"type": "ineq", "fun": lambda x: self.fun(x)},
        )

        self.bnds = ((0, self.max_population), (self.b_min, self.b_max), (self.G_min, self.G_max))

    def make_x0s(self):
        """Information adout this method will be provided later"""

        self.x0s = [
            (0, 0, 0),
            (1 / self.max_population, self.b_min, self.G_min),
            (self.max_population / 32, (self.b_min + self.b_max) / 2, (self.G_min + self.G_max) / 2),
            (self.max_population / 16, (self.b_min + self.b_max) / 2, (self.G_min + self.G_max) / 2),
            (self.max_population / 8, self.b_max, self.G_max),
        ]

    def find_optimal_solutions(self):
        """Information adout this method will be provided later"""

        self.bnds_and_cons()
        self.make_x0s()

        for x0 in self.x0s:
            temp_result = minimize(self.fun, x0, method="SLSQP", bounds=self.bnds, constraints=self.cons)
            self.results = self.results.append(temp_result, ignore_index=True)

        del temp_result

    def select_one_optimal(self):
        """Information adout this method will be provided later"""
        
        return self.results["x"][self.results[self.results["fun"] > 0]["fun"].idxmin()]

    def recalculate_indicators(self, population, b, G) -> dict:
        """
        This method calculates desired indicator based on the number of inhabitants

        Attributes
        ----------
        population: int

        Returns
        -------
        Block's new parameters
            new parameters for the feasibility report
        """
        
        population = ceil(population)
        green = self.green_area(population + self.current_unprov_green_population, G) + self.current_green_area
        sc = school_area(self.SC_coef * (population + self.current_unprov_schoolkids))
        kg = kindergarten_area(self.KG_coef * (population + self.current_unprov_kids))

        return {
            "area": self.area,
            "population": population + self.current_population,
            "b": b * self.Hectare,
            "green_coef_G": G * self.Hectare,
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
