import pandas as pd
from math import ceil
from scipy.optimize import minimize


def kindergarten_area_ranges(kids):
    kids = ceil(kids)
    if kids in range(140, 180):
        return (0.72, 180)
    elif kids in range(180, 281):
        return (1.1, 280)
    else:
        return (0, 0)


def kindergarten_area(kids):
    if kids >= 280:
        return tuple(map(sum, zip(kindergarten_area_ranges(280), kindergarten_area(kids - 280))))
    else:
        return kindergarten_area_ranges(kids)


def school_area_ranges(schoolkids):
    schoolkids = ceil(schoolkids)
    if schoolkids in range(100, 250):
        return (1.2, 250)
    elif schoolkids in range(250, 300):
        return (1.1, 300)
    elif schoolkids in range(300, 600):
        return (1.3, 600)
    elif schoolkids in range(600, 800):
        return (1.5, 800)
    elif schoolkids in range(800, 1101):
        return (1.8, 1100)
    else:
        return (0, 0)


def school_area(schoolkids):
    if schoolkids >= 1100:
        return tuple(map(sum, zip(school_area_ranges(1100), school_area(schoolkids - 1100))))
    else:
        return school_area_ranges(schoolkids)


Hectare = 10000


class MasterPlan:
    def __init__(
        self,
        area,
        current_living_area=None,
        current_industrial_area=None,
        current_population=None,
        current_green_area=None,
        current_op_area=None,
        current_unprov_schoolkids=None,
        current_unprov_kids=None,
        current_unprov_green_population=None,
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
        self.b_min, self.b_max = 18 / Hectare, 30 / Hectare
        self.G_min, self.G_max = 6 / Hectare, 12 / Hectare

        self.SC_coef = 0.12
        self.KG_coef = 0.061
        self.OP_coef = 0.03 / Hectare

        self.P1_coef = 0.42 * 0.15 * 0.012
        self.P2_coef = 0.42 * 0.35 * 0.005

        self.max_building_area = self.area * self.BA_coef - self.current_living_area - self.current_industrial_area
        self.max_free_area = self.area * self.FA_coef - self.current_green_area - self.current_op_area

        self.max_living_area = self.max_building_area * self.LA_coef - self.current_living_area
        self.max_industrial_area = self.max_building_area * self.IA_coef - self.current_industrial_area

        self.max_living_area_full = self.max_living_area * self.F_max
        self.max_population = ceil(self.max_living_area_full / self.b_max) - self.current_population

    def sc_area(self, population):
        return school_area(self.SC_coef * population)[0]

    def kg_area(self, population):
        return kindergarten_area(self.KG_coef * population)[0]

    def op_area(self, population):
        return self.OP_coef * population

    def parking1_area(self, population):
        return self.P1_coef * population

    def parking2_area(self, population):
        return self.P2_coef * population

    @staticmethod
    def living_area(population, b):
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
            {"type": "ineq", "fun": lambda x: self.fun(x)},
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

    def func(self):
        self.bnds_and_cons()
        self.make_x0s()

        for x0 in self.x0s:
            temp_result = minimize(self.fun, x0, method="SLSQP", bounds=self.bnds, constraints=self.cons)
            self.results = self.results.append(temp_result, ignore_index=True)

        del temp_result

    def select_optimal(self):
        return self.results["x"][self.results[self.results["fun"] > 0]["fun"].idxmin()]

    def optimal_values(self, population, b, G):
        population = ceil(population)
        green = self.green_area(population + self.current_unprov_green_population, G) + self.current_green_area
        sc = school_area(self.SC_coef * (population + self.current_unprov_schoolkids))
        kg = kindergarten_area(self.KG_coef * (population + self.current_unprov_kids))

        return {
            "area": self.area,
            "population": population + self.current_population,
            "b": b * Hectare,
            "green_coef_G": G * Hectare,
            "living_area": self.living_area(population, b) + self.current_living_area,
            "school_area": sc[0],
            "school_capacity": sc[1],
            "kindergarten_area": kg[0],
            "kindergarten_capacity": kg[1],
            "green_area": green,
            "G_min_capacity": green / self.G_min,
            "G_max_capacity": green / self.G_max,
            "green_coef_G_capacity": green / G,
            "op_area": self.op_area(population),
            "parking1_area": self.parking1_area(population),
            "parking2_area": self.parking2_area(population),
        }
