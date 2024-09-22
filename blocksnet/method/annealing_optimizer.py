"""
Module providing class for simulated annealing optimization of blocks services constrained by land uses.
"""
import random
import math
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from typing import Callable
from .base_method import BaseMethod
from ..models import Block, ServiceBrick, ServiceType, LandUse
from .provision import Provision

VACANT_AREA_COEF = 0.8
LIVING_AREA_DEMAND = 20

# fsi is build floor area per site area
LU_FSIS = {
    LandUse.RESIDENTIAL: (0.5, 3.0),
    LandUse.BUSINESS: (1.0, 3.0),
    LandUse.RECREATION: (0.05, 0.2),
    LandUse.SPECIAL: (0.05, 0.2),
    LandUse.INDUSTRIAL: (0.3, 1.5),
    LandUse.AGRICULTURE: (0.1, 0.2),
    LandUse.TRANSPORT: (0.2, 1.0),
}

# gsi is footprint area per site area
LU_GSIS = {
    LandUse.RESIDENTIAL: (0.2, 0.8),
    LandUse.BUSINESS: (0.0, 0.8),
    LandUse.RECREATION: (0.0, 0.3),
    LandUse.SPECIAL: (0.05, 0.15),
    LandUse.INDUSTRIAL: (0.2, 0.8),
    LandUse.AGRICULTURE: (0.0, 0.6),
    LandUse.TRANSPORT: (0.0, 0.8),
}


class Variable:
    """
    Represents a variable for optimization in a given block with a specific service type and brick.

    Attributes
    ----------
    block : Block
        The block where the service is located.
    service_type : ServiceType
        The type of service provided.
    brick : ServiceBrick
        The brick representing the service's characteristics.
    value : int, optional
        The quantity of the service brick, by default 0.
    """

    def __init__(self, block: Block, service_type: ServiceType, brick: ServiceBrick, value: int = 0):
        """
        Initializes a Variable instance.

        Parameters
        ----------
        block : Block
            The block where the service is located.
        service_type : ServiceType
            The type of service provided.
        brick : ServiceBrick
            The brick representing the service's characteristics.
        value : int, optional
            The quantity of the service brick, by default 0.
        """
        self.block = block
        self.service_type = service_type
        self.brick = brick
        self.value = value

    @property
    def capacity(self):
        """
        Calculates the total capacity of the service brick in the block.

        Returns
        -------
        int
            The total capacity of the service brick.
        """
        return self.brick.capacity * self.value

    @property
    def area(self):
        """
        Calculates the total area occupied by the service brick.

        Returns
        -------
        int
            The total area occupied by the service brick.
        """
        return self.brick.area * self.value

    def to_dict(self):
        """
        Converts the Variable instance to a dictionary representation.

        Returns
        -------
        dict
            A dictionary containing the block ID, service type, brick capacity, brick area, whether the brick is integrated,
            and the value of the brick.
        """
        return {
            "block_id": self.block.id,
            "service_type": self.service_type.name,
            "capacity": self.brick.capacity,
            "area": self.brick.area,
            "is_integrated": self.brick.is_integrated,
            "value": self.value,
        }


class Indicator:
    """
    Represents indicators for a block based on land use and spatial indices.

    Attributes
    ----------
    block : Block
        The block being evaluated.
    land_use : LandUse
        The type of land use for the block.
    fsi : float
        The Floor Space Index (FSI) of the block.
    gsi : float
        The Ground Space Index (GSI) of the block.
    """

    def __init__(self, block: Block, land_use: LandUse, fsi: float, gsi: float):
        """
        Initializes an Indicator instance.

        Parameters
        ----------
        block : Block
            The block being evaluated.
        land_use : LandUse
            The type of land use for the block.
        fsi : float
            The Floor Space Index (FSI) of the block.
        gsi : float
            The Ground Space Index (GSI) of the block.
        """
        self.block = block
        self.land_use = land_use
        self.fsi = fsi
        self.gsi = gsi

    @property
    def site_area(self):
        """
        Gets the site area of the block.

        Returns
        -------
        float
            The site area of the block.
        """
        return self.block.site_area

    @property
    def footprint_area(self):
        """
        Calculates the footprint area of the block based on the GSI.

        Returns
        -------
        float
            The footprint area of the block.
        """
        return self.site_area * self.gsi

    @property
    def build_floor_area(self):
        """
        Calculates the buildable floor area of the block based on the FSI.

        Returns
        -------
        float
            The buildable floor area of the block.
        """
        return self.site_area * self.fsi

    @property
    def integrated_area(self):
        """
        Determines the integrated area of the block based on land use.

        Returns
        -------
        float
            The integrated area of the block.
        """
        if self.land_use == LandUse.RESIDENTIAL:
            return self.footprint_area
        else:
            return self.build_floor_area

    @property
    def non_integrated_area(self):
        """
        Calculates the non-integrated area of the block.

        Returns
        -------
        float
            The non-integrated area of the block.
        """
        return VACANT_AREA_COEF * self.site_area - self.footprint_area

    @property
    def living_area(self):
        """
        Determines the living area of the block based on land use.

        Returns
        -------
        float
            The living area of the block. Returns 0 if land use is not residential.
        """
        if self.land_use == LandUse.RESIDENTIAL:
            return self.build_floor_area - self.integrated_area
        else:
            return 0

    @property
    def population(self):
        """
        Estimates the population of the block based on living area and demand.

        Returns
        -------
        int
            The estimated population of the block.
        """
        return math.floor(self.living_area / LIVING_AREA_DEMAND)

    def to_dict(self) -> dict:
        """
        Converts the Indicator instance to a dictionary representation.

        Returns
        -------
        dict
            A dictionary containing block ID, site area, land use, footprint area, build floor area, integrated area,
            non-integrated area, living area, and population.
        """
        res = {
            "block_id": self.block.id,
            "site_area": self.site_area,
            "land_use": self.land_use.value,
            "footprint_area": self.footprint_area,
            "build_floor_area": self.build_floor_area,
            "integrated_area": self.integrated_area,
            "non_integrated_area": self.non_integrated_area,
            "living_area": self.living_area,
            "population": self.population,
        }
        return {key: round(value, 2) if isinstance(value, float) else value for key, value in res.items()}


class AnnealingOptimizer(BaseMethod):
    """
    Services optimizer using simulated annealing to maximize scenario provision objective function by building certain services.

    Attributes
    ----------
    on_iteration : Callable[[int, list[Variable], dict[int, Indicator], float], None] | None
        A callback function to be called after each iteration with the current iteration number,
        the current list of variables, indicators, and the best value found so far.
    """

    on_iteration: Callable[[int, list[Variable], dict[int, Indicator], float], None] | None = None

    def _check_constraints(self, X, indicators) -> bool:
        """
        Checks if the current solution satisfies all constraints.

        Parameters
        ----------
        X : list[Variable]
            List of variables representing the current solution.
        indicators : dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.
        """
        df = self.to_bricks_df(X)
        df["integrated_area"] = df.apply(lambda s: s["area"] * s["count"] if s["is_integrated"] else 0, axis=1)
        df["non_integrated_area"] = df.apply(lambda s: s["area"] * s["count"] if not s["is_integrated"] else 0, axis=1)

        if any(df["count"] < 0):
            return False

        df = df.groupby("block_id").agg({"integrated_area": "sum", "non_integrated_area": "sum"})

        for block_id in df.index:

            indicator = indicators[block_id]

            integrated_area = indicator.integrated_area
            if df.loc[block_id, "integrated_area"] > integrated_area:
                return False

            if df.loc[block_id, "non_integrated_area"] > integrated_area:
                return False
        return True

    @staticmethod
    def _perturb(X: list[Variable]) -> tuple[list[Variable], ServiceType]:
        """
        Perturbs the current solution by modifying a randomly chosen variable.

        Parameters
        ----------
        X : list[Variable]
            List of variables representing the current solution.

        Returns
        -------
        tuple[list[Variable], ServiceType]
            A tuple containing the perturbed list of variables and the service type of the perturbed variable.
        """
        new_X = [Variable(x.block, x.service_type, x.brick, x.value) for x in X]
        x = random.choice(new_X)
        delta = random.choice([-1, 1])
        x.value += delta
        return new_X, x.service_type

    def _generate_initial_X(self, blocks_lu: dict[int, LandUse], service_types: dict[str, float]) -> list[Variable]:
        """
        Generates an initial list of variables for the optimization.

        Parameters
        ----------
        blocks_lu : dict[int, LandUse]
            Dictionary mapping block IDs to their corresponding land uses.
        service_types : dict[str, float]
            Dictionary mapping service type names to their weights.

        Returns
        -------
        list[Variable]
            A list of variables representing the initial solution.
        """
        X = []
        for block_id, land_use in blocks_lu.items():
            block = self.city_model[block_id]
            user_service_types = {self.city_model[st_name] for st_name in service_types.keys()}
            lu_service_types = set(self.city_model.get_land_use_service_types(land_use))
            for service_type in user_service_types & lu_service_types:
                for brick in service_type.bricks:
                    x = Variable(block=block, service_type=service_type, brick=brick)
                    X.append(x)
        return X

    def _generate_indicators(self, blocks, fsis, gsis) -> dict[int, Indicator]:
        """
        Generates indicators for each block based on the provided data.

        Parameters
        ----------
        blocks : dict[int, LandUse]
            Dictionary mapping block IDs to their corresponding land uses.
        fsis : dict[int, float]
            Dictionary mapping block IDs to their Floor Space Index (FSI) values.
        gsis : dict[int, float]
            Dictionary mapping block IDs to their Ground Space Index (GSI) values.

        Returns
        -------
        dict[int, Indicator]
            A dictionary mapping block IDs to their corresponding indicators.
        """
        return {b_id: Indicator(self.city_model[b_id], blocks[b_id], fsis[b_id], gsis[b_id]) for b_id in blocks.keys()}

    def to_gdf(self, X: list[Variable], indicators: dict[int, Indicator]) -> gpd.GeoDataFrame:
        """
        Converts the list of variables and indicators into a GeoDataFrame.

        Parameters
        ----------
        X : list[Variable]
            List of variables representing the current solution.
        indicators : dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing optimization result data.
        """
        df = self.to_df(X, indicators)
        df["geometry"] = df.apply(lambda s: self.city_model[s.name].geometry, axis=1)
        df["land_use"] = df.apply(lambda s: indicators[s.name].land_use.value, axis=1)
        return gpd.GeoDataFrame(df, crs=self.city_model.crs)

    def to_bricks_df(self, X: list[Variable]) -> pd.DataFrame:
        """
        Converts the list of variables into a DataFrame representing the bricks.

        Parameters
        ----------
        X : list[Variable]
            List of variables representing the current solution.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing data about the bricks in the solution.
        """
        xs = [
            {
                "block_id": x.block.id,
                "service_type": x.service_type.name,
                "is_integrated": x.brick.is_integrated,
                "area": x.brick.area,
                "capacity": x.brick.capacity,
                "count": x.value,
            }
            for x in X
        ]
        df = pd.DataFrame(list(xs))
        return df[df["count"] != 0]

    def _get_clear_df(self, blocks: list[int]) -> pd.DataFrame:
        """
        Constructs a DataFrame for provision assessment so the blocks being changed are treated as cleared.

        Parameters
        ----------
        blocks : list[int]
            List of changing blocks ids.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information related to blocks being changed for provision assessment.
        """
        gdf = self.city_model.get_blocks_gdf()
        gdf = gdf[gdf.index.isin(blocks)]
        df = gdf[["population"]].copy()
        df["population"] = -df["population"]
        df.sum()
        for column in [column for column in gdf.columns if "capacity_" in column]:
            st_name = column.removeprefix("capacity_")
            df[st_name] = -gdf[column]
        return df

    def to_df(self, X: list[Variable], indicators: dict[int, Indicator]) -> pd.DataFrame:
        """
        Converts the list of variables and indicators into a DataFrame.

        Parameters
        ----------
        X : list[Variable]
            List of variables representing the current solution.
        indicators : dict[int, Indicator]
            Dictionary mapping block IDs to their corresponding indicators.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing data about the blocks and their service capacities.
        """
        service_types = {x.service_type for x in X}
        df = pd.DataFrame(
            [
                {
                    "block_id": x.block.id,
                    "population": indicators[x.block.id].population,
                    x.service_type.name: x.capacity,
                }
                for x in X
            ]
        )
        return df.groupby("block_id").agg({"population": "min", **{st.name: "sum" for st in service_types}})

    def calculate(
        self,
        blocks_lu: dict[int, LandUse],
        blocks_fsi: dict[int, float],
        blocks_gsi: dict[int, float],
        service_types: dict[str, float],
        t_max: float = 100,
        t_min: float = 1e-3,
        rate: float = 0.95,
        max_iter: int = 1000,
    ) -> tuple:
        """
        Executes the services optimization process based on simulated annealing for provided service types weights and blocks landuses.

        Parameters
        ----------
        blocks_lu : dict[int, LandUse]
            Dictionary mapping block IDs to their corresponding land uses.
        blocks_fsi : dict[int, float]
            Dictionary mapping block IDs to their Floor Space Index (FSI) values.
        blocks_gsi : dict[int, float]
            Dictionary mapping block IDs to their Ground Space Index (GSI) values.
        service_types : dict[str, float]
            Dictionary mapping service type names to their weights.
        t_max : float, optional
            The maximum temperature for the annealing process, by default 100.
        t_min : float, optional
            The minimum temperature for the annealing process, by default 1e-3.
        rate : float, optional
            The cooling rate of the temperature, by default 0.95.
        max_iter : int, optional
            The maximum number of iterations, by default 1000.

        Returns
        -------
        tuple
            A tuple containing the best solution found, the indicators, the best value, and the provision values for each service type.
        """
        indicators = self._generate_indicators(blocks_lu, blocks_fsi, blocks_gsi)

        best_X = self._generate_initial_X(blocks_lu, service_types)
        best_value = 0

        clear_df = self._get_clear_df(blocks_lu.keys())

        prov = Provision(city_model=self.city_model, verbose=False)

        def calculate_provision(X, service_type):
            update_df = clear_df.add(self.to_df(X, indicators))
            if update_df[service_type.name].sum() == 0:
                return 0
            gdf = prov.calculate(service_type, update_df, self_supply=True)
            return prov.total(gdf)

        provisions = {st: 0.0 for st in service_types.keys()}

        def objective():
            return sum(provisions[st] * w for st, w in service_types.items())

        # Начальная температура
        T = t_max

        if self.verbose:
            pbar = tqdm(range(max_iter))

        for iteration in range(max_iter):

            if self.verbose:
                pbar.update(1)
                pbar.set_description(f"Value : {round(best_value,3)}")

            if self.on_iteration is not None:
                self.on_iteration(iteration, best_X, indicators, best_value)

            # Генерируем новое решение
            X, st = self._perturb(best_X)

            # Проверка ограничений
            if not self._check_constraints(X, indicators):
                continue

            # recalculate changed provision
            provisions[st.name] = calculate_provision(X, st)

            # Вычисляем значение целевой функции
            value = objective()

            # Если новое решение лучше, принимаем его
            if value > best_value:
                best_value = value
                best_X = X
            else:
                # Принимаем худшее решение с вероятностью, зависящей от температуры
                delta = value - best_value
                if random.random() < math.exp(delta / T):
                    best_value = value
                    best_X = X

            # Охлаждаем температуру
            T = T * rate
            if T < t_min:
                break

        return best_X, indicators, best_value, provisions
