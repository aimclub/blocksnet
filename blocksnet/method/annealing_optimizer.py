import random
import math
import geopandas as gpd
from loguru import logger
import pandas as pd
from tqdm import tqdm
from typing import Callable
from .base_method import BaseMethod
from ..models import Block, ServiceBrick, ServiceType, LandUse
from .provision import Provision

VACANT_AREA_COEF = 0.8
LIVING_AREA_DEMAND = 20


class Variable:
    def __init__(self, block: Block, service_type: ServiceType, brick: ServiceBrick, value: int = 0):
        self.block = block
        self.service_type = service_type
        self.brick = brick
        self.value = value

    @property
    def capacity(self):
        return self.brick.capacity * self.value

    @property
    def area(self):
        return self.brick.area * self.value

    def to_dict(self):
        return {
            "block_id": self.block.id,
            "service_type": self.service_type.name,
            "capacity": self.brick.capacity,
            "area": self.brick.area,
            "is_integrated": self.brick.is_integrated,
            "value": self.value,
        }


class Indicator:
    def __init__(self, block: Block, land_use: LandUse, fsi: float, gsi: float):
        self.block = block
        self.land_use = land_use
        self.fsi = fsi
        self.gsi = gsi

    @property
    def site_area(self):
        return self.block.site_area

    @property
    def footprint_area(self):
        return self.site_area * self.gsi

    @property
    def build_floor_area(self):
        return self.site_area * self.fsi

    @property
    def integrated_area(self):
        if self.land_use == LandUse.RESIDENTIAL:
            return self.footprint_area
        else:
            return self.build_floor_area

    @property
    def non_integrated_area(self):
        return VACANT_AREA_COEF * self.site_area - self.footprint_area

    @property
    def living_area(self):
        if self.land_use == LandUse.RESIDENTIAL:
            return self.build_floor_area - self.integrated_area
        else:
            return 0

    @property
    def population(self):
        return math.floor(self.living_area / LIVING_AREA_DEMAND)

    def to_dict(self) -> dict:
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

    on_iteration: Callable[[int, list[Variable], dict[int, Indicator], float], None] | None = None

    def _check_constraints(self, X, indicators):

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
    def _perturb(X: list[Variable]):
        new_X = [Variable(x.block, x.service_type, x.brick, x.value) for x in X]
        x = random.choice(new_X)
        delta = random.choice([-1, 1])
        x.value += delta
        return new_X, x.service_type

    def _generate_initial_X(self, blocks_lu: dict[int, LandUse], service_types: dict[str, float]) -> list[Variable]:
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
        return {b_id: Indicator(self.city_model[b_id], blocks[b_id], fsis[b_id], gsis[b_id]) for b_id in blocks.keys()}

    def to_gdf(self, X: list[Variable], indicators: dict[int, Indicator]):
        df = self.to_df(X, indicators)
        df["geometry"] = df.apply(lambda s: self.city_model[s.name].geometry, axis=1)
        df["land_use"] = df.apply(lambda s: indicators[s.name].land_use.value, axis=1)
        return gpd.GeoDataFrame(df, crs=self.city_model.crs)

    def to_bricks_df(self, X: list[Variable]):
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
        # xs = filter(lambda v : v['count']>0, xs)
        return pd.DataFrame(list(xs))

    def to_df(self, X: list[Variable], indicators: dict[int, Indicator]):
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

        indicators = self._generate_indicators(blocks_lu, blocks_fsi, blocks_gsi)

        best_X = self._generate_initial_X(blocks_lu, service_types)
        best_value = 0

        prov = Provision(city_model=self.city_model, verbose=False)

        def calculate_provision(X, service_type):
            df = self.to_df(X, indicators)
            if df[service_type.name].sum() == 0:
                return 0
            gdf = prov.calculate(service_type, df, self_supply=True)
            return prov.total(gdf)

        provisions = {st: 0.0 for st in service_types.keys()}

        def objective():
            return sum(provisions[st] * w for st, w in service_types.items())

        # Начальная температура
        T = t_max

        for iteration in tqdm(range(max_iter), disable=(not self.verbose)):

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
