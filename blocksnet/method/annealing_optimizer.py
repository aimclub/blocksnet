import random
import math
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from loguru import logger
from .base_method import BaseMethod
from ..models import Block, ServiceBrick, ServiceType, LandUse
from .provision import Provision

MOCK_POPULATION = 100


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


class AnnealingOptimizer(BaseMethod):
    @staticmethod
    def _check_constraints(X):
        return all([x.value >= 0 for x in X])

    @staticmethod
    def _perturb(X: list[Variable]):
        new_X = [Variable(x.block, x.service_type, x.brick, x.value) for x in X]
        x = random.choice(new_X)
        delta = random.choice([-1, 1])
        x.value += delta
        return new_X, x.service_type

    def _generate_initial_X(self, blocks: dict[int, LandUse], service_types: dict[str, float]) -> list[Variable]:
        X = []
        for block_id, land_use in blocks.items():
            block = self.city_model[block_id]
            user_service_types = {self.city_model[st_name] for st_name in service_types.keys()}
            lu_service_types = set(self.city_model.get_land_use_service_types(land_use))
            for service_type in user_service_types & lu_service_types:
                for brick in service_type.bricks:
                    x = Variable(block=block, service_type=service_type, brick=brick)
                    X.append(x)
        return X

    def to_gdf(self, X):
        df = self.to_df(X)
        df["geometry"] = df.apply(lambda s: self.city_model[s.name].geometry, axis=1)
        return gpd.GeoDataFrame(df, crs=self.city_model.crs)

    @staticmethod
    def to_df(X):

        service_types = {x.service_type for x in X}
        df = pd.DataFrame(
            [{"block_id": x.block.id, "population": MOCK_POPULATION, x.service_type.name: x.capacity} for x in X]
        )
        return df.groupby("block_id").agg({"population": "min", **{st.name: "sum" for st in service_types}})

    def calculate(
        self,
        blocks: dict[int, LandUse],
        service_types: dict[str, float],
        t_max: float = 100,
        t_min: float = 1e-3,
        rate: float = 0.95,
        max_iter: int = 1000,
    ) -> list[Variable]:

        logger.disable("blocksnet.method.provision")

        best_X = self._generate_initial_X(blocks, service_types)
        best_value = 0

        prov = Provision(city_model=self.city_model)

        def calculate_provision(X, service_type):
            df = self.to_df(X)
            if df[service_type.name].sum() == 0:
                return 0
            else:
                gdf = prov.calculate(service_type, df, self_supply=True)
                return prov.total_provision(gdf)

        provisions = {st: 0.0 for st in service_types.keys()}

        def objective():
            return sum([provisions[st] * w for st, w in service_types.items()])

        # Начальная температура
        T = t_max

        for _ in tqdm(range(max_iter)):

            # Генерируем новое решение
            X, st = self._perturb(best_X)

            # Проверка ограничений
            if not self._check_constraints(X):
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

        logger.enable("blocksnet.method.provision")

        return best_X, best_value
