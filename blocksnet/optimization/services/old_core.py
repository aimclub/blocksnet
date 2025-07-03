from typing import Callable

import pandas as pd
from loguru import logger
from tqdm import tqdm

from ...analysis.provision import competitive_provision, provision_strong_total
from ...config import log_config, service_types_config
from ...enums import LandUse
from ...relations import get_accessibility_context
from ...utils.validation import validate_matrix
from .common import ServicesContainer, SimulatedAnnealing, Variable
from .schemas import BlocksSchema, ServicesSchema


class ServicesOptimizer:
    def __init__(self, blocks_df: pd.DataFrame, accessibility_matrix: pd.DataFrame):
        validate_matrix(accessibility_matrix, blocks_df)
        self.blocks_df = BlocksSchema(blocks_df)
        self.accessibility_matrix = accessibility_matrix
        self.services_containers = {}
        self.on_iteration: Callable = None

    def add_service_type(self, name: str, weight: float, services_df: pd.DataFrame):
        services_df = ServicesSchema(services_df)
        services_container = ServicesContainer(name=name, weight=weight, services_df=services_df)
        self.services_containers[name] = services_container

    def remove_service_type(self, name: str):
        del self.services_containers[name]

    @property
    def service_types(self) -> dict[str, float]:
        return {st: sc.weight for st, sc in self.services_containers.items()}

    def _get_service_types(self, land_use: LandUse) -> set[str]:
        user_service_types = set(self.service_types)
        lu_service_types = set(service_types_config[land_use])
        return user_service_types & lu_service_types

    def _get_blocks_df(self, blocks_ids: list[int]) -> pd.DataFrame:
        blocks_df = self.blocks_df.copy()
        blocks_df.loc[blocks_ids, "population"] = 0
        return blocks_df

    def _get_services_df(self, blocks_ids: list[int], service_type: str) -> pd.DataFrame:
        services_df = self.services_containers[service_type].services_df.copy()
        services_df.loc[blocks_ids, "capacity"] = 0
        return services_df

    def _initialize_provisions_dfs(self, blocks_ids: list[int]) -> dict[str, pd.DataFrame]:
        log_level = log_config.logger_level
        disable_tqdm = log_config.disable_tqdm

        logger.info(f"Initial provision assessment")

        acc_mx = self.accessibility_matrix
        blocks_df = self._get_blocks_df(blocks_ids)
        service_types = self.service_types

        provisions_dfs = {}

        for service_type in tqdm(service_types, disable=disable_tqdm):

            log_config.set_disable_tqdm(True)
            log_config.set_logger_level("ERROR")

            services_df = self._get_services_df(blocks_ids, service_type)
            _, demand, accessibility = service_types_config[service_type].values()

            provision_df, _ = competitive_provision(blocks_df.join(services_df), acc_mx, accessibility, demand)

            context_acc_mx = get_accessibility_context(acc_mx, provision_df.loc[blocks_ids], accessibility, out=False)
            provisions_dfs[service_type] = provision_df.loc[context_acc_mx.index]

            log_config.set_disable_tqdm(disable_tqdm)
            log_config.set_logger_level(log_level)

        return provisions_dfs

    def _initialize_variables(self, blocks_lus: dict[int, LandUse]) -> list[Variable]:
        units = service_types_config.units
        blocks_service_types = {block_id: self._get_service_types(lu) for block_id, lu in blocks_lus.items()}
        blocks_units = {
            block_id: units[units.service_type.isin(service_types)]
            for block_id, service_types in blocks_service_types.items()
        }
        variables = [
            Variable(block_id=block_id, **unit)
            for block_id, block_units in blocks_units.items()
            for _, unit in block_units.iterrows()
        ]
        return variables

    def _variables_to_df(self, variables: list[Variable]) -> pd.DataFrame:
        data = [v.to_dict() for v in variables]
        return pd.DataFrame(data)

    def run(
        self,
        blocks_lus: dict[int, LandUse],
        t_max: float = 100,
        t_min: float = 1e-3,
        rate: float = 0.95,
        iterations: int = 1000,
    ):
        provisions_dfs = self._initialize_provisions_dfs(list(blocks_lus))
        provisions_dfs = {
            service_type: provision_df
            for service_type, provision_df in provisions_dfs.items()
            if provision_df.demand.sum() > 0
        }

        variables = self._initialize_variables(blocks_lus)

        def update_variables(X: list[int]):
            for i, x in enumerate(X):
                variables[i].count = x

        def objective(X: list[int], i: int | None):
            update_variables(X)

            if i is not None:
                log_level = log_config.logger_level
                disable_tqdm = log_config.disable_tqdm

                log_config.set_disable_tqdm(True)
                log_config.set_logger_level("ERROR")

                variables_df = self._variables_to_df(variables)
                service_type = variables[i].service_type

                if service_type in provisions_dfs:

                    variables_df = variables_df[variables_df.service_type == service_type]
                    delta_df = variables_df.groupby("block_id").agg({"total_capacity": "sum"})

                    _, demand, accessibility = service_types_config[service_type].values()
                    old_provision_df = provisions_dfs[service_type]
                    old_provision_df.loc[delta_df.index, "capacity"] += delta_df["total_capacity"]
                    new_provision_df, _ = competitive_provision(
                        old_provision_df, self.accessibility_matrix, accessibility, demand
                    )
                    provisions_dfs[service_type] = new_provision_df

                log_config.set_disable_tqdm(disable_tqdm)
                log_config.set_logger_level(log_level)

            val = sum(
                [
                    provision_strong_total(provision_df) * self.service_types[service_type]
                    for service_type, provision_df in provisions_dfs.items()
                ]
            )
            if self.on_iteration is not None:
                self.on_iteration(val)
            return val

        def constraint(X: list[int]) -> bool:
            update_variables(X)

            for i, x in enumerate(X):
                variables[i].count = x

            if any([v.count < 0 for v in variables]):
                return False

            variables_df = self._variables_to_df(variables)
            variables_df = variables_df.groupby("block_id").agg(
                {
                    "total_site_area": "sum",
                    "total_build_floor_area": "sum",
                }
            )
            variables_df["total_area"] = variables_df["total_site_area"] + variables_df["total_build_floor_area"]
            if any(variables_df["total_area"] > self.blocks_df.loc[variables_df.index]["site_area"]):
                return False

            return True

        logger.info("Starting the optimization process")

        X = [v.count for v in variables]
        sa = SimulatedAnnealing(t_max=t_max, t_min=t_min, rate=rate, iterations=iterations)
        X, v = sa.run([v.count for v in variables], objective, constraint)
        update_variables(X)

        logger.success(f"Optimization ended with objective value of {round(v,2)}")

        return self._variables_to_df(variables), v
