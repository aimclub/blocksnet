import geopandas as gpd
import pandas as pd
from ..base_method import BaseMethod
from ...models import Block
from ..provision import Provision
from pydantic import Field, InstanceOf
import itertools
import pygad
from ...utils import SQUARE_METERS_IN_HECTARE


class Genetic(BaseMethod):
    BLOCKS: InstanceOf[pd.DataFrame] = pd.DataFrame()
    PROVISION: InstanceOf[Provision] = None
    SCENARIO: dict
    BUILDING_OPTIONS: InstanceOf[pd.DataFrame] = pd.DataFrame()
    GA_PARAMS: dict = {
        "num_generations": 2,
        "num_parents_mating": 6,
        "sol_per_pop": 10,
        "mutation_type": "adaptive",
        "mutation_percent_genes": (90, 10),
        "crossover_type": "scattered",
        "parent_selection_type": "tournament",
        "K_tournament": 3,
        "stop_criteria": "saturate_50",
        "parallel_processing": 12,
        "keep_parents": 1,
    }

    def flatten_dict(self, services: dict) -> tuple[dict[str, float], pd.DataFrame]:
        """Utility function for get flatten dictionary of services requriments"""
        services_dict = {}
        for service, requirements in services.items():
            if service not in self.SCENARIO:
                continue
            for population, area in requirements.items():
                services_dict[service + "_" + str(population)] = area
        return services_dict, pd.DataFrame([services_dict]).T

    def get_combinations(self, services_dict: dict, comb_len: int) -> list:
        """Determination of all possible combinations of services in the blocks, depending on the scenario"""
        return [
            item
            for sublist in [list(itertools.combinations(list(services_dict.keys()), i)) for i in range(1, comb_len + 1)]
            for item in sublist
        ]

    def get_combinations_area(self, combinations: list, services_df: pd.DataFrame) -> list:
        """Calculation area of services for combinations"""
        return [services_df.loc[list(combination)].sum()[0] for combination in combinations]

    def updating_blocks_combinations(self, combinations_weights):
        """Updating the block dataframe with possible combinations depending on the free area"""
        self.BLOCKS["variants"] = self.BLOCKS["free_area"].apply(
            lambda free_area: [
                i for i, combination_weight in enumerate(combinations_weights) if combination_weight <= free_area
            ]
        )

    def get_building_options(self, combinations: list):
        """Filtering unsuitable combinations and updating all possible"""
        total_building_options = list(set([item for sublist in self.BLOCKS["variants"].tolist() for item in sublist]))
        combinations = [combinations[i] for i in total_building_options]
        self.BUILDING_OPTIONS = pd.DataFrame(
            columns=list(self.SCENARIO.keys()), index=range(len(total_building_options))
        ).fillna(0)
        for i, _ in enumerate(combinations):
            for el in [x.rsplit("_", maxsplit=1) for x in _]:
                self.BUILDING_OPTIONS.loc[i, el[0]] += int(el[1])
        self.BUILDING_OPTIONS.index = total_building_options

    def get_updated_blocks(self, building_options_ids, blocks_ids=None):
        """Get updated blocks with calculated provision"""
        updated_blocks = self.BUILDING_OPTIONS.loc[building_options_ids]
        if blocks_ids:
            updated_blocks.index = blocks_ids
        else:
            updated_blocks.index = self.BLOCKS["id"]

        return updated_blocks.to_dict("index")

    def fitness_func(self, ga_instance, solution, solution_idx):
        """Fitness function for genetic algorithm"""
        updated_blocks = pd.DataFrame.from_dict((self.get_updated_blocks(solution)), orient="index")
        _, fitness = self.PROVISION.calculate_scenario(self.SCENARIO, updated_blocks)
        return fitness

    def get_blocks(self, selected_blocks: list[Block]):
        data = []
        for block in selected_blocks:
            data.append({k: v for k, v in block})
        gdf = gpd.GeoDataFrame(data).set_index("id").set_crs(epsg=self.city_model.epsg)
        gdf["free_area"] = (
            gdf["area"] * 0.8 - gdf["green_area"] - gdf["industrial_area"] - gdf["living_area"]
        ) / SQUARE_METERS_IN_HECTARE
        return gdf.reset_index()

    @property
    def ga_params(self):
        return {
            "fitness_func": self.fitness_func,
            "num_genes": self.BLOCKS.shape[0],
            "gene_space": self.BLOCKS["variants"].tolist(),
            "gene_type": int,
            **self.GA_PARAMS,
        }

    def calculate(
        self, services: dict, comb_len: int, selected_blocks: list[Block] | list[int] = None
    ) -> gpd.GeoDataFrame:
        """Calculation of the optimal development option by services for blocks"""
        if selected_blocks is not None:
            selected_blocks = map(lambda b: b if isinstance(b, Block) else self.city_model[b], selected_blocks)
        self.BLOCKS = self.get_blocks(selected_blocks if selected_blocks is not None else self.city_model.blocks)
        self.PROVISION = Provision(city_model=self.city_model)
        services_dict, services_df = self.flatten_dict(services)
        combinations = self.get_combinations(services_dict, comb_len)
        combinations_area = self.get_combinations_area(combinations, services_df)
        self.updating_blocks_combinations(combinations_area)
        self.BLOCKS = self.BLOCKS[self.BLOCKS["variants"].apply(lambda x: len(x)) != 0]
        self.get_building_options(combinations)

        self.GA_PARAMS = self.ga_params
        ga_instance = pygad.GA(**self.GA_PARAMS)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        updated_blocks = self.get_updated_blocks(solution)
        return updated_blocks
