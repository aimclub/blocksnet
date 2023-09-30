import os
import pandas as pd
import geopandas as gpd
import numpy as np
from blocksnet import CityModel
from blocksnet.method.provision import LpProvision
import itertools
import pygad


class Genetic:
    """The class provides a method for calculating the optimal options for the development
    of a territory (a set of services) for given blocks and a scenario using a genetic algorithm"""

    def __init__(self, CITY_MODEL, BLOCKS, SERVICES, SCENARIO, COMBINATION_SUBSEQ_LEN=None):
        self.CITY_MODEL = CITY_MODEL
        self.BLOCKS = BLOCKS
        self.SERVICES = SERVICES
        self.SCENARIO = SCENARIO
        self.SERVICES_DF = pd.DataFrame()
        self.SERVICES_DICT = {}
        self.COMBINATION_SUBSEQ_LEN = COMBINATION_SUBSEQ_LEN if COMBINATION_SUBSEQ_LEN is not None else 3
        self.LPP = LpProvision(city_model=self.CITY_MODEL)

    def flatten_dict(self):
        """Utility function for get flatten dictionary of services requriments"""
        for service, requirements in self.SERVICES.items():
            if service not in self.SCENARIO:
                continue
            for population, area in requirements.items():
                self.SERVICES_DICT[service + "_" + str(population)] = area
        self.SERVICES_DF = pd.DataFrame([self.SERVICES_DICT]).T

    def get_combinations(self):
        """Determination of all possible combinations of services in the blocks, depending on the scenario"""
        combinations = [
            list(itertools.combinations(list(self.SERVICES_DICT.keys()), i))
            for i in range(1, self.COMBINATION_SUBSEQ_LEN + 1)
        ]
        self.COMBINATIONS = [item for sublist in combinations for item in sublist]

    def get_combinations_area(self):
        """Calculation area of services for combinations"""
        self.COMBINATIONS_WEIGHTS = []
        for combination in self.COMBINATIONS:
            self.COMBINATIONS_WEIGHTS.append(self.SERVICES_DF.loc[list(combination)].sum()[0])

    def updating_blocks_combinations(self):
        """Updating the block dataframe with possible combinations depending on the free area"""
        self.BLOCKS["variants"] = self.BLOCKS["free_area"].apply(
            lambda free_area: [
                i for i, combination_weight in enumerate(self.COMBINATIONS_WEIGHTS) if combination_weight <= free_area
            ]
        )

    def get_building_options(self):
        """Filtering unsuitable combinations and updating all possible"""
        total_building_options = list(set([item for sublist in self.BLOCKS["variants"].tolist() for item in sublist]))
        self.COMBINATIONS = [self.COMBINATIONS[i] for i in total_building_options]
        self.BUILDING_OPTIONS = pd.DataFrame(
            columns=list(self.SCENARIO.keys()), index=range(len(total_building_options))
        ).fillna(0)
        for i, _ in enumerate(self.COMBINATIONS):
            for el in [x.rsplit("_", maxsplit=1) for x in _]:
                self.BUILDING_OPTIONS.loc[i, el[0]] += int(el[1])
        self.BUILDING_OPTIONS.index = total_building_options

    def get_updated_blocks(self, building_options_ids, blocks_ids=None):
        """Get updated blocks with calculated provision"""
        updated_blocks = self.BUILDING_OPTIONS.loc[building_options_ids]
        if blocks_ids:
            updated_blocks.index = blocks_ids
        else:
            updated_blocks.index = self.BLOCKS["block_id"]

        return updated_blocks.to_dict("index")

    def fitness_func(self, ga_instance, solution, solution_idx):
        """Fitness function for genetic algorithm"""
        updated_blocks = self.get_updated_blocks(solution)
        provisions, fitness = self.LPP.get_scenario_provisions(self.SCENARIO, updated_blocks)
        return fitness

    def make_ga_params(
        self,
        num_generations,
        num_parents_mating,
        sol_per_pop,
        parent_selection_type,
        keep_parents,
        crossover_type,
        mutation_type,
        mutation_percent_genes,
        K_tournament,
        stop_criteria,
        parallel_processing,
    ):
        """Setting parameters of the genetic algorithm"""
        self.ga_params = {
            "fitness_func": self.fitness_func,
            "num_generations": num_generations,
            "num_parents_mating": num_parents_mating,
            "sol_per_pop": sol_per_pop,
            "num_genes": self.BLOCKS.shape[0],
            "gene_space": self.BLOCKS["variants"].tolist(),
            "gene_type": int,
            "parent_selection_type": parent_selection_type,
            "keep_parents": keep_parents,
            "crossover_type": crossover_type,
            "mutation_type": mutation_type,
            "mutation_percent_genes": mutation_percent_genes,
            "K_tournament": K_tournament,
            "stop_criteria": stop_criteria,
            "parallel_processing": parallel_processing,
        }

    def calculate_blocks_building_optinons(self, ga_params):
        """Calculation of the optimal development option by services for blocks"""
        self.BLOCKS = self.BLOCKS[self.BLOCKS["landuse"] != "no_dev_area"]
        self.flatten_dict()
        self.get_combinations()
        self.get_combinations_area()
        self.updating_blocks_combinations()
        self.BLOCKS = self.BLOCKS[self.BLOCKS["variants"].apply(lambda x: len(x)) != 0]
        self.get_building_options()
        self.make_ga_params(**ga_params)

        ga_instance = pygad.GA(**self.ga_params)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        updated_blocks = self.get_updated_blocks(solution)
        return ga_instance, updated_blocks
