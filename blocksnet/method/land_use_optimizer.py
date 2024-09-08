import pygad
import geopandas as gpd
from loguru import logger
from tqdm import tqdm
from pydantic import Field
from .base_method import BaseMethod
from ..models import Block, LandUse
from .annealing_optimizer import AnnealingOptimizer


class Indicator:
    def __init__(self, fsi_min, fsi_max, gsi_min, gsi_max):
        self.fsi_min = fsi_min  # минимальный коэффициент плотности застройки
        self.fsi_max = fsi_max  # максимальный коэффициент плотности застройки
        self.gsi_min = gsi_min  # минимальный процент застроенности участка
        self.gsi_max = gsi_max  # максимальный процент застроенности участка


LU_INDICATORS = {
    LandUse.RESIDENTIAL: Indicator(fsi_min=0.5, fsi_max=3.0, gsi_min=0.2, gsi_max=0.8),
    LandUse.BUSINESS: Indicator(fsi_min=1.0, fsi_max=3.0, gsi_min=0.0, gsi_max=0.8),
    LandUse.RECREATION: Indicator(fsi_min=0.05, fsi_max=0.2, gsi_min=0.0, gsi_max=0.3),
    LandUse.SPECIAL: Indicator(fsi_min=0.05, fsi_max=0.2, gsi_min=0.05, gsi_max=0.15),
    LandUse.INDUSTRIAL: Indicator(fsi_min=0.3, fsi_max=1.5, gsi_min=0.2, gsi_max=0.8),
    LandUse.AGRICULTURE: Indicator(fsi_min=0.1, fsi_max=0.2, gsi_min=0.0, gsi_max=0.6),
    LandUse.TRANSPORT: Indicator(fsi_min=0.2, fsi_max=1.0, gsi_min=0.0, gsi_max=0.8),
}

import time
from functools import wraps


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Finished function: {func.__name__} (Elapsed time: {elapsed_time:.3f} seconds)")
        return result

    return wrapper


class LandUseOptimizer(BaseMethod):

    annealing_rate: float = Field(default=0.5, ge=0, lt=1)
    max_iter: int = Field(default=50, ge=1)

    @property
    def annealing_optimizer(self):
        return AnnealingOptimizer(city_model=self.city_model, verbose=False)

    @log_execution_time
    def _fitness(self, blocks_lu: dict[int, LandUse], service_types: dict[str, float], objective: LandUse):

        ao = self.annealing_optimizer

        blocks_fsi = {b_id: LU_INDICATORS[lu].fsi_min for b_id, lu in blocks_lu.items()}
        blocks_gsi = {b_id: LU_INDICATORS[lu].gsi_min for b_id, lu in blocks_lu.items()}
        X, indicators, value, _ = ao.calculate(
            blocks_lu, blocks_fsi, blocks_gsi, service_types, rate=self.annealing_rate, max_iter=self.max_iter
        )

        gdf = ao.to_gdf(X, indicators)
        area = gdf.area.sum()
        lu_area = gdf[gdf.land_use == objective.value].area.sum()

        return value, lu_area / area

    def to_gdf(self, blocks: dict[int, LandUse]):
        return gpd.GeoDataFrame(
            [
                {"block_id": self.city_model[b_id].id, "geometry": self.city_model[b_id].geometry, "land_use": lu}
                for b_id, lu in blocks.items()
            ],
            crs=self.city_model.crs,
        ).set_index("block_id", drop=True)

    def calculate(
        self,
        blocks: list[Block],
        service_types: dict[str, float],
        objective: LandUse,
        generations: int = 10,
        parents: int = 4,
        solutions: int = 8,
        # processes : int = 8
    ) -> tuple[dict[int, LandUse], tuple[float, float], pygad.GA]:

        lu_mapping = {i: lu for i, lu in enumerate(LandUse)}
        gene_space = [[v for v in lu_mapping.keys()] for _ in blocks]

        def fitness_func(ga_instance, solution, solution_idx):

            blocks_lu = {blocks[i].id: lu_mapping[lu_i] for i, lu_i in enumerate(solution)}
            fitness = self._fitness(blocks_lu, service_types, objective)
            return fitness

        def on_generation(ga_instance):
            if self.verbose:
                logger.info(f"Best fitness : {[round(v,2) for v in ga_instance.best_solution()[1]]}")

        ga_instance = pygad.GA(
            fitness_func=fitness_func,
            on_generation=on_generation,
            num_generations=generations,
            num_parents_mating=parents,
            parent_selection_type="tournament",
            crossover_type="single_point",
            sol_per_pop=solutions,
            # parallel_processing=processes,
            num_genes=len(gene_space),
            gene_space=gene_space,
            keep_elitism=2,
        )
        ga_instance.run()

        solution, fitness, _ = ga_instance.best_solution()

        return {b_i: lu_mapping[lu_i] for b_i, lu_i in enumerate(solution)}, fitness, ga_instance
