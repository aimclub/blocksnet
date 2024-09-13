import statistics
import pygad
import geopandas as gpd
import networkx as nx
from loguru import logger
from tqdm import tqdm
from pydantic import Field
from .base_method import BaseMethod
from ..models import Block, LandUse
from .annealing_optimizer import AnnealingOptimizer

ADJACENCY_RULES = [
    # self adjacency
    (LandUse.RESIDENTIAL, LandUse.RESIDENTIAL),
    (LandUse.BUSINESS, LandUse.BUSINESS),
    (LandUse.RECREATION, LandUse.RECREATION),
    (LandUse.INDUSTRIAL, LandUse.INDUSTRIAL),
    (LandUse.TRANSPORT, LandUse.TRANSPORT),
    (LandUse.SPECIAL, LandUse.SPECIAL),
    (LandUse.AGRICULTURE, LandUse.AGRICULTURE),
    # recreation can be adjacent to anything
    (LandUse.RECREATION, LandUse.SPECIAL),
    (LandUse.RECREATION, LandUse.INDUSTRIAL),
    (LandUse.RECREATION, LandUse.BUSINESS),
    (LandUse.RECREATION, LandUse.AGRICULTURE),
    (LandUse.RECREATION, LandUse.TRANSPORT),
    (LandUse.RECREATION, LandUse.RESIDENTIAL),
    # residential
    (LandUse.RESIDENTIAL, LandUse.BUSINESS),
    # business
    (LandUse.BUSINESS, LandUse.INDUSTRIAL),
    (LandUse.BUSINESS, LandUse.TRANSPORT),
    # industrial
    (LandUse.INDUSTRIAL, LandUse.SPECIAL),
    (LandUse.INDUSTRIAL, LandUse.AGRICULTURE),
    (LandUse.INDUSTRIAL, LandUse.TRANSPORT),
    # transport
    (LandUse.TRANSPORT, LandUse.SPECIAL),
    (LandUse.TRANSPORT, LandUse.AGRICULTURE),
    # special
    (LandUse.SPECIAL, LandUse.AGRICULTURE),
]

RULES_GRAPH = nx.from_edgelist(ADJACENCY_RULES)


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

    def _fitness_by_share(self, blocks_lu: dict[int, LandUse], lu_shares: dict[LandUse, float]) -> float:
        gdf = gpd.GeoDataFrame(
            [{"id": i, "geometry": self.city_model[i].geometry, "land_use": lu} for i, lu in blocks_lu.items()],
            crs=self.city_model.crs,
        ).set_index("id", drop=True)
        area = gdf.area.sum()
        shares = []
        for lu, share in lu_shares.items():
            lu_area = gdf[gdf.land_use == lu].area.sum()
            lu_share = lu_area / area
            shares.append(abs(lu_share - share))
        return statistics.mean(shares)

    def _fitness_by_rules(self, blocks_lu, blocks_graph) -> float:
        all_edges = blocks_graph.edges
        proper_edges = [(u, v) for u, v in blocks_graph.edges if RULES_GRAPH.has_edge(blocks_lu[u], blocks_lu[v])]
        return len(proper_edges) / len(all_edges)

    def _fitness_by_provision(self, blocks_lu: dict[int, LandUse], service_types: dict[str, float]) -> float:

        if not LandUse.RESIDENTIAL in blocks_lu.keys():
            return 0

        ao = self.annealing_optimizer

        blocks_fsi = {b_id: LU_INDICATORS[lu].fsi_min for b_id, lu in blocks_lu.items()}
        blocks_gsi = {b_id: LU_INDICATORS[lu].gsi_min for b_id, lu in blocks_lu.items()}
        _, _, value, _ = ao.calculate(
            blocks_lu, blocks_fsi, blocks_gsi, service_types, rate=self.annealing_rate, max_iter=self.max_iter
        )
        return value

    def to_gdf(self, blocks: dict[int, LandUse]):
        return gpd.GeoDataFrame(
            [
                {"block_id": self.city_model[b_id].id, "geometry": self.city_model[b_id].geometry, "land_use": lu}
                for b_id, lu in blocks.items()
            ],
            crs=self.city_model.crs,
        ).set_index("block_id", drop=True)

    @staticmethod
    def _get_blocks_adjacency_graph(blocks: list[Block], buffer=1):
        crs = blocks[0].city.crs
        gdf = gpd.GeoDataFrame(
            [{"id": block.id, "geometry": block.geometry.buffer(buffer)} for block in blocks], crs=crs
        ).set_index("id", drop=True)
        sjoin = gpd.sjoin(gdf[["geometry"]], gdf[["geometry"]], predicate="intersects")
        edge_list = [(i, s.index_right) for i, s in sjoin[sjoin.index != sjoin.index_right][["index_right"]].iterrows()]
        return nx.from_edgelist(edge_list)

    def calculate(
        self,
        blocks: list[Block],
        service_types: dict[str, float],
        lu_shares: dict[LandUse, float],
        generations: int = 10,
        parents: int = 4,
        solutions: int = 8,
        # processes : int = 8
    ) -> tuple[dict[int, LandUse], tuple[float, float], pygad.GA]:

        blocks_graph = self._get_blocks_adjacency_graph(blocks)

        lu_mapping = {i: lu for i, lu in enumerate(LandUse)}
        gene_space = [[v for v in lu_mapping.keys()] for _ in blocks]

        def fitness_func(ga_instance, solution, solution_idx):

            blocks_lu = {blocks[i].id: lu_mapping[lu_i] for i, lu_i in enumerate(solution)}
            rules_ratio = self._fitness_by_rules(blocks_lu, blocks_graph)
            share_value = self._fitness_by_share(blocks_lu, lu_shares)
            provision_value = self._fitness_by_provision(blocks_lu, service_types)
            return rules_ratio + rules_ratio * share_value + rules_ratio * share_value * provision_value

        if self.verbose:
            pbar = tqdm(range(0, generations))

        def on_generation(ga_instance):
            if self.verbose:
                pbar.update(1)
                pbar.set_description_str(f"{round(ga_instance.best_solution()[1],2)}")

        ga_instance = pygad.GA(
            fitness_func=fitness_func,
            on_generation=on_generation,
            num_generations=generations,
            num_parents_mating=parents,
            parent_selection_type="rank",
            crossover_type="scattered",
            sol_per_pop=solutions,
            # parallel_processing=processes,
            num_genes=len(gene_space),
            gene_space=gene_space,
            keep_elitism=parents,
            # mutation_probability=0.01,
            # stop_criteria='reach_1'
        )
        ga_instance.run()

        solution, fitness, _ = ga_instance.best_solution()

        return {b_i: lu_mapping[lu_i] for b_i, lu_i in enumerate(solution)}, fitness, ga_instance
