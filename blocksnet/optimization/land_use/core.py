import networkx as nx
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from ...enums import LandUse
from ...relations import get_adjacency_context
from ...utils.validation import validate_graph
from . import common, utils
from .schemas import BlocksSchema


SOLUTION_COLUMN = "solution"
OBJECTIVES_COLUMN = "objectives"
ASSIGNED_LAND_USE_COLUMN = "assigned_land_use"

OBJECTIVES = {"share_mse": common.share_fitness, "adjacency_penalty": common.adjacency_penalty}


class LandUseOptimizer:
    def __init__(self, blocks_df: pd.DataFrame, adjacency_graph: nx.Graph):
        validate_graph(adjacency_graph, blocks_df)
        self.blocks_df = BlocksSchema(blocks_df)
        self.adjacency_graph = adjacency_graph

    def _get_context(self, blocks_ids: list[int]):
        blocks_df = self.blocks_df.loc[blocks_ids].copy()
        adjacency_graph = get_adjacency_context(self.adjacency_graph, blocks_df)
        context_df = self.blocks_df.loc[list(adjacency_graph.nodes)].copy()
        return blocks_df, context_df, adjacency_graph

    def _result_to_df(self, result, blocks_ids: list[int]) -> pd.DataFrame:
        data = {SOLUTION_COLUMN: list(result.X), OBJECTIVES_COLUMN: list(result.F)}
        df = pd.DataFrame.from_dict(data)

        def explain_solution(solution):
            res = {}
            for i, v in enumerate(solution):
                lu = utils.reverse_transform_lu(v)
                block_id = blocks_ids[i]
                res[block_id] = lu
            return res

        df[ASSIGNED_LAND_USE_COLUMN] = df[SOLUTION_COLUMN].apply(explain_solution)

        for i, objective in enumerate(OBJECTIVES):
            df[objective] = df[OBJECTIVES_COLUMN].apply(lambda f: f[i])

        return df.sort_values("share_mse").reset_index(drop=True)

    def run(
        self,
        blocks_ids: list[int],
        target_shares: dict[LandUse, float],
        population_size: int = 10,
        n_generations: int = 100,
        mutation_probability: float = 0.1,
        seed: int = 42,
    ):
        blocks_df, context_df, adjacency_graph = self._get_context(blocks_ids)

        n_var = len(blocks_ids)
        n_classes = len(LandUse)
        objectives = [
            lambda s: OBJECTIVES["share_mse"](s, blocks_df, target_shares),
            lambda s: OBJECTIVES["adjacency_penalty"](s, blocks_ids, context_df, adjacency_graph),
        ]
        problem = common.Problem(n_var, n_classes, objectives)

        gene_space = utils.generate_gene_space(blocks_df.land_use)
        algorithm = NSGA2(
            pop_size=population_size,
            mutation=common.Mutation(mutation_probability, gene_space),
            sampling=common.Sampling(gene_space),
            crossover=UniformCrossover(prob=1.0),
            repair=RoundingRepair(),
        )

        termination = get_termination("n_gen", n_generations)
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=True,
        )

        return self._result_to_df(result, blocks_ids)
