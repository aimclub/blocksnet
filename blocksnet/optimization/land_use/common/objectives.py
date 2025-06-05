import numpy as np
import pandas as pd
import networkx as nx
from ..utils import reverse_transform_lu
from ....enums import LandUse
from ....config import land_use_config


def share_fitness(solution, blocks_df: pd.DataFrame, target_shares: dict[LandUse, float]):
    blocks_df = blocks_df.copy()
    blocks_df["land_use"] = [reverse_transform_lu(x) for x in solution]

    total_area = blocks_df.area.sum()
    deviations = []

    for lu, target_share in target_shares.items():
        area = blocks_df[blocks_df["land_use"] == lu].area.sum()
        share = area / total_area
        deviation = (share - target_share) ** 2
        deviations.append(deviation)

    return sum(deviations)


def adjacency_penalty(solution, blocks_ids: list[int], context_df: pd.DataFrame, adjacency_graph: nx.Graph):
    context_df = context_df.copy()
    context_df.loc[blocks_ids, "land_use"] = [reverse_transform_lu(x) for x in solution]

    adjacency_rules = land_use_config.adjacency_rules
    edges = [(u, v) for u, v in adjacency_graph.edges if u in blocks_ids or v in blocks_ids]

    def penalty(u, v) -> float:
        u_lu = context_df.loc[u, "land_use"]
        v_lu = context_df.loc[v, "land_use"]

        if u_lu is None or v_lu is None:
            return 0

        if adjacency_rules.has_edge(u_lu, v_lu):
            return 0

        return context_df.loc[u, "area"] * context_df.loc[v, "area"]

    max_penalty = np.sum([context_df.loc[u, "area"] * context_df.loc[v, "area"] for u, v in edges])
    cur_penalty = np.sum([penalty(u, v) for u, v in edges])

    return cur_penalty / max_penalty
