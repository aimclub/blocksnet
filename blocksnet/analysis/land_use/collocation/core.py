import networkx as nx
import pandas as pd
from tqdm import tqdm
from .schemas import BlocksSchema
from blocksnet.config import land_use_config, log_config
from blocksnet.relations import validate_adjacency_graph

COLLOCATION_COLUMN = "collocation"


def land_use_collocation(adjacency_graph: nx.Graph, blocks_df: pd.DataFrame):

    blocks_df = BlocksSchema(blocks_df)
    validate_adjacency_graph(adjacency_graph, blocks_df)

    def collocation(series: pd.Series):

        block_a = series.name
        area_a = series.site_area
        lu_a = series.land_use

        adjacency_sum = 0
        adjacency_max = 0

        for block_b in adjacency_graph.neighbors(block_a):

            area_b = blocks_df.loc[block_b].site_area
            lu_b = blocks_df.loc[block_b].land_use

            adjacency = area_a * area_b
            adjacency_max += adjacency

            adjacency_allowed = int(land_use_config.adjacency_rules.has_edge(lu_a, lu_b))
            adjacency_sum += adjacency * adjacency_allowed

        if adjacency_max == 0:
            return None

        return adjacency_sum / adjacency_max

    if log_config.disable_tqdm:
        blocks_df[COLLOCATION_COLUMN] = blocks_df.apply(collocation, axis=1)
    else:
        blocks_df[COLLOCATION_COLUMN] = blocks_df.progress_apply(collocation, axis=1)

    return blocks_df
