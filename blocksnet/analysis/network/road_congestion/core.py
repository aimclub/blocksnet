"""Estimate congestion on a road graph using an OD matrix."""

import pandas as pd
import networkx as nx
from tqdm import tqdm
from loguru import logger

from blocksnet.relations.accessibility import validate_accessibility_graph
from ..origin_destination import validate_od_matrix
from blocksnet.config import log_config

CONGESTION_KEY = "congestion"


def road_congestion(od_mx: pd.DataFrame, graph: nx.Graph, weight_key: str = "time_min"):
    """Assign congestion levels to edges based on shortest-path flows.

    Parameters
    ----------
    od_mx : pandas.DataFrame
        Origin-destination matrix with identical index and columns representing node identifiers.
    graph : networkx.Graph
        Road graph with edge weights accessible via ``weight_key``.
    weight_key : str, default ``"time_min"``
        Edge attribute used to compute shortest paths.

    Returns
    -------
    networkx.Graph
        Copy of ``graph`` with a ``congestion`` attribute accumulated on each edge.

    Raises
    ------
    ValueError
        If the OD matrix or graph do not satisfy validation requirements.
    """

    validate_od_matrix(od_mx, graph)
    validate_accessibility_graph(graph, weight_key)
    graph = graph.copy()

    logger.info("Calculating shortest path")
    paths = dict(nx.all_pairs_dijkstra_path(graph, weight=weight_key))

    logger.info("Evaluating congestion")
    for edge in graph.edges(data=True):
        data = edge[2]
        data[CONGESTION_KEY] = 0.0

    for i in tqdm(od_mx.index, disable=log_config.disable_tqdm):
        for j in od_mx.columns:
            if i == j:
                continue
            if i in paths and j in paths[i]:
                path = paths[i][j]
                for k in range(len(path) - 1):
                    # FIXME multidigraph edges split congestion
                    graph[path[k]][path[k + 1]][0][CONGESTION_KEY] += od_mx[i][j]
    return graph
