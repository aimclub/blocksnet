import pandas as pd
import networkx as nx
from blocksnet.relations.accessibility import validate_accessibility_graph
from ..origin_destination import validate_od_matrix
from tqdm import tqdm
from blocksnet.config import log_config
from loguru import logger

CONGESTION_KEY = "congestion"


def road_congestion(od_mx: pd.DataFrame, graph: nx.Graph, weight_key: str = "time_min"):
    """Estimate edge congestion from an origin-destination matrix.

    Parameters
    ----------
    od_mx : pandas.DataFrame
        Origin-destination flow matrix whose indices align with ``graph``
        nodes.
    graph : networkx.Graph
        Accessibility graph containing edge weights used for shortest paths.
    weight_key : str, optional
        Edge attribute representing travel cost. Defaults to ``"time_min"``.

    Returns
    -------
    networkx.Graph
        Copy of the input graph with a ``congestion`` attribute on each edge.

    Raises
    ------
    ValueError
        If the matrix or graph fail validation.
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
