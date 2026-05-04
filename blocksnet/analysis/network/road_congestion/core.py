import pandas as pd
import networkx as nx
import numpy as np
from blocksnet.relations.accessibility import validate_accessibility_graph
from ..origin_destination import validate_od_matrix
from tqdm import tqdm
from blocksnet.config import log_config
from loguru import logger

CONGESTION_KEY = "congestion"

LANE_CAPACITY = 1000

LANE_COEF = {
    1: 1.0,
    2: 0.95,
    3: 0.90,
    4: 0.86,
    5: 0.84,
    6: 0.82,
    7: 0.80,
    8: 0.78,
}


def _get_capacity_by_lanes(lanes):
    capacity = LANE_CAPACITY * LANE_COEF[lanes] * lanes

    return capacity

def _normalize_lanes(G: nx.Graph, default: int = 1) -> nx.Graph:
    for _, _, data in G.edges(data=True):
        raw = data.get("lanes", None)

        # list -> возьмём минимальное 
        if isinstance(raw, list):
            raw = min(raw) if raw else None

        # str -> попробуем вытащить число
        if isinstance(raw, str):
            s = raw.strip()
            for sep in [";", "|", ","]:
                if sep in s:
                    s = s.split(sep)[0].strip()
                    break
            raw = s

        try:
            lanes = int(float(raw))  
        except (TypeError, ValueError):
            lanes = default

        if lanes < 1:
            lanes = default

        data["lanes"] = lanes

    return G


def _add_intensity(G: nx.Graph, intensity_default: float = 0) -> nx.Graph:

    for _, _, data in G.edges(data=True):
        data["intensity"] = intensity_default

    return G


def _add_capacity(G: nx.Graph) -> nx.Graph:
    for _, _, data in G.edges(data=True):
        lanes = data.get("lanes", 1)
        data["capacity"] = _get_capacity_by_lanes(lanes)
    return G


def _peprocess_graph(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    _normalize_lanes(H)
    _add_intensity(H)
    _add_capacity(H)
    return H


def road_congestion(od_mx: pd.DataFrame, G: nx.MultiDiGraph, weight_key: str = "time_min") -> nx.MultiDiGraph:

    """
    Assign OD demand to a road network and compute per-edge congestion indicators.

    This function performs a discrete, trip-by-trip shortest-path assignment on a directed
    multigraph. For each OD pair `(i, j)` the demand value is rounded to an integer number of
    trips. Each trip is routed along the current shortest path (Dijkstra) using `weight_key`
    as edge cost. Traversed edges accumulate `intensity` (assigned trip count).

    Edge capacities are derived from the `lanes` attribute and a simple lanes-based coefficient
    model:

    - If `lanes` is missing or invalid, it is treated as 1.
    - Capacity is computed as::

        capacity = LANE_CAPACITY * LANE_COEF[lanes] * lanes

      where `LANE_CAPACITY = 1000` and `LANE_COEF` is defined for lanes 1..6.

    During assignment, the routing graph is updated: if an edge becomes oversaturated
    (`intensity / capacity > 1.0`) that specific multiedge (key) is removed from further routing.
    The returned graph preserves the full original edge set (after preprocessing) and contains
    the final `intensity` and `congestion_level` values.

    Parameters
    ----------
    od_mx : pandas.DataFrame
        Origin-destination matrix with node identifiers as index and columns. Values represent
        demand between zones/nodes and are interpreted as trip counts after rounding to the
        nearest integer. OD pairs with `i == j` or `demand <= 0` are skipped.
    G : networkx.MultiDiGraph
        Directed road network graph. Nodes must include all IDs present in `od_mx`.
        Edges are expected to contain `weight_key` (e.g., travel time in minutes). The `lanes`
        attribute is optional and may be `int`, `float`, `str` (e.g., "2", "2;1"), or `list`
        (the minimal value is taken). Missing/invalid values are coerced to 1.
    weight_key : str, default="time_min"
        Edge attribute used as the weight for shortest-path routing.

    Returns
    -------
    networkx.MultiDiGraph
        A copy of the (preprocessed) input graph with added/updated edge attributes:

        - `lanes` : int
            Normalized number of lanes (at least 1).
        - `capacity` : float
            Per-edge capacity derived from `lanes`.
        - `intensity` : float
            Number of assigned trips traversing the edge.
        - `congestion_level` : float
            Congestion ratio computed as `intensity / capacity` (with `capacity` safeguarded by
            `max(capacity, 1e-9)`).

    Notes
    -----
    - Assignment is discrete (one unit of intensity per trip) and can be slow for large OD totals.
      If you have fractional flows, they are rounded to integers before assignment.
    - For parallel edges between the same `(u, v)`, the edge with the smallest `weight_key`
      value is chosen for each step along the path.
    - Oversaturated edges are removed only from the internal routing graph; the output graph
      keeps all edges and reports their final loads.
    - Lanes outside the supported range (1..6) will raise an error when capacity is computed
      unless you extend `LANE_COEF` accordingly.

    Raises
    ------
    ValueError
        If the OD matrix and graph are incompatible (e.g., missing nodes) or if `weight_key`
        cannot be used for routing due to invalid/missing edge weights.
    networkx.NodeNotFound
        If a source/target node from `od_mx` is not present in the routing graph.
    networkx.NetworkXNoPath
        If no path exists for an OD pair (routing for that pair stops).

    Examples
    --------
    >>> import pandas as pd
    >>> import networkx as nx
    >>> od = pd.DataFrame([[0, 3], [1, 0]], index=[0, 1], columns=[0, 1])
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, time_min=1.0, lanes=1)
    >>> G.add_edge(1, 0, time_min=1.0, lanes=1)
    >>> H = road_congestion(od, G, weight_key="time_min")
    >>> H[0][1][0]["intensity"], round(H[0][1][0]["congestion_level"], 3)
    (3.0, 0.003)
    """

    validate_od_matrix(od_mx, G)
    validate_accessibility_graph(G, weight_key)

    logger.info("Preprocessing graph")
    H = _peprocess_graph(G)
    graph_congestion = H.copy()
    graph_routing = H.copy()

    for _, _, k, data in graph_congestion.edges(keys=True, data=True):
        data["intensity"] = 0.0

    logger.info("Calculating shortest paths")
    for i in tqdm(od_mx.index, disable=log_config.disable_tqdm):
        for j, demand in od_mx.loc[i].items():
            if i == j or demand <= 0:
                continue

            trips = int(round(float(demand)))

            for _ in range(trips):
                try:
                    route = nx.shortest_path(graph_routing, source=i, target=j, weight=weight_key, method="dijkstra")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    break

                for u, v in zip(route[:-1], route[1:]):
                    k = min(graph_routing[u][v], key=lambda kk: graph_routing[u][v][kk].get(weight_key, np.inf))

                    graph_routing[u][v][k]["intensity"] += 1.0
                    graph_congestion[u][v][k]["intensity"] += 1.0

                    capacity = float(graph_routing[u][v][k]["capacity"])
                    congestion_level = graph_routing[u][v][k]["intensity"] / max(capacity, 1e-9)
                    if congestion_level > 1.0:
                        graph_routing.remove_edge(u, v, key=k)

    logger.info("Computing congestion level")
    for _, _, _, data in graph_congestion.edges(keys=True, data=True):
        data["congestion_level"] = data["intensity"] / max(data["capacity"], 1e-9)

    return graph_congestion