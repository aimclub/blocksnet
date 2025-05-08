import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ....enums import LandUse
from .schemas import BlocksSchema, NodesSchema, validate_graph, validate_matrix


LU_COEF_COLUMN = "lu_coeff"
ACCESSIBILITY_TIME = 10
LU_WEIGHTS = {
    None: 0.06,
    LandUse.INDUSTRIAL: 0.25,
    LandUse.BUSINESS: 0.3,
    LandUse.SPECIAL: 0.1,
    LandUse.TRANSPORT: 0.1,
    LandUse.RESIDENTIAL: 0.1,
    LandUse.AGRICULTURE: 0.05,
    LandUse.RECREATION: 0.05,
}


def _preprocess_blocks(blocks: gpd.GeoDataFrame, crs: int, lu_weights: dict) -> gpd.GeoDataFrame:

    blocks = BlocksSchema(blocks)
    blocks.to_crs(crs, inplace=True)

    scaler = MinMaxScaler()
    blocks[["density", "diversity"]] = scaler.fit_transform(blocks[["density", "diversity"]])
    # compute attractiveness
    blocks[LU_COEF_COLUMN] = blocks["land_use"].apply(lambda x: lu_weights.get(x, 0))
    blocks["attractiveness"] = blocks["density"] + blocks["diversity"] + blocks[LU_COEF_COLUMN]
    blocks[["population", "attractiveness"]] = scaler.fit_transform(blocks[["population", "attractiveness"]])
    blocks = blocks.set_geometry("geometry")
    return blocks


def _compute_node_weights(
    blocks: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    walk_acc_matrix: pd.DataFrame,
    crs: int,
) -> None:

    nodes = NodesSchema(nodes)
    nodes.to_crs(crs, inplace=True)

    # build block→nearby-stops map
    walk_dict = {}
    for i, row in walk_acc_matrix.iterrows():
        walk_dict[i] = [(j, v) for j, v in row.items() if v <= ACCESSIBILITY_TIME]
        if not walk_dict[i]:
            walk_dict[i] = [(row.idxmin(), row.min())]

    # convert to weighted contributions
    block_to_weights = {}
    for blk, stops in walk_dict.items():
        stop_ids = np.array([s for s, _ in stops])
        dists = np.array([d if d > 0 else 0.1 for _, d in stops], float)
        w = 1 / dists
        wn = w / w.sum()
        block_to_weights[blk] = list(zip(stop_ids, wn))

    # aggregate into node attributes
    stops_dict = {s: {"att": 0, "pop": 0} for s in nodes.index}
    for blk, contributions in block_to_weights.items():
        for stop, weight in contributions:
            stops_dict[stop]["att"] += blocks.iloc[blk]["attractiveness"] * weight
            stops_dict[stop]["pop"] += blocks.iloc[blk]["population"] * weight

    nodes["att"] = [stops_dict[s]["att"] for s in nodes.index]
    nodes["pop"] = [stops_dict[s]["pop"] for s in nodes.index]
    return nodes


def origin_destination_matrix(
    blocks: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    walk_acc_matrix: pd.DataFrame,
    drive_acc_matrix: pd.DataFrame,
    crs: int,
    lu_weights: dict[LandUse, float] = LU_WEIGHTS,
) -> pd.DataFrame:

    # validate inputs
    validate_matrix(walk_acc_matrix, blocks, nodes)
    validate_matrix(drive_acc_matrix, nodes)

    # 1) preprocess blocks, compute attractiveness & population
    blocks = _preprocess_blocks(blocks, crs, lu_weights)

    # 2) compute node 'att' and 'pop' fields
    nodes = _compute_node_weights(blocks, nodes, walk_acc_matrix, crs)

    # 3) build OD via gravity model
    adj_mx = drive_acc_matrix.replace(0, np.nan)
    od = pd.DataFrame(
        np.outer(nodes["pop"], nodes["att"]) / adj_mx,
        index=adj_mx.index,
        columns=adj_mx.columns,
    ).fillna(0)

    return od


def road_congestion(od_mx: pd.DataFrame, graph: nx.MultiDiGraph):

    graph = graph.copy()
    validate_graph(graph, "time_min")
    path = dict(nx.all_pairs_dijkstra_path(graph, weight="time_min"))

    for u, v, d in graph.edges(data=True):
        d["congestion"] = 0.0

    for i in range(len(graph.nodes)):
        for j in range(len(graph.nodes)):
            if i in path and j in path[i]:
                p = path[i][j]
                for k in range(len(p) - 1):
                    graph[p[k]][p[k + 1]][0]["congestion"] += od_mx[i][j]
    return graph
