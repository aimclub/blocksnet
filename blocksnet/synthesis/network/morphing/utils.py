import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
import neatnet
from loguru import logger
from shapely import set_precision
from scipy.spatial.distance import cdist



def graph_to_gdf(graph):
    nodes_data = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')

    x_col = 'x' if 'x' in nodes_data.columns else 'lon'
    y_col = 'y' if 'y' in nodes_data.columns else 'lat'

    if x_col not in nodes_data.columns or y_col not in nodes_data.columns:
        raise ValueError("Не найдено узлов с координатами")

    gdf = gpd.GeoDataFrame(
        nodes_data,
        geometry=gpd.points_from_xy(nodes_data[x_col], nodes_data[y_col]),
        # crs='EPSG:4326'
        crs = graph.graph["crs"]
    )

    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)

    return gdf_utm


def simplify_graph(G: nx.Graph, fix_artifacts: bool = True) -> nx.Graph:
    """
    Упрощает граф с помощью neatnet, сохраняя топологию и геометрию.
    """
    crs = G.graph.get('crs', 'EPSG:4326')
    edges = []

    for u, v, data in G.edges(data=True):
        if 'geometry' not in data:
            node_u = G.nodes[u]
            node_v = G.nodes[v]
            coords_u = node_u.get('pos') or node_u['geometry'].coords[0]
            coords_v = node_v.get('pos') or node_v['geometry'].coords[0]
            geom = LineString([coords_u, coords_v])
        else:
            geom = data['geometry']
        edges.append({'u': u, 'v': v, 'geometry': geom})

    gdf_edges = gpd.GeoDataFrame(edges, geometry='geometry', crs=crs)
    original_crs = gdf_edges.crs

    if gdf_edges.crs.is_geographic:
        gdf_edges = gdf_edges.to_crs("EPSG:3857")

    try:
        simplified_gdf = neatnet.neatify(gdf_edges, artifact_threshold_fallback=7, artifact_threshold=7, consolidation_tolerance=0)
    except Exception as e:
        if not fix_artifacts and "face_artifact_index" in str(e):
            simplified_gdf = gdf_edges
            logger.warning("Simplify failed, using original geometry")
        else:
            raise

    if simplified_gdf.crs != original_crs:
        simplified_gdf = simplified_gdf.to_crs(original_crs)

    simplified_gdf['geometry'] = set_precision(simplified_gdf['geometry'], grid_size=1e-7)

    simplified_G = nx.Graph()
    point_to_node = {}
    current_id = 0

    for _, row in simplified_gdf.iterrows():
        geom = row['geometry']
        if not isinstance(geom, LineString):
            continue

        start = tuple(geom.coords[0])
        end = tuple(geom.coords[-1])

        for point in [start, end]:
            if point not in point_to_node:
                point_to_node[point] = current_id
                simplified_G.add_node(current_id, pos=point)
                current_id += 1

        u = point_to_node[start]
        v = point_to_node[end]
        simplified_G.add_edge(u, v, geometry=geom)

    simplified_G.graph['crs'] = original_crs

    for node, data in simplified_G.nodes(data=True):
        if 'pos' in data:
            x, y = data['pos']
            data['x'] = x
            data['y'] = y
            del data['pos']
    
    g = nx.MultiDiGraph()
    for node, data in simplified_G.nodes(data=True):
        g.add_node(node, **data)

    for u, v, data in simplified_G.edges(data=True):
        g.add_edge(u, v, **data)

    g.graph.update(simplified_G.graph)
    return g


def adj_matrix(G):
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    adj_matrix = np.zeros((n, n), dtype=float)
    
    for u, v, data in G.edges(data=True):
        i = node_to_idx[u]
        j = node_to_idx[v]
        adj_matrix[i, j] = 1
    
    return adj_matrix



def compute_distance_matrix(graph, gdf):

    coords = np.array([(point.x, point.y) for point in gdf.geometry])
    distance_matrix = cdist(coords, coords)

    return distance_matrix