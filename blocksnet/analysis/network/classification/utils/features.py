import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from .preprocessing import graph_to_gdf


def _avg_neighbor_distance(gdf):
    """Compute average nearest neighbor distance"""
    coords = np.array([[pt.x, pt.y] for pt in gdf.geometry if not pt.is_empty])
    if len(coords) < 2:
        return np.nan
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    distances = nbrs.kneighbors(coords)[0][:, 1]
    return np.mean(distances)


def _link_length_entropy(lengths):
    """Calculate entropy of link length distribution"""
    if len(lengths) < 2:
        return 0.0
    counts, _ = np.histogram(lengths, bins="auto")
    probs = counts / len(lengths)
    return entropy(probs[probs > 0])


def calculate_graph_features(graph: nx.Graph) -> dict:
    gdf = graph_to_gdf(graph)
    # Edge length calculations
    edge_lengths = []
    for u, v in graph.edges():
        if u in gdf.index and v in gdf.index:
            dist = gdf.loc[u].geometry.distance(gdf.loc[v].geometry)
            edge_lengths.append(dist)

    # Degree calculations
    degrees = dict(graph.degree)
    leaf_count = sum(1 for d in degrees.values() if d == 1)

    # Spatial features
    hull = gdf.union_all().convex_hull
    hull_area = hull.area if hull else 0

    return {
        "avg_degree": np.mean(list(degrees.values())),
        "leaf_nodes_proportion": leaf_count / len(degrees),
        "link_density": len(edge_lengths) / hull_area if hull_area > 0 else 0,
        "avg_clustering": nx.average_clustering(graph),
        "density": nx.density(graph),
        "diameter": nx.diameter(graph) if nx.is_connected(graph) else np.nan,
        "assortativity": nx.degree_assortativity_coefficient(graph),
        "avg_near_neighbors_dist": _avg_neighbor_distance(gdf),
        "avg_edge_length": np.mean(edge_lengths) if edge_lengths else 0,
        "ratio_nodes_to_link_length": len(degrees) / sum(edge_lengths) if edge_lengths else 0,
        "link_length_entropy": _link_length_entropy(edge_lengths),
    }
