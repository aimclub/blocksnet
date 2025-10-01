import geopandas as gpd
import numpy as np
from shapely import convex_hull


def clusterize(buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    import warnings

    warnings.filterwarnings(
        "ignore", message=".*force_all_finite.*", category=FutureWarning, module="sklearn.utils.deprecation"
    )

    import hdbscan

    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["center"] = buildings_gdf.geometry.centroid
    coords = np.array([[point.x, point.y] for point in buildings_gdf.center])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
    labels = clusterer.fit_predict(coords)
    buildings_gdf["cluster"] = labels

    return buildings_gdf


def make_convex_hulls(clustered_gdf: gpd.GeoDataFrame):
    clusters = clustered_gdf["cluster"].unique()
    convex_hulls = []

    for cluster_id in clusters:
        if cluster_id == -1:
            continue

        cluster_points = clustered_gdf[clustered_gdf["cluster"] == cluster_id]

        if len(cluster_points) < 2:
            continue

        combined = cluster_points.union_all()
        hull = convex_hull(combined)

        convex_hulls.append({"cluster": cluster_id, "geometry": hull})

    if not convex_hulls:
        return gpd.GeoDataFrame(columns=["cluster", "geometry"], geometry="geometry", crs=clustered_gdf.crs)

    hull_gdf = gpd.GeoDataFrame(convex_hulls, geometry="geometry", crs=clustered_gdf.crs)

    return hull_gdf
