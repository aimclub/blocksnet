import networkx as nx
from typing import Literal
import shapely
import geopandas as gpd
from .schemas import TerritorySchema

IDUEDU_CRS = 4326


def _get_geometry(territory_gdf: gpd.GeoDataFrame) -> shapely.Polygon:
    territory_gdf = TerritorySchema(territory_gdf)
    polygon_geom = territory_gdf.union_all().convex_hull
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_geom], crs=territory_gdf.crs)
    if polygon_gdf.crs.to_epsg() != IDUEDU_CRS:
        polygon_gdf = polygon_gdf.to_crs(IDUEDU_CRS)

    return polygon_gdf.iloc[0].geometry


def get_accessibility_graph(
    territory_gdf: gpd.GeoDataFrame, graph_type: Literal["drive", "walk", "intermodal"], *args, **kwargs
) -> nx.Graph:

    geometry = _get_geometry(territory_gdf)

    import iduedu as ie

    if graph_type == "drive":
        return ie.get_drive_graph(polygon=geometry, *args, **kwargs)
    if graph_type == "walk":
        return ie.get_walk_graph(polygon=geometry, *args, **kwargs)
    if graph_type == "intermodal":
        return ie.get_intermodal_graph(polygon=geometry, *args, **kwargs)
    raise ValueError("Graph type must be drive, walk or intermodal")
