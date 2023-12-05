from typing import Literal

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from pydantic import BaseModel, Field, InstanceOf, field_validator
from shapely import LineString, MultiPolygon, Point, Polygon, line_locate_point
from shapely.ops import linemerge, nearest_points, split

from ..models import BaseRow, GeoDataFrame


OX_CRS = 4326
METERS_IN_KILOMETER = 1000
MINUTES_IN_HOUR = 60


class GraphNode(BaseModel):
    x: float
    y: float


class GraphEdge(BaseModel):
    geometry: InstanceOf[LineString] = None
    weight: float = Field(ge=0)
    transport_type: Literal["walk", "drive", "subway", "tram", "bus", "trolleybus"]


class CityRow(BaseRow):
    geometry: Polygon | MultiPolygon


class GraphGenerator(BaseModel):
    city_geometry: InstanceOf[GeoDataFrame[CityRow]]
    """City geometry or geometries"""
    overpass_url: str = "http://lz4.overpass-api.de/api/interpreter"
    """Overpass url used in OSM queries"""
    speed: dict[str, int] = {"walk": 4, "drive": 25, "subway": 12, "tram": 15, "trolleybus": 12, "bus": 17}
    """Average transport type speed in km/h"""
    waiting_time: dict[str, int] = {
        "subway": 5,
        "tram": 5,
        "trolleybus": 5,
        "bus": 5,
    }
    """Average waiting time in min"""

    @field_validator("city_geometry", mode="before")
    def validate_fields(gdf):
        if not isinstance(gdf, GeoDataFrame[CityRow]):
            gdf = GeoDataFrame[CityRow](gdf)
        return gdf

    @staticmethod
    def to_graphml(graph: nx.MultiDiGraph, file_path: str):
        """Save graph as OX .graphml"""
        ox.save_graphml(graph, file_path)

    @staticmethod
    def from_graphml(file_path: str):
        """Load graph from OX .graphml"""
        return ox.load_graphml(file_path)

    @property
    def local_crs(self):
        return self.city_geometry.crs

    @classmethod
    def plot(cls, graph: nx.MultiDiGraph):
        _, edges = ox.graph_to_gdfs(graph)
        edges.plot(column="transport_type", legend=True).set_axis_off()

    def _get_speed(self, transport_type: str):
        """Return transport type speed in meters per minute"""
        return METERS_IN_KILOMETER * self.speed[transport_type] / MINUTES_IN_HOUR

    def _get_basic_graph(self, network_type: Literal["walk", "drive"]):
        """Returns walk or drive graph for the city geometry"""
        speed = self._get_speed(network_type)
        G = ox.graph_from_polygon(polygon=self.city_geometry.to_crs(OX_CRS).unary_union, network_type=network_type)
        G = ox.project_graph(G, to_crs=self.local_crs)
        for edge in G.edges(data=True):
            _, _, data = edge
            length = data["length"]
            data["weight"] = length / speed
            data["transport_type"] = network_type
        G = ox.project_graph(G, self.local_crs)
        print(f"Graph made for '{network_type}' network type")
        return G

    def _get_routes(
        self, bounds: pd.DataFrame, public_transport_type: Literal["subway", "tram", "trolleybus", "bus"]
    ) -> pd.DataFrame:
        """Returns OSM routes for the given geometry shapely geometry bounds and given transport type"""
        bbox = f"{bounds.loc[0,'miny']},{bounds.loc[0,'minx']},{bounds.loc[0,'maxy']},{bounds.loc[0,'maxx']}"
        tags = f"'route'='{public_transport_type}'"
        overpass_query = f"""
    [out:json];
            (
                relation({bbox})[{tags}];
            );
    out geom;
    """
        result = requests.get(self.overpass_url, params={"data": overpass_query})
        json_result = result.json()["elements"]
        print(f"Fetched routes for '{public_transport_type}'")
        return pd.DataFrame(json_result)

    @staticmethod
    def _coordinates_to_linestring(coordinates: list[dict[str, float]]) -> LineString:
        """For given route coordinates dicts returns a concated linestring"""
        points = []
        for point in coordinates:
            points.append(Point(point["lon"], point["lat"]))
        linestring = LineString(points)
        return linestring

    def _ways_to_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Returns GeoDataFrame for the given route ways, converting way's coordinates to linestring"""
        copy = df.copy()
        copy["geometry"] = df["coordinates"].apply(lambda x: self._coordinates_to_linestring(x))
        return gpd.GeoDataFrame(copy, geometry=copy["geometry"]).set_crs(epsg=OX_CRS).to_crs(self.local_crs)

    def _nodes_to_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Returns GeoDataFrame for the given route nodes, converting lon and lat columns to geometry column and local CRS"""
        return (
            gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]))
            .set_crs(epsg=OX_CRS)
            .to_crs(self.local_crs)
        )

    def _graph_from_route(self, nodes: pd.DataFrame, ways: pd.DataFrame, transport_type: str) -> list[nx.MultiGraph]:
        """Get graph for the given route nodes and ways"""
        linestring = linemerge(list(ways["geometry"]))
        nodes = nodes.copy()
        nodes["geometry"] = nodes["geometry"].apply(lambda x: nearest_points(linestring, x)[0])
        nodes["distance"] = nodes["geometry"].apply(lambda x: line_locate_point(linestring, x))
        nodes = nodes.loc[nodes["geometry"].within(self.city_geometry.unary_union)]
        sorted_nodes = nodes.sort_values(by="distance").reset_index()
        sorted_nodes["hash"] = sorted_nodes["geometry"].apply(lambda x: f"{transport_type}_{hash(x)}")
        G = nx.MultiDiGraph()
        for index in list(sorted_nodes.index)[:-1]:
            n1 = sorted_nodes.loc[index]
            n2 = sorted_nodes.loc[index + 1]
            d = n2["distance"] - n1["distance"]
            id1 = n1["hash"]  # hash(n1['geometry'])
            id2 = n2["hash"]  # hash(n1['geometry'])
            speed = self._get_speed(transport_type)
            G.add_edge(id1, id2, weight=d / speed, transport_type=transport_type)
            G.nodes[id1]["x"] = n1["geometry"].x
            G.nodes[id1]["y"] = n1["geometry"].y
            G.nodes[id2]["x"] = n2["geometry"].x
            G.nodes[id2]["y"] = n2["geometry"].y
        return G

    def _get_pt_graph(self, pt_type: Literal["subway", "tram", "trolleybus", "bus"]) -> list[nx.MultiGraph]:
        """Get public transport routes graphs for the given transport_type"""
        routes: pd.DataFrame = self._get_routes(self.city_geometry.to_crs(OX_CRS).bounds, pt_type)
        graphs = []
        for i in routes.index:
            df = pd.DataFrame(routes.loc[i, "members"])
            nodes_df = df.loc[lambda x: x["type"] == "node"].copy()
            ways_df = df.loc[lambda x: x["type"] == "way"].copy().rename(columns={"geometry": "coordinates"})
            if len(nodes_df) == 0 or len(ways_df) == 0:
                continue
            nodes_gdf = self._nodes_to_gdf(nodes_df)
            ways_gdf = self._ways_to_gdf(ways_df)
            graphs.append(self._graph_from_route(nodes_gdf, ways_gdf, pt_type))
        graph = None
        if len(graphs) > 0:
            graph = nx.compose_all(graphs)
            graph.graph["crs"] = self.local_crs
        print(f"Graph made for '{pt_type}'")
        return graph

    @staticmethod
    def validate_graph(graph) -> nx.MultiDiGraph:
        """Returns validated copy of the graph, according to ```GraphEdge``` and ```GraphNode``` classes"""
        graph = graph.copy()
        for d in map(lambda e: e[2], graph.edges(data=True)):
            d = GraphEdge(**d).__dict__
        for d in map(lambda n: n[1], graph.nodes(data=True)):
            d = GraphNode(**d).__dict__
        return graph

    def get_graph(self, graph_type: Literal["intermodal", "walk", "drive"]):
        """Returns intermodal graph for the city geometry bounds"""
        if graph_type != "intermodal":
            graph = self._get_basic_graph(graph_type)
            return self.validate_graph(graph)

        walk_graph: nx.MultiDiGraph = self._get_basic_graph("walk")
        walk_nodes, _ = ox.graph_to_gdfs(walk_graph)

        pt_types: list[str] = ["bus", "trolleybus", "tram", "subway"]
        pt_graphs: list[nx.MultiDiGraph] = list(map(lambda t: self._get_pt_graph(t), pt_types))
        pt_graphs = list(filter(lambda g: g is not None, pt_graphs))
        pt_graph = nx.compose_all(pt_graphs)
        pt_graph.crs = self.local_crs
        pt_nodes, _ = ox.graph_to_gdfs(pt_graph)

        intermodal_graph = nx.compose(walk_graph, pt_graph)
        pt_to_walk = pt_nodes.sjoin_nearest(walk_nodes, how="left", distance_col="distance")
        for i in pt_to_walk.index:
            gs = pt_to_walk.loc[i]
            transport_node = i
            walk_node = gs["index_right"]
            distance = gs["distance"]
            speed = self._get_speed("walk")
            weight = distance / speed
            intermodal_graph.add_edge(transport_node, walk_node, weight=weight, transport_type="walk")
            intermodal_graph.add_edge(walk_node, transport_node, weight=weight + 5, transport_type="walk")
        intermodal_graph.graph["crs"] = self.local_crs

        return self.validate_graph(intermodal_graph)
