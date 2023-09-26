import requests
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
from pydantic import BaseModel, Field, InstanceOf
from typing import Literal
from shapely import Polygon, MultiPolygon, LineString, Point, line_locate_point
from shapely.ops import nearest_points, linemerge, split

OX_CRS = 4326
METERS_IN_KILOMETER = 1000
MINUTES_IN_HOUR = 60


class GraphGenerator(BaseModel):

    city_geometry: InstanceOf[gpd.GeoDataFrame]
    """City geometry or geometries"""
    local_crs: int
    """EPSG crs of the given city"""
    overpass_url: str = "http://lz4.overpass-api.de/api/interpreter"
    """Overpass url used in OSM queries"""
    speed: dict[str, int] = {"walk": 4, "subway": 12, "tram": 15, "trolleybus": 12, "bus": 17}
    """Average transport type speed in km/h"""
    waiting_time: dict[str, int] = {
        "subway": 5,
        "tram": 5,
        "trolleybus": 5,
        "bus": 5,
    }
    """Average waiting time in min"""

    def _get_speed(self, transport_type: str):
        """Return transport type speed in meters per minute"""
        return METERS_IN_KILOMETER * self.speed[transport_type] / MINUTES_IN_HOUR

    def _get_walk_graph(self):
        """Returns pedestrian graph for the given geometry"""
        speed = self._get_speed("walk")
        G = ox.graph_from_polygon(polygon=self.city_geometry.unary_union, network_type="walk")
        nodes, edges = ox.graph_to_gdfs(G)
        nodes.to_crs(self.local_crs, inplace=True)
        nodes["x"] = nodes["geometry"].apply(lambda g: g.x)
        nodes["y"] = nodes["geometry"].apply(lambda g: g.y)
        nodes.drop(labels=["highway", "street_count"], axis=1, inplace=True)
        G = ox.graph_from_gdfs(nodes, edges)
        for edge in G.edges(data=True):
            _, _, data = edge
            length = data["length"]
            data.clear()
            data["weight"] = length / speed
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
        return gpd.GeoDataFrame(copy, geometry=copy["geometry"]).set_crs(OX_CRS).to_crs(self.local_crs)

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
        G = nx.MultiGraph()
        for index in list(sorted_nodes.index)[:-1]:
            n1 = sorted_nodes.loc[index]
            n2 = sorted_nodes.loc[index + 1]
            d = n2["distance"] - n1["distance"]
            id1 = n1["hash"]  # hash(n1['geometry'])
            id2 = n2["hash"]  # hash(n1['geometry'])
            speed = METERS_IN_KILOMETER * self.speed[transport_type] / MINUTES_IN_HOUR
            G.add_edge(id1, id2, weight=d / speed, transport_type=transport_type)
            G.nodes[id1]["x"] = n1["geometry"].x
            G.nodes[id1]["y"] = n1["geometry"].y
            G.nodes[id2]["x"] = n2["geometry"].x
            G.nodes[id2]["y"] = n2["geometry"].y
        return G

    def _get_transport_graphs(
        self, transport_type: Literal["subway", "tram", "trolleybus", "bus"]
    ) -> list[nx.MultiGraph]:
        """Get transport routes graphs for the given transport_type"""
        routes = self._get_routes(self.city_geometry.bounds, transport_type)
        graphs = []
        for i in routes.index:
            df = pd.DataFrame(routes.loc[i, "members"])
            nodes_df = df.loc[lambda x: x["type"] == "node"].copy()
            ways_df = df.loc[lambda x: x["type"] == "way"].copy().rename(columns={"geometry": "coordinates"})
            nodes_gdf = self._nodes_to_gdf(nodes_df)
            ways_gdf = self._ways_to_gdf(ways_df)
            graphs.append(self._graph_from_route(nodes_gdf, ways_gdf, transport_type))
        return graphs

    def get_graph(self):
        """Returns intermodal graph for the city geometry bounds"""
        G_walk: nx.MultiDiGraph = self._get_walk_graph()
        # transport_types = ['subway', 'tram', 'trolleybus', 'bus']
        G_subways: list[nx.MultiGraph] = self._get_transport_graphs("subway")
        G_trams: list[nx.MultiGraph] = self._get_transport_graphs("tram")
        G_trolleybuses: list[nx.MultiGraph] = self._get_transport_graphs("trolleybus")
        G_buses: list[nx.MultiGraph] = self._get_transport_graphs("bus")

        walk_nodes, _ = ox.graph_to_gdfs(G_walk)
        walk_nodes.set_crs(epsg=self.local_crs, allow_override=True, inplace=True)

        G_transport = nx.compose_all(graphs=[*G_subways, *G_trams, *G_trolleybuses, *G_buses]).to_directed()
        transport_nodes = gpd.GeoDataFrame(G_transport.nodes(data=True), columns=["id", "data"])
        transport_nodes["geometry"] = transport_nodes["data"].apply(lambda d: Point(d["x"], d["y"]))
        transport_nodes.set_geometry("geometry", inplace=True)
        transport_nodes.set_crs(epsg=self.local_crs, inplace=True)

        sjoin = transport_nodes.sjoin_nearest(walk_nodes, how="left", distance_col="distance")
        intermodal_graph = nx.compose(G_walk, G_transport)
        intermodal_graph.graph = {"epsg": self.local_crs}
        for i in sjoin.index:
            gs = sjoin.loc[i]
            transport_node = gs["id"]
            walk_node = gs["index_right"]
            distance = gs["distance"]
            speed = 1000 * 4 / 60
            weight = distance / speed
            intermodal_graph.add_edge(transport_node, walk_node, weight=weight)
            intermodal_graph.add_edge(walk_node, transport_node, weight=weight + 5)

        return intermodal_graph
