import geopandas as gpd
import osmnx as ox
from pydantic import BaseModel, InstanceOf, field_validator

from ..models.service_type import ServiceType


KMH_TO_MM = 1_000 / 60


class Fetcher(BaseModel):

    units: InstanceOf[gpd.GeoDataFrame]

    @field_validator("units", mode="after")
    @staticmethod
    def validate_units(gdf):
        assert gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all(), "geometry should be Polygon or MultiPolygon"
        return gdf[["geometry"]]

    @property
    def crs(self):
        return self.units.crs

    @property
    def geometry(self):
        return self.units.to_crs(4326).unary_union

    def fetch_towns(self):
        gdf = ox.features_from_polygon(self.geometry, tags={"place": ["town", "city", "village", "hamlet"]})
        gdf = gdf.to_crs(self.crs)
        gdf.geometry = gdf.representative_point()
        gdf["population"] = gdf["population"].fillna(0)
        gdf["population"] = gdf["population"].apply(lambda s: str(s).split(" ", maxsplit=1)[0]).apply(int)
        gdf = gdf[gdf["population"] > 0].reset_index()
        return gdf[["geometry", "name", "population"]]

    def fetch_services(self, service_type: ServiceType | str, capacity: int = 250) -> gpd.GeoDataFrame:
        gdf = ox.features_from_polygon(self.geometry, tags=service_type.osm_tags)
        gdf = gdf.reset_index().to_crs(self.crs)
        gdf["capacity"] = capacity
        gdf.geometry = gdf.representative_point()
        return gdf[["geometry", "capacity"]]

    def fetch_graph(self, speed=60):
        speed *= KMH_TO_MM
        graph = ox.graph_from_polygon(self.geometry, network_type="drive", simplify=True)
        for _, _, data in graph.edges(data=True):
            data["time_min"] = data["length"] / speed
        return graph
