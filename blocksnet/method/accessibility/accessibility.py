import geopandas as gpd
from ..base_method import BaseMethod
from ...models import Block
from ...models import ServiceType


class Accessibility(BaseMethod):
    """Class provides methods for block accessibility assessment"""

    def plot(self, gdf: gpd.GeoDataFrame, service_type: ServiceType | str = None):
        if service_type is not None and not isinstance(service_type, ServiceType):
            service_type = self.city_model[service_type]
        if service_type:
            gdf = gdf.copy()
            gdf["accessible"] = gdf["distance"] <= service_type.accessibility
            gdf.plot(column="accessible", cmap="cool", legend=True).set_axis_off()
        else:
            gdf.plot(column="distance", cmap="cool", legend=True, vmin=0, vmax=60).set_axis_off()

    def calculate(self, block: Block):
        blocks_list = map(lambda b: {"id": b.id, "geometry": b.geometry}, self.city_model.blocks)
        blocks_gdf = gpd.GeoDataFrame(blocks_list).set_crs(epsg=self.city_model.epsg)
        blocks_gdf["distance"] = blocks_gdf["id"].apply(
            lambda b: self.city_model.graph[block][self.city_model[b]]["weight"]
        )
        return blocks_gdf
