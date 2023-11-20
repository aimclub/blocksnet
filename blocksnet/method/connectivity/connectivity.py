from ..base_method import BaseMethod
import statistics
import geopandas as gpd


class Connectivity(BaseMethod):
    @staticmethod
    def plot(gdf: gpd.GeoDataFrame):
        gdf.plot(column="median", legend=True, cmap="cool").set_axis_off()

    def calculate(self):
        blocks_gdf = self.city_model.get_blocks_gdf()[["geometry"]]
        blocks_gdf["median"] = 0
        for block in self.city_model.blocks:
            out_edges = self.city_model.get_out_edges(block)
            filter_edges = filter(lambda edge: block != edge[1], out_edges)
            map_edges = map(lambda edge: edge[2], filter_edges)  # only weights left
            blocks_gdf.loc[block.id, "median"] = statistics.median(map_edges)  # median value set
        return blocks_gdf
