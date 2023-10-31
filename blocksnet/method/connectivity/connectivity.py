from ..base_method import BaseMethod
import statistics
import geopandas as gpd


class Connectivity(BaseMethod):
    @staticmethod
    def plot(gdf: gpd.GeoDataFrame):
        gdf.plot(column="median", legend=True, cmap="cool").set_axis_off()

    def calculate(self):
        blocks_gdf = self.city_model.get_blocks_gdf().drop(columns=["population"])
        blocks_gdf["median"] = 0
        graph = self.city_model.graph
        for block in graph.nodes:
            out_edges = graph.out_edges(block, data="weight")
            filter_edges = filter(lambda edge: block != edge[1], out_edges)
            map_edges = map(lambda edge: edge[2], filter_edges)  # only weights left
            blocks_gdf.loc[block.id, "median"] = statistics.median(map_edges)  # median value set
        return blocks_gdf
