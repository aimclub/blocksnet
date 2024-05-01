import geopandas as gpd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from ...models import Block, ServiceType
from ..base_method import BaseMethod

ACCESSIBILITY_TO_COLUMN = "accessibility_to"
ACCESSIBILITY_FROM_COLUMN = "accessibility_from"


class Accessibility(BaseMethod):
    """Class provides methods for block accessibility assessment"""

    def plot(self, gdf: gpd.GeoDataFrame, figsize: tuple[int] = (10, 10)):
        # service_type = self.city_model[service_type]
        fig = plt.figure(figsize=figsize)
        grid = GridSpec(1, 2)

        ax_to = fig.add_subplot(grid[0, 0])
        gdf.plot(ax=ax_to, column=ACCESSIBILITY_TO_COLUMN, cmap="Greens", vmax=60, legend=True).set_axis_off()
        ax_to.set_title("To other blocks")
        ax_to.set_axis_off()

        ax_from = fig.add_subplot(grid[0, 1])
        gdf.plot(ax=ax_from, column=ACCESSIBILITY_FROM_COLUMN, cmap="Blues", vmax=60, legend=True).set_axis_off()
        ax_from.set_title("From other blocks")
        ax_from.set_axis_off()

    def get_context(
        self, selected_blocks: list[Block] | list[int], service_type: ServiceType | str
    ) -> gpd.GeoDataFrame:
        service_type = self.city_model[service_type]
        accessibility = service_type.accessibility
        context_blocks_ids = set()
        for selected_block in selected_blocks:
            accessible_blocks_gdf = self.calculate(selected_block)
            accessible_blocks_gdf = accessible_blocks_gdf.query(
                f"accessibility_to<={accessibility} | accessibility_from<={accessibility}"
            )
            context_blocks_ids.update(accessible_blocks_gdf.index)
        context_blocks_gdf = self.city_model.get_blocks_gdf(simplify=True)
        context_blocks_gdf = context_blocks_gdf.loc[list(context_blocks_ids)]
        return context_blocks_gdf[["geometry"]]

    def calculate(self, block: Block | int) -> gpd.GeoDataFrame:
        blocks_list = map(lambda b: {"id": b.id, "geometry": b.geometry}, self.city_model.blocks)
        blocks_gdf = gpd.GeoDataFrame(blocks_list).set_crs(epsg=self.city_model.epsg)
        blocks_gdf[ACCESSIBILITY_TO_COLUMN] = blocks_gdf["id"].apply(lambda b: self.city_model[block, b])
        blocks_gdf[ACCESSIBILITY_FROM_COLUMN] = blocks_gdf["id"].apply(lambda b: self.city_model[b, block])
        return blocks_gdf
