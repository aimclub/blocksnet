import shapely
import geopandas as gpd
from blocksnet.config import log_config
from .schemas import BlocksSchema

ASPECT_RATIO_COLUMN = "aspect_ratio"


def _calculate_aspect_ratio(polygon: shapely.Polygon) -> float | None:
    rectangle = polygon.minimum_rotated_rectangle
    rectangle_coords = list(rectangle.exterior.coords)

    side_lengths = [
        (
            (rectangle_coords[i][0] - rectangle_coords[i - 1][0]) ** 2
            + (rectangle_coords[i][1] - rectangle_coords[i - 1][1]) ** 2
        )
        ** 0.5
        for i in range(1, 5)
    ]

    length_1, length_2 = side_lengths[0], side_lengths[1]
    length_max = max(length_1, length_2)
    length_min = min(length_1, length_2)

    if length_min == 0:
        return None
    return length_max / length_min


def calculate_aspect_ratio(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks_gdf = BlocksSchema(blocks_gdf)

    if log_config.disable_tqdm:
        apply = blocks_gdf.geometry.apply
    else:
        apply = blocks_gdf.geometry.progress_apply

    blocks_gdf[ASPECT_RATIO_COLUMN] = apply(_calculate_aspect_ratio)

    return blocks_gdf
