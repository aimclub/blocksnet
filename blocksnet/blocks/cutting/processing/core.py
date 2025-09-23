import geopandas as gpd
from loguru import logger
from blocksnet.utils.validation import ensure_crs
from .schemas import validate_and_preprocess_gdfs
from .cut import cut_blocks
from .split import split_blocks
from blocksnet.blocks.classification import BlocksClassifier


def cut_urban_blocks(
    boundaries_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame | None,
    polygons_gdf: gpd.GeoDataFrame | None,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    classifier: BlocksClassifier | None = None,
) -> gpd.GeoDataFrame:

    boundaries_gdf, lines_gdf, polygons_gdf, buildings_gdf = validate_and_preprocess_gdfs(
        boundaries_gdf, lines_gdf, polygons_gdf, buildings_gdf
    )

    if classifier is not None:
        if not isinstance(classifier, BlocksClassifier):
            raise TypeError("Classifier must be an instance of blocksnet.blocks.classification.BlocksClassifier")

    blocks_gdf = cut_blocks(boundaries_gdf, lines_gdf, polygons_gdf)
    logger.success(f"{len(blocks_gdf)} blocks are successfully cut")

    if classifier is not None:
        blocks_gdf = split_blocks(blocks_gdf, lines_gdf, polygons_gdf, buildings_gdf, classifier)

    return gpd.GeoDataFrame(blocks_gdf.reset_index(drop=True)[["geometry"]])
