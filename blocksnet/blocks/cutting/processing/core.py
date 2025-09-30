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
    """Generate urban blocks within territory boundaries.

    Parameters
    ----------
    boundaries_gdf : geopandas.GeoDataFrame
        Territorial boundaries to cut into blocks.
    lines_gdf : geopandas.GeoDataFrame or None
        Linear obstacles such as streets and rivers.
    polygons_gdf : geopandas.GeoDataFrame or None
        Polygonal obstacles that should subtract from the territory.
    buildings_gdf : geopandas.GeoDataFrame or None, default=None
        Optional building footprints used for splitting with a classifier.
    classifier : BlocksClassifier or None, default=None
        Optional classifier used to split blocks based on predicted
        categories.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the resulting block geometries.

    Raises
    ------
    TypeError
        If *classifier* is provided but not an instance of
        :class:`BlocksClassifier`.
    """

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
