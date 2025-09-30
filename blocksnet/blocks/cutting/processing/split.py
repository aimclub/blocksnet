from typing import cast
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from loguru import logger
from blocksnet.blocks.classification import BlocksClassifier
from blocksnet.enums import BlockCategory
from blocksnet.config import log_config
from .cut import cut_blocks
from .utils.clustering import clusterize, make_convex_hulls
from .utils.extend_lines import extend_lines
from .utils.bend_buildings import bend_buildings
from .utils.merge_blocks import merge_empty_blocks, merge_invalid_blocks


def _classify(blocks_gdf: gpd.GeoDataFrame, classifier: BlocksClassifier) -> gpd.GeoDataFrame:
    """Classify.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.
    classifier : BlocksClassifier
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    blocks_df = classifier.run(blocks_gdf)
    return blocks_gdf.join(blocks_df)


def _split_urban_block(
    block_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    classifier: BlocksClassifier,
) -> gpd.GeoDataFrame:

    """Split urban block.

    Parameters
    ----------
    block_gdf : gpd.GeoDataFrame
        Description.
    lines_gdf : gpd.GeoDataFrame
        Description.
    polygons_gdf : gpd.GeoDataFrame
        Description.
    buildings_gdf : gpd.GeoDataFrame
        Description.
    classifier : BlocksClassifier
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    buildings_gdf = buildings_gdf.sjoin(block_gdf, predicate="within")
    lines_gdf = lines_gdf.sjoin(block_gdf, predicate="intersects")

    try:

        clusters_gdf = clusterize(buildings_gdf)
        hull_gdf = make_convex_hulls(clusters_gdf)
        if hull_gdf.empty:
            hull_gdf = block_gdf.copy()
        new_roads = extend_lines(lines_gdf, hull_gdf)

        new_roads["new_geometry"] = new_roads["geometry"]

        for idx, row in new_roads.iterrows():
            geom = row["geometry"]
            if geom is None or geom.is_empty:
                new_roads.at[idx, "new_geometry"] = geom
            else:
                new_roads.at[idx, "new_geometry"] = bend_buildings(geom, buildings_gdf)

        new_roads["geometry"] = new_roads["new_geometry"]
        new_roads = new_roads.drop(columns=["new_geometry"])
        all_roads = gpd.GeoDataFrame(
            pd.concat([new_roads, lines_gdf], ignore_index=True), geometry="geometry", crs=lines_gdf.crs
        )

        polygon_boundaries = hull_gdf.boundary
        boundaries_gdf = gpd.GeoDataFrame(geometry=polygon_boundaries, crs=hull_gdf.crs)
        new_roads_with_bounds = gpd.GeoDataFrame(
            pd.concat([all_roads, boundaries_gdf, lines_gdf], ignore_index=True), geometry="geometry", crs=hull_gdf.crs
        )
        new_roads_with_bounds = new_roads_with_bounds.reset_index(drop=True)
        new_roads_with_bounds = new_roads_with_bounds[["geometry"]]

        blocks_res = cut_blocks(block_gdf, new_roads_with_bounds, polygons_gdf)

        blocks_gdf_classified = _classify(blocks_res, classifier)
        combined_gdf = merge_invalid_blocks(blocks_gdf_classified)
        combined_gdf = merge_empty_blocks(combined_gdf, buildings_gdf)
        return combined_gdf

    except Exception as e:
        logger.warning(f"Failed to process block: {e}")
        return block_gdf


def _get_split_candidates(
    blocks_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame, classifier: BlocksClassifier
) -> gpd.GeoDataFrame:
    """Get split candidates.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.
    buildings_gdf : gpd.GeoDataFrame
        Description.
    classifier : BlocksClassifier
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    sjoin_gdf = blocks_gdf.sjoin(buildings_gdf)

    blocks_ids = sjoin_gdf.index.unique()
    blocks_gdf = blocks_gdf.loc[blocks_ids].copy()

    blocks_df = classifier.run(blocks_gdf)
    blocks_df = blocks_df[blocks_df.category == BlockCategory.LARGE]
    blocks_ids = blocks_df.index.unique()
    blocks_gdf = blocks_gdf.loc[blocks_ids].copy()

    return blocks_gdf


def split_blocks(
    blocks_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    classifier: BlocksClassifier,
) -> gpd.GeoDataFrame:

    """Split blocks.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.
    lines_gdf : gpd.GeoDataFrame
        Description.
    polygons_gdf : gpd.GeoDataFrame
        Description.
    buildings_gdf : gpd.GeoDataFrame
        Description.
    classifier : BlocksClassifier
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    if buildings_gdf.empty:
        logger.info("Empty buildings geodataframe. Splitting can't be done")
        return blocks_gdf

    candidates_gdf = _get_split_candidates(blocks_gdf, buildings_gdf, classifier)

    if candidates_gdf.empty:
        logger.info("No candidates were found for splitting")
        return blocks_gdf
    logger.info(f"{len(candidates_gdf)} candidates were found for splitting")

    split_gdfs = []
    disable_tqdm = log_config.disable_tqdm
    logger_level = log_config.logger_level
    for i in tqdm(candidates_gdf.index, disable_tqdm=disable_tqdm):
        log_config.set_disable_tqdm(True)
        log_config.set_logger_level("ERROR")

        block_gdf = candidates_gdf.loc[[i]]
        split_gdf = _split_urban_block(block_gdf, lines_gdf, polygons_gdf, buildings_gdf, classifier)
        split_gdfs.append(split_gdf)

        log_config.set_disable_tqdm(disable_tqdm)
        log_config.set_logger_level(logger_level)

    split_gdf = cast(gpd.GeoDataFrame, pd.concat(split_gdfs))
    logger.info(f"{len(candidates_gdf)} blocks are successfully splitted into {len(split_gdf)} blocks")

    blocks_gdf = blocks_gdf[~blocks_gdf.index.isin(candidates_gdf.index)]
    blocks_gdf = cast(gpd.GeoDataFrame, pd.concat([blocks_gdf, split_gdf]))

    logger.success(f"{len(blocks_gdf)} blocks overall")

    return blocks_gdf.reset_index(drop=True)
