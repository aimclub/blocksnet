import geopandas as gpd
import pandas as pd
from .schemas import BlocksSchema
from tqdm import tqdm
from loguru import logger
from shapely.geometry.base import BaseGeometry
from concurrent.futures import ThreadPoolExecutor
from ...config import log_config
from . import utils, const

AREA_COLUMN = "area"
MRR_AREA_COLUMN = "mrr_area"
LENGTH_COLUMN = "length"
AREA_TO_LENGTH_COLUMN = "area_to_length"
AREA_TO_MRR_AREA_COLUMN = "area_to_mrr_area"
BLOCK_ID_COLUMN = "block_id"

FETCH_FUNCS = [
    utils.fetch_other,
    utils.fetch_leisure,
    utils.fetch_landuse,
    utils.fetch_amenity,
    utils.fetch_buildings,
    utils.fetch_natural,
    utils.fetch_waterway,
    utils.fetch_highway,
    utils.fetch_path,
    utils.fetch_railway,
]


def _fetch_occupied_areas(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch occupied areas.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    geometry = blocks_gdf.buffer(const.BLOCKS_BUFFER).to_crs(4326).union_all()
    occupied_gdfs = []

    logger.info("Fetching OSM geometries")
    with ThreadPoolExecutor(max_workers=len(FETCH_FUNCS)) as executor:
        futures = {executor.submit(func, geometry): func for func in FETCH_FUNCS}
        occupied_gdfs = []
        for future in tqdm(futures, total=len(futures), disable=log_config.disable_tqdm):
            gdf = future.result()
            occupied_gdfs.append(gdf)

    occupied_gdf = pd.concat(occupied_gdfs).to_crs(blocks_gdf.crs)
    occupied_gdf = gpd.GeoDataFrame(geometry=[occupied_gdf.union_all()], crs=blocks_gdf.crs)
    return occupied_gdf


def _generate_features(vacant_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Generate features.

    Parameters
    ----------
    vacant_gdf : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    vacant_gdf = vacant_gdf.copy()

    logger.info("Generating geometries features")
    vacant_gdf[AREA_COLUMN] = vacant_gdf.area
    vacant_gdf[MRR_AREA_COLUMN] = vacant_gdf.geometry.apply(lambda g: g.minimum_rotated_rectangle.area)
    vacant_gdf[LENGTH_COLUMN] = vacant_gdf.geometry.apply(lambda g: g.length)
    vacant_gdf[AREA_TO_LENGTH_COLUMN] = vacant_gdf[AREA_COLUMN] / vacant_gdf[LENGTH_COLUMN]
    vacant_gdf[AREA_TO_MRR_AREA_COLUMN] = vacant_gdf[AREA_COLUMN] / vacant_gdf[MRR_AREA_COLUMN]

    return vacant_gdf


def _filter_vacant_areas(vacant_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter vacant areas.

    Parameters
    ----------
    vacant_gdf : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    vacant_gdf = vacant_gdf.copy()

    logger.info("Filtering geometries")
    vacant_gdf = vacant_gdf.loc[vacant_gdf[AREA_COLUMN] >= const.AREA_MIN]
    vacant_gdf = vacant_gdf.loc[vacant_gdf[AREA_TO_MRR_AREA_COLUMN] >= const.AREA_TO_MRR_AREA_MIN]
    vacant_gdf = vacant_gdf.loc[vacant_gdf[AREA_TO_LENGTH_COLUMN] >= const.AREA_TO_LENGTH_MIN]

    return vacant_gdf.reset_index(drop=True)


def _label_areas_with_blocks(vacant_gdf: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame):
    """Label areas with blocks.

    Parameters
    ----------
    vacant_gdf : gpd.GeoDataFrame
        Description.
    blocks_gdf : gpd.GeoDataFrame
        Description.

    """
    centroid_gdf = vacant_gdf.copy()
    centroid_gdf.geometry = centroid_gdf.representative_point()
    centroid_gdf = centroid_gdf.sjoin(blocks_gdf, predicate="within")
    vacant_gdf[BLOCK_ID_COLUMN] = centroid_gdf["index_right"]
    return vacant_gdf


def get_vacant_areas(blocks_gdf: gpd.GeoDataFrame, basic_filter: bool = True) -> gpd.GeoDataFrame:
    """Get vacant areas.

    Parameters
    ----------
    blocks_gdf : gpd.GeoDataFrame
        Description.
    basic_filter : bool, default: True
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    blocks_gdf = BlocksSchema(blocks_gdf)
    occupied_gdf = _fetch_occupied_areas(blocks_gdf)
    vacant_gdf = blocks_gdf.overlay(occupied_gdf, how="difference")
    vacant_gdf = vacant_gdf.explode("geometry", ignore_index=True)
    vacant_gdf = _generate_features(vacant_gdf)
    if basic_filter:
        vacant_gdf = _filter_vacant_areas(vacant_gdf)
    _label_areas_with_blocks(vacant_gdf, blocks_gdf)
    return vacant_gdf
