from functools import wraps
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from loguru import logger
from shapely.ops import polygonize
from .schemas import BoundariesSchema, LineObjectsSchema, PolygonObjectsSchema
from blocksnet.utils.validation import ensure_crs 
from blocksnet.machine_learning.classification import BlocksClassifier
from blocksnet.machine_learning.classification.blocks.common import BlockCategory
from shapely.geometry import (
    Point,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon
)
from .utils import clusters, make_convex_hulls, extend_roads_to_boundary, new_borderline, merge_empty_blocks, merge_invalid_blocks


def _validate_and_process_gdfs(func):
    @wraps(func)
    def wrapper(
        boundaries_gdf: gpd.GeoDataFrame,
        lines_gdf: gpd.GeoDataFrame | None,
        polygons_gdf: gpd.GeoDataFrame | None,
        *args,
        **kwargs,
    ):
        logger.info("Checking boundaries schema")
        boundaries_gdf = BoundariesSchema(boundaries_gdf)
        crs = boundaries_gdf.crs

        logger.info("Checking line objects schema")
        if lines_gdf is None or lines_gdf.empty:
            lines_gdf = LineObjectsSchema.create_empty(crs)
        else:
            lines_gdf = LineObjectsSchema(lines_gdf).explode("geometry", ignore_index=True)

        logger.info("Checking polygon objects schema")
        if polygons_gdf is None or polygons_gdf.empty:
            polygons_gdf = PolygonObjectsSchema.create_empty(crs)
        else:
            polygons_gdf = PolygonObjectsSchema(polygons_gdf).explode("geometry", ignore_index=True)

        ensure_crs(boundaries_gdf, lines_gdf, polygons_gdf)

        return func(boundaries_gdf, lines_gdf, polygons_gdf, *args, **kwargs)

    return wrapper


def _exclude_polygons(boundaries_gdf: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame):
    logger.info("Excluding polygon objects from blocks")
    polygons_gdf = gpd.GeoDataFrame(geometry=[polygons_gdf.union_all()], crs=polygons_gdf.crs)
    return boundaries_gdf.overlay(polygons_gdf, how="difference")


def _get_enclosures(boundaries_gdf: gpd.GeoDataFrame, lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Setting up enclosures")
    barriers = (
        pd.concat([lines_gdf.geometry, boundaries_gdf.boundary]).explode(ignore_index=True).reset_index(drop=True)
    )

    unioned = barriers.union_all()
    polygons = polygonize(unioned)
    enclosures = gpd.GeoSeries(list(polygons), crs=lines_gdf.crs)
    _, enclosure_idxs = enclosures.representative_point().sindex.query(boundaries_gdf.geometry, predicate="contains")
    enclosures = enclosures.iloc[np.unique(enclosure_idxs)]
    enclosures = enclosures.rename("geometry").reset_index()
    return enclosures


def _fill_holes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Filling holes inside the blocks")
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.boundary
    gdf = gdf.explode(index_parts=False)
    gdf.geometry = gdf.geometry.map(lambda g: shapely.Polygon(g) if g.geom_type != "Point" else np.nan)
    gdf = gdf.dropna(subset="geometry").reset_index(drop=True)
    return gdf


def _filter_overlapping(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Filtering overlapping geometries")
    gdf = gdf.copy()
    overlaps = gdf.geometry.sindex.query(gdf.geometry, predicate="contains")
    contained_geoms_idxs = {y for x, y in zip(overlaps[0], overlaps[1]) if x != y}
    gdf = gdf.drop(list(contained_geoms_idxs)).reset_index(drop=True)
    return gdf


def _filter_bottlenecks(gdf: gpd.GeoDataFrame, min_block_width: float) -> gpd.GeoDataFrame:
    logger.info("Filtering bottlenecks and small blocks")

    def _filter_bottlenecks_helper(poly):
        try:
            return poly.intersection(poly.buffer(-min_block_width / 2).buffer(min_block_width / 2, join_style=2))
        except:
            return poly

    gdf.geometry = gdf["geometry"].apply(_filter_bottlenecks_helper)
    gdf = gdf[~gdf["geometry"].is_empty]
    gdf = gdf.explode(ignore_index=True)
    gdf = gdf[gdf.type == "Polygon"]
    return gdf

def _filter_water_objects(
    water_gdf: gpd.GeoDataFrame | None,
    blocks_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
    logger.info("Filtering water objects")
    if water_gdf is None or water_gdf.empty or len(blocks_gdf) < 2:
        return gpd.GeoDataFrame(geometry=[], crs=water_gdf.crs)
    
    water_gdf = water_gdf.reset_index(drop=True)
    joined = gpd.sjoin(water_gdf, blocks_gdf, how="inner", predicate="intersects")

    water_counts = joined.groupby(level=0).size()
    valid_water_indices = water_counts[water_counts >= 2].index
    result = water_gdf.loc[valid_water_indices]
    return result

def _classify(blocks_gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("Classifying blocks")
    classifier = BlocksClassifier()
    blocks_gdf_classified = classifier.evaluate(blocks_gdf)
    blocks_gdf_classified['geometry'] = blocks_gdf.geometry
    blocks_gdf_classified = gpd.GeoDataFrame(blocks_gdf_classified, geometry=blocks_gdf_classified.geometry, crs=blocks_gdf.crs)
    return blocks_gdf_classified

def _filter_large_buildings_blocks(
    blocks_gdf : gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    logger.info("Filtering large blocks with buildings")

    if 'building_id' not in buildings_gdf.columns:
        buildings_gdf = buildings_gdf.reset_index(drop=True)
        buildings_gdf['building_id'] = buildings_gdf.index.astype(int)
    
    if 'block_id' not in blocks_gdf.columns:
        blocks_gdf = blocks_gdf.reset_index(drop=True)
        blocks_gdf['block_id'] = blocks_gdf.index.astype(int)

    large_blocks = blocks_gdf[blocks_gdf['category'] == BlockCategory.LARGE].copy()

    if 'block_id' not in large_blocks.columns:
        large_blocks = large_blocks.reset_index(drop=True)
        large_blocks['block_id'] = large_blocks.index.astype(int)

    joined = gpd.sjoin(buildings_gdf, large_blocks, how='inner', predicate='within')

    if not joined.empty:
        grouped = joined.groupby('block_id').agg(
            building_ids=('building_id', list)
        )
        result_blocks = large_blocks.merge(grouped, on='block_id', how='left')
    else:
        result_blocks = large_blocks.copy()
        result_blocks['building_ids'] = None

    result_blocks['building_ids'] = result_blocks['building_ids'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    large_blocks_with_buildings = result_blocks[result_blocks['building_ids'].map(len) > 0]

    selected_block_ids = large_blocks_with_buildings['block_id'].tolist()
    other_blocks = blocks_gdf[~blocks_gdf['block_id'].isin(selected_block_ids)].copy()
    return large_blocks_with_buildings, other_blocks


def _cut_to_blocks(
    roads_with_bounds : gpd.GeoDataFrame,
    water_objects : gpd.GeoDataFrame | None, 
    big_polygon : Polygon) -> gpd.GeoDataFrame:

    from blocksnet.blocks.cutting import preprocess_urban_objects

    if isinstance(water_objects, gpd.GeoDataFrame) and water_objects.empty:
        water_objects = None

    lines_gdf, polygons_gdf = preprocess_urban_objects(roads_with_bounds, None, water_objects)
    bound_gdf = gpd.GeoDataFrame(geometry=[big_polygon], crs=roads_with_bounds.crs)
    blocks_res = cut_urban_blocks(boundaries_gdf=bound_gdf, lines_gdf=lines_gdf, polygons_gdf=polygons_gdf, fill_holes=False)
    return blocks_res


def _cut_large_block(
    block_geometry : Polygon | MultiPolygon | LineString | MultiLineString,
    buildings_gdf : gpd.GeoDataFrame,
    polygons_gdf : gpd.GeoDataFrame,
    lines_gdf : gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:

    block_geometry_gdf = gpd.GeoDataFrame(geometry=[block_geometry], crs=lines_gdf.crs)

    try:

        within_buildings = buildings_gdf[buildings_gdf.geometry.within(block_geometry)]
        intersecting_polygons = polygons_gdf[polygons_gdf.geometry.intersects(block_geometry)]
        intersecting_lines = lines_gdf[lines_gdf.geometry.intersects(block_geometry)]

        if intersecting_polygons.empty:
            intersecting_polygons = None

        clusters_gdf = clusters(within_buildings)
        hull_gdf = make_convex_hulls(clusters_gdf)
        if hull_gdf.empty:
            hull_gdf = block_geometry_gdf.copy()
        new_roads = extend_roads_to_boundary(intersecting_lines, hull_gdf)
        
        new_roads['new_geometry'] = new_roads['geometry']

        for idx, row in new_roads.iterrows():
            geom = row['geometry']
            if geom is None or geom.is_empty:
                new_roads.at[idx, 'new_geometry'] = geom 
            else:
                new_roads.at[idx, 'new_geometry'] = new_borderline(geom, within_buildings)

        new_roads['geometry'] = new_roads['new_geometry']
        new_roads = new_roads.drop(columns=['new_geometry'])
        all_roads = gpd.GeoDataFrame(pd.concat([new_roads, intersecting_lines], ignore_index=True), geometry="geometry", crs=intersecting_lines.crs)

        polygon_boundaries = hull_gdf.boundary
        boundaries_gdf = gpd.GeoDataFrame(geometry=polygon_boundaries, crs=hull_gdf.crs)
        new_roads_with_bounds = gpd.GeoDataFrame(pd.concat([all_roads, boundaries_gdf, intersecting_lines], ignore_index=True), geometry="geometry", crs=hull_gdf.crs)
        new_roads_with_bounds = new_roads_with_bounds.reset_index(drop=True)
        new_roads_with_bounds = new_roads_with_bounds[['geometry']]

        blocks_res = _cut_to_blocks(new_roads_with_bounds, intersecting_polygons, block_geometry)

        blocks_gdf_classified = _classify(blocks_res)
        combined_gdf = merge_invalid_blocks(blocks_gdf_classified)
        combined_gdf = merge_empty_blocks(combined_gdf, buildings_gdf)
        return combined_gdf
    
    except Exception as e:
        logger.warning(f"Failed to process block: {e}")
        return block_geometry_gdf


@_validate_and_process_gdfs
def cut_urban_blocks(
    boundaries_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame | None,
    polygons_gdf: gpd.GeoDataFrame | None,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    fill_holes: bool = True,
    min_block_width: float | None = None
) -> gpd.GeoDataFrame:
    
    boundaries_gdf['block_id'] = range(len(boundaries_gdf))
    blocks_gdf = _get_enclosures(boundaries_gdf, lines_gdf)

    valid_polygons = _filter_water_objects(polygons_gdf, blocks_gdf)
    boundaries_gdf = _exclude_polygons(boundaries_gdf, valid_polygons)

    if fill_holes:
        blocks_gdf = _fill_holes(blocks_gdf)

    blocks_gdf = _filter_overlapping(blocks_gdf)

    if min_block_width is not None:
        blocks_gdf = _filter_bottlenecks(blocks_gdf, min_block_width)
        
    final_gdf = blocks_gdf.reset_index(drop=True)[["geometry"]]
    
    if buildings_gdf is not None and not buildings_gdf.empty:

        buildings_gdf['building_id'] = range(len(buildings_gdf))

        logger.info("Splitting large blocks")

        blocks_gdf_classified = _classify(blocks_gdf)
        large_blocks, other_blocks = _filter_large_buildings_blocks(blocks_gdf_classified, buildings_gdf)

        results = large_blocks.apply(
        lambda row: _cut_large_block(
            block_geometry=row.geometry,
            buildings_gdf=buildings_gdf,
            polygons_gdf=polygons_gdf,
            lines_gdf=lines_gdf
        ), axis=1)

        cut_blocks_list = results.tolist()  # список GeoDataFrame'ов
        cut_blocks_gdf = gpd.GeoDataFrame(pd.concat(cut_blocks_list, ignore_index=True), crs=cut_blocks_list[0].crs)

        final_gdf = gpd.GeoDataFrame(
        pd.concat([cut_blocks_gdf, other_blocks], ignore_index=True),
        crs=cut_blocks_gdf.crs
        )
        final_gdf = final_gdf.reset_index(drop=True)[["geometry"]]

        blocks_gdf_classified = _classify(final_gdf)
        combined_gdf = merge_invalid_blocks(blocks_gdf_classified)
        combined_gdf = merge_empty_blocks(combined_gdf, buildings_gdf)
        final_gdf = combined_gdf.copy()
    
    valid_polygons = _filter_water_objects(polygons_gdf, final_gdf)
    boundaries_gdf = _exclude_polygons(boundaries_gdf, valid_polygons)

    final_gdf = final_gdf.reset_index(drop=True)[["geometry"]]
    logger.success("Blocks are successfully cut!")
    return final_gdf
