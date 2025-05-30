import geopandas as gpd


INDEX_LEFT_COLUMN = "index_left"
INDEX_RIGHT_COLUMN = "index_right"
INTERSECTION_AREA_COLUMN = "intersection_area"
SHARE_LEFT_COLUMN = "share_left"
SHARE_RIGHT_COLUMN = "share_right"


def sjoin_intersections(left_gdf: gpd.GeoDataFrame, right_gdf: gpd.GeoDataFrame):

    left_gdf = left_gdf.copy()
    right_gdf = right_gdf.copy()

    left_gdf[INDEX_LEFT_COLUMN] = left_gdf.index
    right_gdf[INDEX_RIGHT_COLUMN] = right_gdf.index

    overlay_gdf = gpd.overlay(
        left_gdf,
        right_gdf,
        how="intersection",
        keep_geom_type=False,
    )
    overlay_gdf[INTERSECTION_AREA_COLUMN] = overlay_gdf.area

    left_areas = left_gdf.geometry.area
    right_areas = right_gdf.geometry.area
    overlay_areas = overlay_gdf[INTERSECTION_AREA_COLUMN]

    overlay_gdf[SHARE_LEFT_COLUMN] = overlay_areas / overlay_gdf[INDEX_LEFT_COLUMN].map(left_areas)
    overlay_gdf[SHARE_RIGHT_COLUMN] = overlay_areas / overlay_gdf[INDEX_RIGHT_COLUMN].map(right_areas)

    return overlay_gdf
