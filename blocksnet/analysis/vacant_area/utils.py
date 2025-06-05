import geopandas as gpd
import osmnx as ox
from typing import Callable
from . import const


def _fetch_osm(geometry, tags: dict, filter_func: Callable | None = None) -> gpd.GeoDataFrame | None:
    try:
        gdf = ox.features_from_polygon(geometry, tags=tags)
        if filter_func is not None:
            gdf = filter_func(gdf)
        return gdf.reset_index(drop=True)[["geometry"]]
    except (ValueError, AttributeError) as exc:
        return None


def _buffer_geometries(gdf: gpd.GeoDataFrame | None, buffer_size: int) -> gpd.GeoDataFrame:
    if gdf is None:
        return gdf
    current_crs = gdf.crs
    local_crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(local_crs)
    gdf.geometry = gdf.geometry.buffer(buffer_size)
    return gdf.to_crs(current_crs)


def fetch_other(geometry) -> gpd.GeoDataFrame | None:
    return _fetch_osm(geometry, {"man_made": True, "aeroway": True, "military": True})


def fetch_leisure(geometry) -> gpd.GeoDataFrame | None:
    return _fetch_osm(geometry, {"leisure": True})


def fetch_landuse(geometry) -> gpd.GeoDataFrame | None:
    filter_func = lambda gdf: gdf[gdf["landuse"] != "residential"]
    return _fetch_osm(geometry, {"landuse": True}, filter_func)


def fetch_amenity(geometry) -> gpd.GeoDataFrame | None:
    return _fetch_osm(geometry, {"amenity": True})


def fetch_buildings(geometry) -> gpd.GeoDataFrame | None:
    gdf = _fetch_osm(geometry, {"building": True})
    return _buffer_geometries(gdf, const.BUILDINGS_BUFFER)


def fetch_natural(geometry) -> gpd.GeoDataFrame | None:
    filter_func = lambda gdf: gdf[gdf["natural"] != "bay"]
    return _fetch_osm(geometry, {"natural": True}, filter_func)


def fetch_waterway(geometry) -> gpd.GeoDataFrame | None:
    return _fetch_osm(geometry, {"waterway": True})


def fetch_highway(geometry) -> gpd.GeoDataFrame | None:
    filter_func = lambda gdf: gdf[~gdf["highway"].isin(["path", "footway", "pedestrian"])]
    gdf = _fetch_osm(geometry, {"highway": True}, filter_func)
    return _buffer_geometries(gdf, const.ROADS_BUFFER)


def fetch_path(geometry) -> gpd.GeoDataFrame | None:
    gdf = _fetch_osm(geometry, {"highway": "path", "highway": "footway"})
    return _buffer_geometries(gdf, const.PATH_BUFFER)


def fetch_railway(geometry) -> gpd.GeoDataFrame | None:
    filter_func = lambda gdf: gdf[gdf["railway"] != "subway"]
    return _fetch_osm(geometry, {"railway": True}, filter_func)
