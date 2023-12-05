import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

import pyproj
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_crs_info
from pyproj import CRS


def verbose_print(text, verbose=True):
    if verbose:
        print(text)


def get_projected_crs(poly):
    """
    Returns a local projected CRS for a given polygon.
    If no matching local CRS found, returns EPSG:3857.

    Attributes
    ----------
    poly: shapely.Polygon or shapely.MultiPolygon
        A polygon of a place.

    Returns
    -------
    projected_crs: pyproj.CRS
        Projected CRS.
    """

    try:
        coords = [list(set(x)) for x in poly.envelope.boundary.coords.xy]

        area_of_interest = AreaOfInterest(
            west_lon_degree=coords[0][0],
            east_lon_degree=coords[0][1],
            north_lat_degree=coords[1][0],
            south_lat_degree=coords[1][1],
        )

        utm_crs_list = query_crs_info(
            pj_types=pyproj.enums.PJType.PROJECTED_CRS,
            area_of_interest=area_of_interest,
        )

        projected_crs = CRS.from_epsg(utm_crs_list[0].code)

    except:
        projected_crs = CRS(3857)

    return projected_crs


def fill_holes(gdf):
    """
    Fills holes in geometries of a given GeoDataFrame

    Attributes
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with geometries to be filled.

    Returns
    -------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with filled geometries.
    """

    gdf["geometry"] = gdf["geometry"].boundary
    gdf = gdf.explode(index_parts=False)
    gdf["geometry"] = gdf["geometry"].map(lambda x: Polygon(x) if x.geom_type != "Point" else np.nan)
    gdf = gdf.dropna(subset="geometry").reset_index(drop=True).to_crs(4326)

    return gdf


def drop_contained_geometries(gdf):
    """
    Drops geometries that are contained inside other geometries.

    Attributes
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with contained geometries.

    Returns
    -------
    gdf: gpd.GeoDataFrame
        GeoDataFrame without contained geometries.
    """

    gdf = gdf.reset_index(drop=True)

    overlaps = gdf["geometry"].sindex.query(gdf["geometry"], predicate="contains")
    contains_dict = {x: [] for x in overlaps[0]}
    for x, y in zip(overlaps[0], overlaps[1]):
        if x != y:
            contains_dict[x].append(y)

    contained_geoms_idxs = list({x for v in contains_dict.values() for x in v})
    gdf = gdf.drop(contained_geoms_idxs)
    gdf = gdf.reset_index(drop=True)

    return gdf


def filter_bottlenecks(gdf, projected_crs, min_width=40):
    """
    Divides geometries in narrow places and removes small geometries.

    Attributes
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with contained geometries.

    projected_crs: int or pyproj.CRS
        Metric projected CRS of a given area.

    min_width: int or float
        Minimum allowed width of geometries in resulting GeoDataFrame.

    Returns
    -------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with processed geometries.
    """

    def _filter_bottlenecks_helper(poly, min_width=40):
        try:
            return poly.intersection(poly.buffer(-min_width / 2).buffer(min_width / 2, join_style=2))
        except:
            return poly

    gdf["geometry"] = (
        gdf.to_crs(projected_crs)["geometry"]
        .map(lambda x: _filter_bottlenecks_helper(x, min_width))
        .set_crs(projected_crs)
        .to_crs(4326)
    )
    gdf = gdf[gdf["geometry"] != Polygon()]
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.type == "Polygon"]

    if "area" in gdf.columns:
        gdf["area"] = gdf.to_crs(projected_crs).area

    return gdf
