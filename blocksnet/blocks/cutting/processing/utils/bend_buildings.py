import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, Point
from shapely.ops import unary_union
import shapely.ops as ops


def _get_inters_buildings(line: LineString, build_gdf: gpd.GeoDataFrame):

    """Get inters buildings.

    Parameters
    ----------
    line : LineString
        Description.
    build_gdf : gpd.GeoDataFrame
        Description.

    """
    build_gdf_exploded = build_gdf.explode(index_parts=False)
    build_gdf_exploded["intersects"] = build_gdf_exploded.geometry.intersects(line)
    intersected = build_gdf_exploded[build_gdf_exploded["intersects"]].drop(columns=["intersects"])
    non_intersected = build_gdf_exploded[~build_gdf_exploded["intersects"]].drop(columns=["intersects"])

    return intersected, non_intersected


def _line_around_polygon(line: LineString, polygon: Polygon):

    """Line around polygon.

    Parameters
    ----------
    line : LineString
        Description.
    polygon : Polygon
        Description.

    """
    inner_part = line.intersection(polygon)
    outer_parts = line.difference(polygon)

    if outer_parts.is_empty:
        return LineString()

    coords = []
    if isinstance(inner_part, LineString):
        coords = list(inner_part.coords)
    elif isinstance(inner_part, MultiLineString):
        if inner_part.geoms:
            coords = list(inner_part.geoms[0].coords) + list(inner_part.geoms[-1].coords)

    if len(coords) < 2:
        return line

    start_point = Point(coords[0])
    end_point = Point(coords[-1])

    boundary = polygon.exterior

    start_on_boundary = boundary.interpolate(boundary.project(start_point))
    end_on_boundary = boundary.interpolate(boundary.project(end_point))

    try:
        subpath = ops.substring(
            boundary, boundary.project(start_on_boundary), boundary.project(end_on_boundary), normalized=False
        )
    except ValueError:
        return line

    new_line_parts = []

    if outer_parts.geom_type == "MultiLineString":
        for part in outer_parts.geoms:
            new_line_parts.append(part)
    else:
        new_line_parts.append(outer_parts)

    new_line_parts.append(subpath)

    result_line = ops.linemerge(new_line_parts)

    return result_line if not result_line.is_empty else line


def _min_distance_to_buildings(inters_gdf: gpd.GeoDataFrame, non_inters_gdf: gpd.GeoDataFrame, buffer_dist: float = 20):
    """Min distance to buildings.

    Parameters
    ----------
    inters_gdf : gpd.GeoDataFrame
        Description.
    non_inters_gdf : gpd.GeoDataFrame
        Description.
    buffer_dist : float, default: 20
        Description.

    """
    results = []
    non_inters_gdf = non_inters_gdf.explode(index_parts=False)

    for idx, inters_row in inters_gdf.iterrows():
        inters_geom = inters_row.geometry

        buffer = inters_geom.buffer(buffer_dist)

        nearby_buildings = non_inters_gdf[
            (non_inters_gdf.intersects(buffer)) | (non_inters_gdf.distance(buffer) < 1e-6)
        ]

        if nearby_buildings.empty:
            continue

        distances = nearby_buildings.distance(inters_geom)
        min_dist = distances.min()
        results.append(min_dist)

    if len(results) == 0:
        return buffer_dist

    return min(results)


def bend_buildings(line: LineString, build_gdf: gpd.GeoDataFrame):

    """Bend buildings.

    Parameters
    ----------
    line : LineString
        Description.
    build_gdf : gpd.GeoDataFrame
        Description.

    """
    inters_build_gdf, non_inters_builds = _get_inters_buildings(line, build_gdf[["geometry"]])
    buffer_distance = _min_distance_to_buildings(inters_build_gdf, non_inters_builds)

    inters_build_gdf["buffer_geometry"] = inters_build_gdf.apply(
        lambda row: row.geometry.buffer(distance=buffer_distance, quad_segs=2), axis=1
    )

    multy = unary_union(inters_build_gdf["buffer_geometry"])
    try:
        list_pols = list(multy.geoms)
    except:
        list_pols = [multy]

    modified_border_line = line

    for pol in list_pols:
        modified_border_line = _line_around_polygon(modified_border_line, pol)

    return modified_border_line
