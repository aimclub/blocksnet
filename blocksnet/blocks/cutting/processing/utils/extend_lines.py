import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.strtree import STRtree


def _extend_line(p1: Point, p2: Point, length: float) -> LineString:
    """Extend line.

    Parameters
    ----------
    p1 : Point
        Description.
    p2 : Point
        Description.
    length : float
        Description.

    Returns
    -------
    LineString
        Description.

    """
    dx, dy = p2.x - p1.x, p2.y - p1.y
    mag = (dx**2 + dy**2) ** 0.5
    if mag == 0:
        return LineString([p1, p1])
    factor = length / mag
    new_point = Point(p1.x - dx * factor, p1.y - dy * factor)
    return LineString([p1, new_point])


def _get_first_intersection(geometry, ref_line):
    """Get first intersection.

    Parameters
    ----------
    geometry : Any
        Description.
    ref_line : Any
        Description.

    """
    if geometry.is_empty:
        return None
    if geometry.geom_type in ["Point", "LineString"]:
        if ref_line.distance(geometry) < 1e-8:
            return geometry
        return None
    if geometry.geom_type == "MultiPoint":
        return min(geometry.geoms, key=lambda g: ref_line.project(g))
    if geometry.geom_type == "GeometryCollection":
        points = [g for g in geometry.geoms if g.geom_type == "Point"]
        if points:
            return min(points, key=lambda g: ref_line.project(g))
    return None


def _nearest_point(*points, ref_point: Point):
    """Nearest point.

    Parameters
    ----------
    *points : tuple
        Description.
    ref_point : Point
        Description.

    """
    points = [p for p in points if p]
    return min(points, key=lambda p: ref_point.distance(p)) if points else None


def extend_lines(roads: gpd.GeoDataFrame, cluster_boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    """Extend lines.

    Parameters
    ----------
    roads : gpd.GeoDataFrame
        Description.
    cluster_boundaries : gpd.GeoDataFrame
        Description.

    Returns
    -------
    gpd.GeoDataFrame
        Description.

    """
    cluster_boundaries_pols = cluster_boundaries[
        cluster_boundaries.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ]

    if cluster_boundaries_pols.empty:
        return gpd.GeoDataFrame(geometry=[], crs=roads.crs)

    combined_cluster_poly = unary_union(cluster_boundaries_pols.geometry)
    combined_boundary = unary_union(cluster_boundaries.boundary)

    if isinstance(combined_cluster_poly, (MultiPolygon, GeometryCollection)):
        combined_cluster_poly_filt = [geom for geom in combined_cluster_poly.geoms if isinstance(geom, Polygon)]
    else:
        combined_cluster_poly_filt = [combined_cluster_poly] if isinstance(combined_cluster_poly, Polygon) else []

    combined_cluster_poly_filt = gpd.GeoDataFrame(geometry=combined_cluster_poly_filt, crs=roads.crs)

    roads_clipped = gpd.clip(roads, combined_cluster_poly_filt)

    road_geoms = list(roads_clipped.geometry)
    str_tree = STRtree(road_geoms)

    extend_len = 1000
    extended_lines = []

    for road in road_geoms:
        if isinstance(road, MultiLineString):
            clipped_roads = list(road.geoms)
        elif isinstance(road, LineString):
            clipped_roads = [road]
        else:
            continue

        for clipped_road in clipped_roads:
            coords = list(clipped_road.coords)
            if len(coords) < 2:
                continue

            start, next_start = Point(coords[0]), Point(coords[1])
            end, prev_end = Point(coords[-1]), Point(coords[-2])

            start_line = _extend_line(start, next_start, extend_len)
            end_line = _extend_line(end, prev_end, extend_len)

            start_b_inter = _get_first_intersection(start_line.intersection(combined_boundary), start_line)
            end_b_inter = _get_first_intersection(end_line.intersection(combined_boundary), end_line)

            nearby_start_indices = str_tree.query(start_line)
            nearby_start_roads = [road_geoms[i] for i in nearby_start_indices if not road_geoms[i].equals(clipped_road)]

            nearby_end_indices = str_tree.query(end_line)
            nearby_end_roads = [road_geoms[i] for i in nearby_end_indices if not road_geoms[i].equals(clipped_road)]

            start_r_inter = _get_first_intersection(
                unary_union([start_line.intersection(g) for g in nearby_start_roads]), start_line
            )
            end_r_inter = _get_first_intersection(
                unary_union([end_line.intersection(g) for g in nearby_end_roads]), end_line
            )

            final_start = _nearest_point(start_b_inter, start_r_inter, ref_point=start)
            final_end = _nearest_point(end_b_inter, end_r_inter, ref_point=end)

            if final_start:
                extended_lines.append(LineString([start, final_start]))
            if final_end:
                extended_lines.append(LineString([end, final_end]))

    return gpd.GeoDataFrame(geometry=extended_lines, crs=roads.crs)
