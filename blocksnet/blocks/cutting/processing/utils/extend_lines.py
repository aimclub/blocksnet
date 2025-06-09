import geopandas as gpd
from shapely.geometry import (
    Point,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon
)
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

def extend_line(start_point : Point, direct_point : Point, extend_len : float):
    dx =  - direct_point.x + start_point.x
    dy =  - direct_point.y + start_point.y

    extended_point = Point(start_point.x + dx*extend_len, 
                              start_point.y + dy*extend_len)
    
    extension_line = LineString([start_point, extended_point])

    return extension_line

def get_first_intersection(geom, reference_line):
    if geom.is_empty:
        # print('none')
        return None
        
    if geom.geom_type == "Point":
        return geom
    elif geom.geom_type == "MultiPoint":
        start_point = Point(reference_line.coords[0])
        return min(geom.geoms, key=lambda p: p.distance(start_point))
    elif geom.geom_type == "GeometryCollection":
        for item in geom.geoms:
            if item.geom_type == "Point":
                return item
            elif item.geom_type == "MultiPoint":
                start_point = Point(reference_line.coords[0])
                return min(item.geoms, key=lambda p: p.distance(start_point))
    return None

from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import geopandas as gpd
from tqdm import tqdm


def extend_line(p1: Point, p2: Point, length: float) -> LineString:
    dx, dy = p2.x - p1.x, p2.y - p1.y
    mag = (dx**2 + dy**2)**0.5
    if mag == 0:
        return LineString([p1, p1])
    factor = length / mag
    new_point = Point(p1.x - dx * factor, p1.y - dy * factor)
    return LineString([p1, new_point])


def get_first_intersection(geometry, ref_line):
    if geometry.is_empty:
        return None
    if geometry.geom_type in ["Point", "LineString"]:
        # Проверяем, что пересечение лежит на линии расширения
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


def nearest_point(*points, ref_point: Point):
    points = [p for p in points if p]
    return min(points, key=lambda p: ref_point.distance(p)) if points else None


def extend_roads_to_boundary(
    roads: gpd.GeoDataFrame,
    boundary_geom: Polygon | MultiPolygon | LineString | MultiLineString,
    cluster_boundaries: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:

    # Определение глобальной границы
    if isinstance(boundary_geom, (Polygon, MultiPolygon)):
        global_boundary = boundary_geom.boundary
    elif isinstance(boundary_geom, (LineString, MultiLineString)):
        global_boundary = boundary_geom
    else:
        raise TypeError("boundary_geom должен быть Polygon/MultiPolygon или LineString/MultiLineString")

    # Объединяем кластеры
    combined_cluster_poly = unary_union(cluster_boundaries.geometry)
    combined_boundary = unary_union(cluster_boundaries.boundary)

    # Обрезаем дороги по границам кластера
    roads_clipped = gpd.clip(roads, combined_cluster_poly)

    # Пространственный индекс для ускорения поиска пересечений
    road_geoms = list(roads_clipped.geometry)
    str_tree = STRtree(road_geoms)

    extend_len = 1000
    extended_lines = []

    for road in tqdm(road_geoms):
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

            start_line = extend_line(start, next_start, extend_len)
            end_line = extend_line(end, prev_end, extend_len)

            start_b_inter = get_first_intersection(start_line.intersection(combined_boundary), start_line)
            end_b_inter = get_first_intersection(end_line.intersection(combined_boundary), end_line)

            # Получаем индексы ближайших дорог из индекса
            nearby_start_indices = str_tree.query(start_line)
            nearby_start_roads = [road_geoms[i] for i in nearby_start_indices if not road_geoms[i].equals(clipped_road)]

            nearby_end_indices = str_tree.query(end_line)
            nearby_end_roads = [road_geoms[i] for i in nearby_end_indices if not road_geoms[i].equals(clipped_road)]

            start_r_inter = get_first_intersection(
                unary_union([start_line.intersection(g) for g in nearby_start_roads]), start_line
            )
            end_r_inter = get_first_intersection(
                unary_union([end_line.intersection(g) for g in nearby_end_roads]), end_line
            )

            final_start = nearest_point(start_b_inter, start_r_inter, ref_point=start)
            final_end = nearest_point(end_b_inter, end_r_inter, ref_point=end)

            if final_start:
                extended_lines.append(LineString([start, final_start]))
            if final_end:
                extended_lines.append(LineString([end, final_end]))

    return gpd.GeoDataFrame(geometry=extended_lines, crs=roads.crs)
