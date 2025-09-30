import shapely


def calculate_outer_radius(polygon: shapely.Polygon) -> float:
    """Calculate outer radius.

    Parameters
    ----------
    polygon : shapely.Polygon
        Description.

    Returns
    -------
    float
        Description.

    """
    center = polygon.centroid
    corners = [shapely.Point(coord) for coord in polygon.exterior.coords]
    return max(center.distance(corner) for corner in corners)


def calculate_inner_radius(polygon: shapely.Polygon) -> float:
    """Calculate inner radius.

    Parameters
    ----------
    polygon : shapely.Polygon
        Description.

    Returns
    -------
    float
        Description.

    """
    center = polygon.representative_point()
    corners = [shapely.Point(coord) for coord in polygon.exterior.coords]
    side_centers = [shapely.MultiPoint([corners[i], corners[i + 1]]).centroid for i in range(len(corners) - 1)]
    return min(center.distance(point) for point in side_centers)


def calculate_aspect_ratio(polygon: shapely.Polygon) -> float:
    """Calculate aspect ratio.

    Parameters
    ----------
    polygon : shapely.Polygon
        Description.

    Returns
    -------
    float
        Description.

    """
    rectangle = polygon.minimum_rotated_rectangle
    rectangle_coords = list(rectangle.exterior.coords)
    side_lengths = [
        (
            (rectangle_coords[i][0] - rectangle_coords[i - 1][0]) ** 2
            + (rectangle_coords[i][1] - rectangle_coords[i - 1][1]) ** 2
        )
        ** 0.5
        for i in range(1, 5)
    ]
    length_1, length_2 = side_lengths[0], side_lengths[1]
    aspect_ratio = max(length_1, length_2) / min(length_1, length_2)
    return aspect_ratio
