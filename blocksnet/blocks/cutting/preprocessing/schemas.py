import shapely
from ....utils.validation import GdfSchema


class RoadsSchema(GdfSchema):
    @classmethod
    """RoadsSchema class.

    """
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.LineString, shapely.MultiLineString}


class RailwaysSchema(GdfSchema):
    @classmethod
    """RailwaysSchema class.

    """
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.LineString, shapely.MultiLineString}


class WaterSchema(GdfSchema):
    @classmethod
    """WaterSchema class.

    """
    def _geometry_types(cls):
        """Geometry types.

        """
        return {shapely.LineString, shapely.MultiLineString, shapely.Polygon, shapely.MultiPolygon}
