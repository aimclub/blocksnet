class DataValidator:
    def __init__(self):
        pass

    def validate_geodataframe(self, gdf, expected_geom_type=None):
        """
        Validates a GeoDataFrame.
        Checks if it's not empty, and if the geometry type matches the expected type, if provided.
        """
        if gdf is None or gdf.empty:
            raise ValueError("GeoDataFrame is empty or None")

        if expected_geom_type and not all(gdf.geom_type == expected_geom_type):
            raise ValueError(f"GeoDataFrame contains unexpected geometry types, expected: {expected_geom_type}")

    def validate(self, territory, roads, railways, water):
        """
        Validates all input GeoDataFrames.
        """
        self.validate_geodataframe(territory, expected_geom_type="Polygon")
        if roads is not None:
            self.validate_geodataframe(roads, expected_geom_type="LineString")
        if railways is not None:
            self.validate_geodataframe(railways, expected_geom_type="LineString")
        if water is not None:
            self.validate_geodataframe(water, expected_geom_type=["Polygon", "LineString"])
