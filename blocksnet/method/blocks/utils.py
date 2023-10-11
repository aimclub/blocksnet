import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from functools import reduce


class Utils:
    @staticmethod
    def _fill_spaces_in_blocks(block: gpd.GeoSeries) -> Polygon:
        """
        This geometry will be cut later from city's geometry.
        The water entities will split blocks from each other. The water geometries are taken using overpass turbo.
        The tags used in the query are: "natural"="water", "waterway"~"river|stream|tidal_channel|canal".
        This function is designed to be used with 'apply' function, applied to the pd.DataFrame.


        Parameters
        ----------
        row : GeoSeries


        Returns
        -------
        water_geometry : Union[Polygon, Multipolygon]
            Geometry of water. The water geometries are also buffered a little so the division of city's geometry
            could be more noticable.
        """

        new_block_geometry = None

        if len(block["rings"]) > 0:
            empty_part_to_fill = [Polygon(ring) for ring in block["rings"]]

            if len(empty_part_to_fill) > 0:
                new_block_geometry = reduce(
                    lambda geom1, geom2: geom1.union(geom2), [block["geometry"]] + empty_part_to_fill
                )

        if new_block_geometry:
            return new_block_geometry

        return block["geometry"]

    @staticmethod
    def _fix_blocks_geometries(city_geometry):
        """
        After cutting several entities from city's geometry, blocks might have unnecessary spaces inside them.
        In order to avoid this, this functions prepares data to fill empty spaces inside each city block

        Returns
        -------
        None
        """

        city_geometry = city_geometry.explode(ignore_index=True)
        city_geometry["rings"] = city_geometry.interiors
        city_geometry["geometry"] = city_geometry[["geometry", "rings"]].apply(
            Utils._fill_spaces_in_blocks, axis="columns"
        )

        return city_geometry

    @staticmethod
    def polygon_to_multipolygon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        This function makes one multipolygon from many polygons in the gdf.
        This step allows to fasten overlay operation between two multipolygons since overlay operates ineratively
        by passed geometry objects.

        Attributes
        ----------
        gdf: gpd.GeoDataFrame
            A gdf with many geometries-polygons

        Returns
        -------
        gdf: Multipolygon
            A gdf with one geometry-MultiPolygon
        """

        crs = gdf.crs
        gdf = gdf.unary_union
        if isinstance(gdf, Polygon):
            gdf = gpd.GeoDataFrame(geometry=[gdf], crs=crs)
        else:
            gdf = gpd.GeoDataFrame(geometry=[MultiPolygon(gdf)], crs=crs)
        return gdf
