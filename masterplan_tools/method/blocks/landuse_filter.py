import geopandas as gpd
import osmnx as ox  # pylint: disable=import-error
import pandas as pd
from .utils import Utils

class LuFilter:
    def __init__(self, city_blocks: gpd.GeoDataFrame, landuse_geometries=None):
        self.landuse_geometries = landuse_geometries
        self.city_blocks = city_blocks.copy()
        self.local_crs = city_blocks.crs.to_epsg()

    def _receiving_landuse(self) -> gpd.GeoDataFrame:
        landuse = ox.geometries_from_polygon(
            self.city_blocks.to_crs(4326).geometry.unary_union,
            tags={"landuse": True, "leisure": True, "building": True, "natural": "wood"},
        )

        selected_columns = ["landuse", "leisure", "geometry"]
        landuse_selected = landuse[selected_columns].copy()

        del landuse

        landuse_selected["landuse"] = landuse_selected["landuse"].combine_first(landuse_selected["leisure"])
        landuse_selected.drop("leisure", axis=1, inplace=True)
        landuse_selected = landuse_selected[landuse_selected["landuse"] != "grass"]
        landuse_selected["landuse"].fillna("building", inplace=True)
        landuse_selected.reset_index(drop=True, inplace=True)
        landuse_selected.to_crs(self.local_crs, inplace=True)

        landuse_selected = landuse_selected[
            landuse_selected.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])
        ]
        landuse_selected.reset_index(drop=True, inplace=True)
        return landuse_selected

    def _pruning_landuse(self, landuse_selected: gpd.GeoDataFrame, no_dev_area: bool = False) -> None:

        if "landuse" in landuse_selected.columns:
            landuse_tags = landuse_selected.loc[
                landuse_selected["landuse"].isin(["cemetery", "industrial", "park", "building", "allotments"])
            ].copy()

            to_drop = gpd.GeoDataFrame(
                geometry=[landuse_tags.loc[~landuse_tags["landuse"].isin(["building"])].unary_union],
                crs=self.city_blocks.crs.to_epsg(),
            )
            
            # display(self.city_blocks)
            # display(to_drop)
            territory_without_landuse = gpd.overlay(self.city_blocks, to_drop, how="difference")
            territory_without_landuse = territory_without_landuse.explode(ignore_index=True).reset_index(drop=True)

            territory_with_landuse = gpd.overlay(self.city_blocks, to_drop, how="intersection")
            territory_with_landuse["landuse"] = "important"
            # territory_with_landuse.geometry = territory_with_landuse.buffer(-5)

            to_drop = landuse_tags.loc[landuse_tags["landuse"].isin(["building"])].copy()
            to_drop.geometry = to_drop.representative_point()

            # Perform the spatial join
            result = gpd.sjoin(territory_without_landuse, to_drop, predicate="contains")

            # Select only those polygons in gdf1 which contain polygons from gdf2
            selected = result[result.index_right.notnull()].copy()
            territory_without_landuse.loc[selected.index, "landuse"] = "buildings"

        else:
            # display(self.city_blocks)
            # display(landuse_selected)

            territory_without_landuse = gpd.overlay(
                self.city_blocks, landuse_selected, how="difference", keep_geom_type=True
            )
            territory_with_landuse = gpd.overlay(
                landuse_selected, self.city_blocks, how="intersection", keep_geom_type=True
            )

        if no_dev_area:
            territory_with_landuse["landuse"] = "no_dev_area"
        else:
            territory_with_landuse["landuse"] = "selected_area"

        gdf = pd.concat([territory_without_landuse, territory_with_landuse])
        gdf = gdf[["landuse", "geometry"]]
        gdf = gdf[~gdf.is_empty]
        gdf = gdf.explode(ignore_index=True).reset_index(drop=True)

        self.city_blocks = gdf.copy()

    def filter_geom_types(gdf):
        return gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

    def filter_lu(self, osm_landuse=None) -> gpd.GeoDataFrame:

        if not isinstance(osm_landuse, gpd.GeoDataFrame):
            osm_landuse = self._receiving_landuse()

        self.city_blocks = Utils.polygon_to_multipolygon(self.city_blocks)

        osm_landuse = LuFilter.filter_geom_types(osm_landuse)

        no_dev = LuFilter.filter_geom_types(self.landuse_geometries.no_development.to_gdf())
        no_dev = Utils.polygon_to_multipolygon(no_dev)
        
        lu = LuFilter.filter_geom_types(self.landuse_geometries.landuse.to_gdf())
        lu = Utils.polygon_to_multipolygon(lu)

        self._pruning_landuse(osm_landuse)

        self._pruning_landuse(no_dev, no_dev_area=True)

        self._pruning_landuse(lu)
        
        lu_b = osm_landuse[osm_landuse['landuse']=='building']
        lu_b = gpd.GeoDataFrame(geometry=lu_b.loc[:, 'geometry'], crs=osm_landuse.crs.to_epsg())
        self.city_blocks = gpd.sjoin(self.city_blocks, lu_b, predicate="contains")

        self.city_blocks["landuse"].fillna("no_dev_area", inplace=True)

        return self.city_blocks
