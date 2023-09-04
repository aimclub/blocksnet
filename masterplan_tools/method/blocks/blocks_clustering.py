from warnings import simplefilter

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points
from sklearn.cluster import DBSCAN
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from .utils import Utils

simplefilter("ignore", category=ConvergenceWarning)
simplefilter(action="ignore", category=FutureWarning)


class BlocksClusterization:
    def __init__(self, blocks, params):
        self.local_crs: int = blocks.crs.to_epsg()
        self.blocks: gpd.GeoDataFrame = blocks
        self.initial_blocks = blocks.copy()
        self.buildings_geom: gpd.GeoDataFrame = params.buildings.to_gdf()
        self.buildings_centroids: gpd.GeoDataFrame = None
        self.cutoff_ratio: float = 0.03
        self.blocks_to_consider: list = None

    def prepare_blocks(self):
        """
        This function prepares blocks to further analysis:
        - checks crs;
        - sets id columns;
        - filters only those blocks with buildings (by landuse type);
        - sets block area.
        """
        if "id" not in self.blocks.columns:
            self.blocks = self.blocks.reset_index(drop=False).rename(columns={"index": "id"})
        self.blocks = self.blocks[self.blocks["landuse"] == "buildings"]
        self.blocks.to_crs(self.local_crs, inplace=True)
        self.blocks["blocks_area"] = self.blocks.area

    def prepare_buildings(self):
        """
        This function prepares buildings to further usage:
        - sets crs;
        - creates separate gdf with only centroids of the buildings;
        - sets building area;
        - joins blocks with buildings in those blocks.
        """

        self.buildings_geom.to_crs(self.local_crs, inplace=True)
        self.buildings_centroids = gpd.GeoDataFrame(
            geometry=self.buildings_geom.representative_point(), crs=self.local_crs
        )
        self.buildings_centroids["building_area"] = self.buildings_geom.area

        if "geom" in self.buildings_geom.columns:
            self.buildings_geom.rename(columns={"geom": "geometry"}, inplace=True)
            self.buildings_geom.set_geometry("geometry", inplace=True)

        self.buildings_centroids = self.buildings_centroids.sjoin(self.blocks)

    def set_block_buildings_area(self):
        """
        This function calculates the ratio between blocks area and buildings area in those blocks
        """

        total_b_area = (
            self.buildings_centroids[["id", "building_area"]]
            .groupby("id")
            .sum()
            .reset_index()
            .rename(columns={"building_area": "total_building_area"})
        )

        self.blocks = pd.merge(self.blocks, total_b_area)
        self.blocks["area_ratio"] = self.blocks["total_building_area"] / self.blocks["blocks_area"]

        # TODO: check why this might happen
        self.blocks = self.blocks[self.blocks["area_ratio"] < 1]

    def plot_buildings_area_ratio(self, bins):
        """
        This function shows the distribution of buildings' area ratio in blocks
        """

        # FIXME seaborn is not in the project dependencies. And it is a correct usage without returning a canvas, etc.?
        sns.histplot(self.blocks["area_ratio"], bins=bins)

    def select_blocks_to_cluster(self):
        """
        This function select only those blocks where area ratio is less than specified cutoff ratio
        """

        self.blocks_to_consider = list(self.blocks[self.blocks["area_ratio"] < self.cutoff_ratio]["id"])

    @staticmethod
    def get_optimal_params(X):
        """Find optimal parameters for DBSCAN using silhouette score"""
        max_silhouette = -1
        optimal_eps, optimal_min_samples = None, None

        # Try different eps and min_samples values
        for eps in np.arange(10, 200, 10):
            for min_samples in range(2, 10):
                # Fit DBSCAN
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db.fit_predict(X)

                # Calculate silhouette score
                try:
                    silhouette = silhouette_score(X, labels)
                except Exception:  # pylint: disable=broad-except
                    silhouette = -1

                if silhouette > max_silhouette:
                    max_silhouette = silhouette
                    optimal_eps, optimal_min_samples = eps, min_samples

        return optimal_eps, optimal_min_samples

    def get_clusters(self, t_build):
        """
        This function sets cluster numbers for each buildings in the block
        """

        houses = t_build.copy()
        # get clusters
        if houses.shape[0] == 2:
            idxs = list(houses.index)

            houses.loc[idxs[0], "cluster"] = 0
            houses.loc[idxs[1], "cluster"] = 1

        elif houses.shape[0] == 1:
            houses["cluster"] = 0

        else:
            # Find optimal eps and min_samples values
            X = houses[["Latitude", "Longitude"]]

            eps, min_samples = self.get_optimal_params(X)

            # Run DBSCAN with optimal parameters
            try:
                db = DBSCAN(eps=eps, min_samples=min_samples)
                houses["cluster"] = db.fit_predict(X)
            except ValueError as ex:
                print("Exception ", ex)
                houses["cluster"] = 0

        return houses

    def get_poly_from_multipoly(self, polygon, multi_poly):
        # Get the index of the polygon that intersects with the MultiPolygon
        idx_intersects = multi_poly.intersects(polygon.geometry.iloc[0])

        # Keep only the polygon(s) that intersects with the Polygon
        polygon_to_keep = multi_poly[idx_intersects]

        return polygon_to_keep

    def get_connection_lines(self, my_inner_poly, outer_poly, inner_polys):
        if not isinstance(outer_poly, gpd.GeoDataFrame):
            outer_poly = gpd.GeoDataFrame(geometry=[outer_poly], crs=self.local_crs)

        for poly in inner_polys:
            if str(poly["geometry"]) != str(my_inner_poly["geometry"]):
                outer_poly = outer_poly.overlay(poly, how="difference")

        if isinstance(outer_poly["geometry"].item(), MultiPolygon):
            outer_poly = self.get_poly_from_multipoly(my_inner_poly, outer_poly)

        # bigger_poly_boundary = LineString(outer_poly['geometry'].item().exterior.coords)
        bigger_poly_boundary = outer_poly["geometry"].item().boundary

        # Create the inner and outer polygons from their borders
        inner_poly_boundary = my_inner_poly.boundary.item()
        inner_points = list(inner_poly_boundary.coords)
        touchng_points = []

        for idx in [0, 1]:
            for func in ["min", "max"]:
                if func == "min":
                    min_max_xy_point = Point(
                        min(inner_points, key=lambda item: item[idx])  # pylint: disable=cell-var-from-loop
                    )
                else:
                    min_max_xy_point = Point(
                        max(inner_points, key=lambda item: item[idx])  # pylint: disable=cell-var-from-loop
                    )

                points_on_outer_geom = nearest_points(bigger_poly_boundary, min_max_xy_point)
                points_on_outer_geom = points_on_outer_geom[0]

                if not points_on_outer_geom.within(my_inner_poly.geometry.item()):
                    touchng_points.append(points_on_outer_geom)
                    line = LineString([min_max_xy_point, points_on_outer_geom])
                    inner_poly_boundary = inner_poly_boundary.union(line)

        return inner_poly_boundary

    def get_inner_poly(self, houses_cluster, cluster):
        points = self.buildings_geom.sjoin(
            houses_cluster[houses_cluster["cluster"] == cluster].loc[:, ["cluster", "geometry"]], how="inner"
        )
        rectangle = MultiPoint(points[["geometry"]].unary_union.convex_hull.boundary.coords).minimum_rotated_rectangle
        inner_poly = gpd.GeoDataFrame(geometry=[rectangle], crs=self.local_crs)

        return inner_poly

    def get_inner_geom(self, my_inner_poly, new_inner_polygons_list, i, inner_polys, if_break=False):
        i = i.geometry.item()

        if my_inner_poly.representative_point().item().intersects(i):
            inner_poly_boundary2 = self.get_connection_lines(my_inner_poly, i, inner_polys)

            inner_poly_boundary2 = inner_poly_boundary2.convex_hull
            res = i.difference(inner_poly_boundary2)
            new_inner_polygons_list.append(
                gpd.GeoDataFrame(geometry=[i.intersection(inner_poly_boundary2)], crs=self.local_crs)
            )

            res = gpd.GeoDataFrame(geometry=[res], crs=self.local_crs)

            if isinstance(res["geometry"].item(), MultiPolygon):
                res = res.explode().reset_index(drop=True)

            if res.shape[0] > 1:
                for i in res.index:
                    new_inner_polygons_list.append(gpd.GeoDataFrame(geometry=res.loc[i, :], crs=self.local_crs))
            else:
                new_inner_polygons_list.append(res)
            if_break = True  # FIXME is it right?

            return if_break
        return False

    def clusterize_blocks(self):
        old_new_blocks = []
        new_polys = []
        bigger_polytmp = []

        for c, block_id in enumerate(tqdm(self.blocks_to_consider)):
            inner_polys = []
            new_inner_polygons_list = []

            t_build = self.buildings_centroids[self.buildings_centroids["id"] == block_id].copy()
            t_build["Latitude"] = t_build.geometry.x
            t_build["Longitude"] = t_build.geometry.y

            houses_cluster = self.get_clusters(t_build)
            bigger_poly = self.blocks[self.blocks["id"] == block_id]
            houses_cluster = houses_cluster[houses_cluster["cluster"] != -1]

            if len(set(houses_cluster["cluster"])) == 1 and houses_cluster["cluster"].shape[0] > 1:
                new_polys.append(houses_cluster)
                continue

            for _ in set(houses_cluster["cluster"]):
                # Create spatial index
                gdf_sindex = houses_cluster.sindex
                line = bigger_poly.boundary.item()

                # Get nearest geometry
                nearest_geom_index = list(gdf_sindex.nearest(line, 1))[0]
                cluster = houses_cluster.iloc[nearest_geom_index]["cluster"].item()

                try:
                    inner_polygon = self.get_inner_poly(houses_cluster, cluster)
                    inner_polys.append(inner_polygon)
                except AttributeError:
                    print("AttributeError happened")
                    continue

                houses_cluster = houses_cluster[houses_cluster["cluster"] != cluster]

            for inner_polygon in inner_polys:
                flag = 0
                if new_inner_polygons_list:
                    for c, polygon_geom in enumerate(new_inner_polygons_list):
                        if isinstance(polygon_geom.geometry.item(), MultiPolygon):
                            polygon_geom = polygon_geom.geometry.item()
                            for j in polygon_geom.geoms:
                                j = gpd.GeoDataFrame(geometry=[j], crs=self.local_crs)
                                if_break = self.get_inner_geom(
                                    inner_polygon, new_inner_polygons_list, j, inner_polys=inner_polys
                                )
                                if if_break:
                                    flag = 1
                                    del new_inner_polygons_list[c]
                                    break
                        else:
                            if_break = self.get_inner_geom(
                                inner_polygon, new_inner_polygons_list, polygon_geom, inner_polys=inner_polys
                            )
                            if if_break:
                                flag = 1
                                del new_inner_polygons_list[c]
                                break

                if not flag:
                    inner_poly_boundary2 = self.get_connection_lines(
                        inner_polygon, bigger_poly, inner_polys=inner_polys
                    )
                    inner_poly_boundary2 = inner_poly_boundary2.convex_hull

                    new_poly = bigger_poly.intersection(inner_poly_boundary2)
                    new_poly = gpd.GeoDataFrame(geometry=new_poly, crs=self.local_crs)
                    new_inner_polygons_list.append(new_poly)

                    bigger_poly = bigger_poly.difference(inner_poly_boundary2)

                    bigger_poly = gpd.GeoDataFrame(geometry=bigger_poly, crs=self.local_crs)
                    bigger_polytmp.append(bigger_poly)

                    if isinstance(bigger_poly["geometry"].item(), MultiPolygon):
                        bigger_poly = bigger_poly.explode(index_parts=True).reset_index(drop=True)
                        bigger_poly["area"] = bigger_poly.area
                        bigger_poly.sort_values(by="area", ascending=False, inplace=True)

                        new_inner_polygons_list.append(
                            gpd.GeoDataFrame(geometry=[bigger_poly.iloc[1:, :].unary_union], crs=self.local_crs)
                        )
                        bigger_poly = bigger_poly.iloc[[0]]

                        # FIXME: asserts can be disabled with optimization. Replace with a correct exception
                        assert isinstance(bigger_poly["geometry"].item(), Polygon)

            new_polys += new_inner_polygons_list

            old_new_blocks.append(bigger_poly)

        for c, polygon_geom in enumerate(old_new_blocks + new_polys):
            if c == 0:
                new_poly = polygon_geom
            else:
                new_poly = pd.concat([new_poly, polygon_geom])

        new_poly = new_poly[new_poly.geometry.is_empty == False]
        new_poly["landuse"] = "no_dev_area"

        self.blocks = pd.concat(
            [
                self.initial_blocks[~self.initial_blocks.index.isin(self.blocks_to_consider)],
                new_poly,
            ]
        )

        self.blocks = self.blocks[["geometry", "landuse"]].reset_index(drop=True)

    def update_blocks(self):
        # Perform the spatial join operation
        join: gpd.GeoDataFrame = gpd.sjoin(
            self.blocks,
            self.buildings_centroids[["geometry"]],
            how="inner",
            predicate="contains",
        )

        join = (
            join.reset_index(drop=False)
            .drop_duplicates(subset="index")
            .set_index("index")
            .drop(columns=["index_right"])
        )

        # Update the landuse column for the resulting GeoDataFrame
        join.loc[join["landuse"] == "no_dev_area", "landuse"] = "buildings"

        # Note: You can also update the original gdf DataFrame with the updated values:
        self.blocks.update(join[["landuse"]])

        self.blocks.loc[
            (self.blocks["landuse"] == "no_dev_area") | (self.blocks["landuse"] == "buildings"),
            "geometry",
        ] = self.blocks.loc[
            (self.blocks["landuse"] == "no_dev_area") | (self.blocks["landuse"] == "buildings"),
            "geometry",
        ].buffer(
            -1
        )

        self.blocks = self.blocks[~self.blocks.is_empty]

    def fix_blocks(self):
        self.blocks = Utils._fix_blocks_geometries(self.blocks)

        self.blocks = self.blocks[["geometry", "landuse"]]
        gdf_no_overlay = self.blocks.copy()

        # loop over each polygon in the GeoDataFrame
        for idx, row in tqdm(self.blocks.iterrows()):
            # check if the polygon is fully contained within another polygon
            mask = self.blocks.geometry.contains(row.geometry).values
            if mask.sum() > 1:
                # remove the polygon from the new GeoDataFrame
                gdf_no_overlay.drop(idx, inplace=True)
            else:
                # do nothing
                pass

        self.blocks = gdf_no_overlay

    def run(self):
        self.prepare_blocks()
        self.prepare_buildings()
        self.set_block_buildings_area()
        self.select_blocks_to_cluster()
        self.clusterize_blocks()
        self.update_blocks()
        self.fix_blocks()

        return self.blocks
