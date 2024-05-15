import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from longsgis import voronoiDiagram4plg
from tqdm import tqdm
from pydantic import BaseModel, model_validator, field_validator
from ..models.geodataframe import GeoDataFrame, BaseRow


class BlockRow(BaseRow):
    geometry: shapely.Polygon


class BuildingRow(BaseRow):
    geometry: shapely.Polygon | shapely.MultiPolygon

    # @field_validator('geometry', mode='before')
    # @staticmethod
    # def validate_geometry(geometry):
    #   return geometry.representative_point()


class BlocksSplitter(BaseModel):
    blocks: GeoDataFrame[BlockRow]
    buildings: GeoDataFrame[BuildingRow]

    @field_validator("blocks", mode="before")
    @staticmethod
    def validate_blocks(blocks):
        if not isinstance(blocks, GeoDataFrame[BlockRow]):
            blocks = GeoDataFrame[BlockRow](blocks)
        return blocks

    @field_validator("buildings", mode="before")
    @staticmethod
    def validate_buildings(buildings):
        if not isinstance(buildings, GeoDataFrame[BuildingRow]):
            buildings = GeoDataFrame[BuildingRow](buildings)
        return buildings

    @model_validator(mode="after")
    @staticmethod
    def validate_model(self):
        blocks = self.blocks
        buildings = self.buildings
        assert blocks.crs == buildings.crs, "Blocks CRS must match buildings CRS"
        return self

    @staticmethod
    def _drop_index_columns(gdf):
        if "index_left" in gdf.columns:
            gdf.drop(columns=["index_left"], inplace=True)
        if "index_right" in gdf.columns:
            gdf.drop(columns=["index_right"], inplace=True)

    @staticmethod
    def _split_block(block: shapely.Polygon, buildings: gpd.GeoDataFrame, n_clusters):
        vd = voronoiDiagram4plg(buildings, block)
        vd = vd.explode(index_parts=True)

        X = vd["geometry"].apply(lambda geom: geom.bounds).tolist()
        X_flat = [coord for rect in X for coord in rect]
        X = np.array(X_flat).reshape(-1, 4)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        vd["cluster"] = kmeans.fit_predict(X_scaled)
        vd = vd.dissolve(by="cluster")
        vd = vd.explode(index_parts=False)

        return vd.reset_index(drop=True)

    def split_blocks(self, n_clusters: int = 4, points_quantile: float = 0.98, area_quantile: float = 0.95):

        get_geom_coords = lambda geom: len(geom.exterior.coords)

        blocks = self.blocks.copy()
        quantile_num_points = blocks.geometry.apply(get_geom_coords).quantile(points_quantile)
        quantile_area = blocks.area.quantile(area_quantile)
        filtered_blocks = self.blocks[
            (blocks.geometry.apply(get_geom_coords) > quantile_num_points) & (blocks.area > quantile_area)
        ]

        sjoin = self.buildings.sjoin(filtered_blocks)
        sjoin = sjoin.rename(columns={"index_right": "block_id"})

        gdfs = []

        for block_id, buildings_gdf in sjoin.groupby("block_id"):
            try:
                block_geometry = blocks.loc[block_id, "geometry"]
                # buildings_gdf['geometry'] = buildings_gdf['geometry'].apply(lambda g : shapely.intersection(g, block_geometry))
                gdfs.append(self._split_block(block_geometry, buildings_gdf, n_clusters))
            except:
                gdfs.append(filtered_blocks[filtered_blocks.index == block_id])

        new_blocks = pd.concat(gdfs)
        old_blocks = blocks[~blocks.index.isin(filtered_blocks.index)]

        # return new_blocks
        return pd.concat([new_blocks, old_blocks]).reset_index()[["geometry"]]
