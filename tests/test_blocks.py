"""Testing blocks cutter behavior"""

import os
import pytest
import geopandas as gpd
from blocksnet.method.blocks import BlocksCutter, CutParameters, LandUseParameters

data_path = "./tests/data/blocks"
local_crs = 32636


@pytest.fixture
def city_geometry():
    return gpd.read_parquet(os.path.join(data_path, "city_geometry.parquet")).to_crs(local_crs)


@pytest.fixture
def cut_params(city_geometry):
    water_geometry = gpd.read_parquet(os.path.join(data_path, "water_geometry.parquet")).to_crs(local_crs)
    roads_geometry = gpd.read_parquet(os.path.join(data_path, "roads_geometry.parquet")).to_crs(local_crs)
    railways_geometry = gpd.read_parquet(os.path.join(data_path, "railways_geometry.parquet")).to_crs(local_crs)
    return CutParameters(city=city_geometry, water=water_geometry, roads=roads_geometry, railways=railways_geometry)


@pytest.fixture
def lu_params():
    no_development = gpd.read_parquet(os.path.join(data_path, "no_development.parquet")).to_crs(local_crs)
    landuse = gpd.read_parquet(os.path.join(data_path, "landuse.parquet")).to_crs(local_crs)
    buildings = gpd.read_parquet(os.path.join(data_path, "buildings_geom.parquet")).to_crs(local_crs)
    return LandUseParameters(no_development=no_development, landuse=landuse, buildings=buildings)


@pytest.fixture
def blocks(cut_params, lu_params):
    return BlocksCutter(cut_parameters=cut_params, lu_parameters=lu_params).get_blocks()


def test_intersection(blocks):
    """Check if some of blocks intersect others"""
    assert (
        len(gpd.sjoin(blocks, blocks, how="inner", predicate="intersects").loc[lambda x: x["id_left"] != x["id_right"]])
        == 0
    )


def test_within(blocks, city_geometry):
    """Check if city blocks are inside initial city geometry with buffer"""
    assert blocks["geometry"].apply(lambda geom: city_geometry["geometry"].buffer(10).contains(geom))[0].all()
