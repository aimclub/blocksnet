"""Testing city model behavior"""

import os
import pytest
import geopandas as gpd
import pandas as pd
from blocksnet import City

data_path = "./tests/data"


@pytest.fixture
def blocks():
    return gpd.read_parquet(os.path.join(data_path, "_blocks.parquet"))


@pytest.fixture
def accessibility_matrix():
    return pd.read_pickle(os.path.join(data_path, "_acc_mx.pickle"))


@pytest.fixture
def local_crs(blocks):
    return blocks.crs


@pytest.fixture
def buildings(local_crs):
    return gpd.read_parquet(os.path.join(data_path, "buildings.parquet")).to_crs(local_crs)


@pytest.fixture
def city(blocks, accessibility_matrix, buildings):
    city = City(blocks, accessibility_matrix)
    city.update_buildings(buildings)
    return city


def test_coherence(city, blocks, accessibility_matrix):
    blocks_gdf = city.get_blocks_gdf(True)
    assert all(blocks.index == blocks_gdf.index)
    assert all(accessibility_matrix.index == city.accessibility_matrix.index)
    assert all(accessibility_matrix.columns == city.accessibility_matrix.columns)
    assert all(blocks_gdf.index == city.accessibility_matrix.index)
    assert all(city.accessibility_matrix.index == city.accessibility_matrix.columns)


def test_output(city):
    city.to_pickle(os.path.join(data_path, "_city.pickle"))
