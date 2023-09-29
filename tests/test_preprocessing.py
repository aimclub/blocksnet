"""Testing blocks cutter behavior"""

import os
import pytest
import geopandas as gpd
from blocksnet.preprocessing import DataGetter, AggregateParameters

data_path = "./tests/data/preprocessing"
local_crs = 32636


@pytest.fixture
def cutted_blocks():
    return gpd.read_parquet(os.path.join(data_path, "cutted_blocks.parquet")).to_crs(local_crs)


@pytest.fixture
def getter(cutted_blocks):
    return DataGetter(blocks=cutted_blocks)


@pytest.fixture
def aggr_params():
    buildings = gpd.read_parquet(os.path.join(data_path, "buildings.parquet")).to_crs(local_crs)
    greenings = gpd.read_parquet(os.path.join(data_path, "greenings.parquet")).to_crs(local_crs)
    parkings = gpd.read_parquet(os.path.join(data_path, "parkings.parquet")).to_crs(local_crs)

    return AggregateParameters(buildings=buildings, greenings=greenings, parkings=parkings)


@pytest.fixture
def aggr_blocks(getter, aggr_params):
    return getter.aggregate_blocks_info(params=aggr_params)


def test_count(cutted_blocks, aggr_blocks):
    """Check if we receive as much blocks as we give"""
    assert len(cutted_blocks) == len(aggr_blocks.to_gdf())


def test_ids(aggr_blocks):
    """Check if IDs are unique and equal to index"""
    gdf = aggr_blocks.to_gdf()
    assert len(gdf["block_id"].unique()) == len(gdf)
    assert (gdf.index == gdf["block_id"]).all()


# def test_area(aggr_blocks):
#   gdf = aggr_blocks.to_gdf()
#   assert (gdf['area'] >= (gdf['current_green_area'] + gdf['current_industrial_area'] + gdf['current_living_area'])).all()
