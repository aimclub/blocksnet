"""Testing blocks cutter behavior"""

import os
import pytest
import geopandas as gpd
from blocksnet import BlocksGenerator

data_path = "./tests/data/blocks"
local_crs = 32637


@pytest.fixture
def territory_geometry():
    return gpd.read_file(os.path.join(data_path, "territory.geojson")).to_crs(local_crs)


@pytest.fixture
def blocks_cutter(territory_geometry):
    water_geometry = gpd.read_file(os.path.join(data_path, "water.geojson")).to_crs(local_crs)
    roads_geometry = gpd.read_file(os.path.join(data_path, "roads.geojson")).to_crs(local_crs)
    railways_geometry = gpd.read_file(os.path.join(data_path, "railways.geojson")).to_crs(local_crs)
    return BlocksGenerator(
        territory=territory_geometry, roads=roads_geometry, railways=railways_geometry, water=water_geometry
    )


@pytest.fixture
def blocks(blocks_cutter):
    return blocks_cutter.generate_blocks()


# def test_intersection(blocks):
#     """Check if some of blocks intersect others"""
#     blocks_intersection = gpd.sjoin(blocks, blocks, how="inner", predicate="intersects")
#     assert len(blocks_intersection.loc[lambda x: x.index != x["index_right"]]) == 0


def test_within(blocks, territory_geometry):
    """Check if city blocks are inside initial city geometry with buffer"""
    assert blocks.within(territory_geometry.geometry.unary_union.buffer(1)).all()
