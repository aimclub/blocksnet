"""Testing blocks generator behavior"""

import os
import pytest
import geopandas as gpd
from blocksnet import BlocksGenerator, LandUse, LandUseProcessor

data_path = "./tests/data"

zones_to_lu = {"zone_1": LandUse.RECREATION, "zone_2": LandUse.RESIDENTIAL, "zone_3": LandUse.BUSINESS}


@pytest.fixture
def boundaries():
    gdf = gpd.read_parquet(os.path.join(data_path, "boundaries.parquet"))
    crs = gdf.estimate_utm_crs()
    return gdf.to_crs(crs)


@pytest.fixture
def local_crs(boundaries):
    return boundaries.crs


@pytest.fixture
def blocks_generator(boundaries, local_crs):
    gdfs = {
        name: gpd.read_parquet(os.path.join(data_path, f"{name}.parquet")).to_crs(local_crs)
        for name in ["roads", "railways", "water"]
    }
    return BlocksGenerator(boundaries, **gdfs)


@pytest.fixture
def blocks(blocks_generator):
    return blocks_generator.run()


@pytest.fixture
def land_use_processor(blocks):
    zones = blocks.copy()
    zones.geometry = zones.buffer(1000)
    zones["zone"] = zones.apply(lambda s: f"zone_{(s.name)%3+1}", axis=1)
    return LandUseProcessor(blocks, zones, zones_to_lu)


@pytest.fixture
def lu_blocks(land_use_processor):
    return land_use_processor.run()


def test_intersection(blocks):
    """Check if some of blocks intersect others"""
    sjoin = gpd.sjoin(blocks, blocks, predicate="intersects")
    sjoin["intersection"] = sjoin.apply(
        lambda s: s.geometry.intersection(blocks.loc[s.index_right, "geometry"]), axis=1
    )
    sjoin["intersects_other"] = sjoin.index != sjoin.index_right
    sjoin["intersection_poly"] = sjoin["intersection"].geom_type.isin(["Polygon", "MultiPolygon"])
    assert not any(sjoin["intersects_other"] & sjoin["intersection_poly"])


def test_within(blocks, boundaries):
    """Check if city blocks are inside initial city geometry with buffer"""
    assert blocks.within(boundaries.buffer(10).unary_union).all()


def test_sameness(blocks, lu_blocks):
    assert all(blocks.index == lu_blocks.index)
    assert all(blocks.geometry == lu_blocks.geometry)


def test_output(lu_blocks):
    lu_blocks.to_parquet(os.path.join(data_path, "_blocks.parquet"))
