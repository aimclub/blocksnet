"""Testing blocks cutter behavior"""

import os
import pytest
import geopandas as gpd
import osmnx as ox
from blocksnet.preprocessing import GraphGenerator, AdjacencyCalculator

data_path = "./tests/data/preprocessing"
local_crs = 32636


@pytest.fixture
def city_geometry():
    return gpd.read_parquet(os.path.join(data_path, "city_geometry.parquet")).to_crs(local_crs)


@pytest.fixture
def blocks():
    return gpd.read_parquet(os.path.join(data_path, "cutted_blocks.parquet")).to_crs(local_crs)


@pytest.fixture
def intermodal_graph(city_geometry):
    return GraphGenerator(city_geometry=city_geometry).get_graph("intermodal")


@pytest.fixture
def adjacency_matrix(blocks, intermodal_graph):
    return AdjacencyCalculator(blocks=blocks, graph=intermodal_graph).get_dataframe()


def test_within(intermodal_graph, city_geometry):
    """Check if graph nodes are within the initial geometry"""
    nodes, _ = ox.graph_to_gdfs(intermodal_graph)
    assert nodes.within(city_geometry.geometry.unary_union).all()


def test_index(blocks, adjacency_matrix):
    """Check if blocks index matches matrix index and columns"""
    assert (blocks.index == adjacency_matrix.index).all()
    assert (blocks.index == adjacency_matrix.columns).all()
