"""Testing blocks cutter behavior"""

import os
import pytest
import geopandas as gpd
import osmnx as ox
from blocksnet.preprocessing import GraphGenerator, AdjacencyCalculator

data_path = "./tests/data/preprocessing"
local_crs = 32636


@pytest.fixture
def blocks():
    return gpd.read_parquet(os.path.join(data_path, "cutted_blocks.parquet")).to_crs(local_crs)


@pytest.fixture
def graph_generator(blocks):
    return GraphGenerator(territory=blocks)


@pytest.fixture
def intermodal_graph(graph_generator):
    return graph_generator.get_graph("intermodal")


@pytest.fixture
def adjacency_matrix(blocks, intermodal_graph):
    return AdjacencyCalculator(blocks=blocks, graph=intermodal_graph).get_dataframe()


def test_within(intermodal_graph, graph_generator):
    """Check if graph nodes are within the initial geometry"""
    nodes, _ = ox.graph_to_gdfs(intermodal_graph)
    assert nodes.within(graph_generator.territory.geometry.unary_union).all()


def test_index(blocks, adjacency_matrix):
    """Check if blocks index matches matrix index and columns"""
    assert (blocks.index == adjacency_matrix.index).all()
    assert (blocks.index == adjacency_matrix.columns).all()
