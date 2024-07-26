"""Testing graph generator behavior"""

import os
import pytest
import geopandas as gpd
import osmnx as ox
from blocksnet import GraphGenerator, AdjacencyCalculator

data_path = "./tests/data"


@pytest.fixture
def blocks():
    return gpd.read_parquet(os.path.join(data_path, "_blocks.parquet"))


@pytest.fixture
def graph_generator(blocks):
    return GraphGenerator(blocks)


@pytest.fixture
def graph(graph_generator):
    return graph_generator.run("intermodal")


@pytest.fixture
def adjacency_calculator(blocks, graph):
    return AdjacencyCalculator(blocks, graph)


@pytest.fixture
def adjacency_matrix(adjacency_calculator):
    return adjacency_calculator.run()


def test_within(graph, graph_generator):
    """Check if graph nodes are within the initial geometry"""
    nodes, _ = ox.graph_to_gdfs(graph)
    assert all(nodes.within(graph_generator.territory.unary_union))


def test_index(blocks, adjacency_matrix):
    """Check if blocks index matches matrix index and columns"""
    assert (blocks.index == adjacency_matrix.index).all()
    assert (blocks.index == adjacency_matrix.columns).all()


def test_values(adjacency_matrix):
    """Check if adjacency matrix values are positive"""
    values = adjacency_matrix.values.ravel()
    assert all(v >= 0 for v in values)


def test_output(adjacency_matrix):
    adjacency_matrix.to_pickle(os.path.join(data_path, "_adj_mx.pickle"))
