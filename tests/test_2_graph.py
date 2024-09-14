"""Testing graph generator behavior"""

import os
import pytest
import geopandas as gpd
import osmnx as ox
from blocksnet.preprocessing.accessibility_processor import AccessibilityProcessor, IDUEDU_CRS

data_path = "./tests/data"


@pytest.fixture
def blocks():
    return gpd.read_parquet(os.path.join(data_path, "_blocks.parquet"))


@pytest.fixture
def accessibility_processor(blocks):
    return AccessibilityProcessor(blocks)


@pytest.fixture
def graph(accessibility_processor):
    return accessibility_processor.get_intermodal_graph()


@pytest.fixture
def accessibility_matrix(accessibility_processor, graph):
    return accessibility_processor.get_accessibility_matrix(graph)


def test_within(graph, accessibility_processor):
    """Check if graph nodes are within the initial geometry"""
    nodes, _ = ox.graph_to_gdfs(graph)
    assert all(nodes.to_crs(IDUEDU_CRS).within(accessibility_processor.polygon))


def test_index(blocks, accessibility_matrix):
    """Check if blocks index matches matrix index and columns"""
    assert (blocks.index == accessibility_matrix.index).all()
    assert (blocks.index == accessibility_matrix.columns).all()


def test_values(accessibility_matrix):
    """Check if adjacency matrix values are positive"""
    values = accessibility_matrix.values.ravel()
    assert all(v >= 0 for v in values)


def test_output(accessibility_matrix):
    accessibility_matrix.to_pickle(os.path.join(data_path, "_acc_mx.pickle"))
