"""Testing methods behavior"""

import os
import pytest
import geopandas as gpd
import pandas as pd
from blocksnet import City
from blocksnet.method.accessibility import Accessibility, ACCESSIBILITY_FROM_COLUMN, ACCESSIBILITY_TO_COLUMN

data_path = "./tests/data"


@pytest.fixture
def city():
    return City.from_pickle(os.path.join(data_path, "_city.pickle"))


@pytest.fixture
def crs(city):
    return city.crs


@pytest.fixture
def accessibility(city):
    return Accessibility(city_model=city)


@pytest.fixture
def block(city):
    return city[123]


@pytest.fixture
def service_type(city):
    return city["school"]


@pytest.fixture
def accessibility_result(accessibility, block):
    return accessibility.calculate(block)


def test_accessibility(accessibility_result):
    assert all(accessibility_result[ACCESSIBILITY_FROM_COLUMN] >= 0) and all(
        accessibility_result[ACCESSIBILITY_TO_COLUMN] >= 0
    )
