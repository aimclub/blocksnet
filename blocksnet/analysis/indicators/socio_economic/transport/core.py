import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from blocksnet.enums.land_use import LandUse
from blocksnet.relations import validate_accessibility_matrix
from .schemas import BlocksAreaSchema, BlocksAccessibilitySchema, NetworkSchema
from .indicator import TransportIndicator
from ..const import M_IN_KM, SQM_IN_SQKM, MIN_IN_H


def _calculate_area(blocks_df: pd.DataFrame) -> float:
    blocks_df = BlocksAreaSchema(blocks_df)
    return blocks_df["site_area"].sum() / SQM_IN_SQKM


def _calculate_length(network: nx.Graph | gpd.GeoDataFrame) -> float:

    from blocksnet.relations import accessibility_graph_to_gdfs

    if isinstance(network, nx.Graph):
        _, edges_df = accessibility_graph_to_gdfs(network)
        network_df = NetworkSchema(edges_df.reset_index(drop=True))
    elif isinstance(network, gpd.GeoDataFrame):
        network_df = NetworkSchema(network)
    else:
        raise TypeError("Network must be an instance of nx.Graph or gpd.GeoDataFrame")

    return network_df.length.sum() / M_IN_KM


def _calculate_connectivity(blocks_df: pd.DataFrame, accessibility_matrix: pd.DataFrame) -> float:

    residential_idx = blocks_df[blocks_df["land_use"] == LandUse.RESIDENTIAL].index
    acc_mx = accessibility_matrix.loc[residential_idx, residential_idx]
    return np.mean(acc_mx.to_numpy()) / MIN_IN_H


def _calculate_count(blocks_df: pd.DataFrame, name: str) -> tuple[int, pd.DataFrame]:
    column = f"count_{name}"
    if column in blocks_df:
        blocks_df = BlocksAccessibilitySchema(blocks_df.rename(columns={column: "count"}))
    else:
        raise RuntimeError(f"Column {column} not found in blocks")

    count = blocks_df["count"].sum()
    return count, blocks_df


def _calculate_accessibility(counts_df: pd.DataFrame, accessibility_matrix: pd.DataFrame) -> float:

    residential_idx = counts_df[counts_df["land_use"] == LandUse.RESIDENTIAL].index
    services_idx = counts_df[counts_df["count"] > 0].index
    acc_mx = accessibility_matrix.loc[residential_idx, services_idx]
    accs = acc_mx.min(axis=1)
    return np.mean(accs.to_numpy()) / MIN_IN_H


def calculate_transport_indicators(
    blocks_df: pd.DataFrame, accessibility_matrix: pd.DataFrame, network: nx.Graph | gpd.GeoDataFrame
) -> dict[TransportIndicator, float]:

    validate_accessibility_matrix(accessibility_matrix, blocks_df)

    area = _calculate_area(blocks_df)
    length = _calculate_length(network)
    density = length / area
    connectivity = _calculate_connectivity(blocks_df, accessibility_matrix)

    result = {
        TransportIndicator.ROAD_NETWORK_DENSITY: float(density),
        TransportIndicator.SETTLEMENTS_CONNECTIVITY: float(connectivity),
        TransportIndicator.ROAD_NETWORK_LENGTH: float(length),
    }

    mapping = {
        "fuel": (TransportIndicator.FUEL_STATIONS_COUNT, TransportIndicator.AVERAGE_FUEL_STATION_ACCESSIBILITY),
        "train_station": (
            TransportIndicator.RAILWAY_STOPS_COUNT,
            TransportIndicator.AVERAGE_RAILWAY_STOP_ACCESSIBILITY,
        ),
    }

    for name, indicators in mapping.items():
        count_indicator, accessibility_indicator = indicators
        count, df = _calculate_count(blocks_df, name)
        accessibility = _calculate_accessibility(df, accessibility_matrix)
        result[count_indicator] = int(count)
        result[accessibility_indicator] = float(accessibility)

    return result
