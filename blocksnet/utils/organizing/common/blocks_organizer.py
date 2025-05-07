from functools import singledispatchmethod
import geopandas as gpd
import pandas as pd
from loguru import logger
from ...validation import ensure_crs


def _join_dfs(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    df_1_columns = set(df_1.columns)
    df_2_columns = set(df_2.columns)
    drop_columns = list(df_1_columns & df_2_columns)
    return df_1.join(df_2.drop(columns=drop_columns))


class BlocksOrganizer:
    def __init__(self, blocks: gpd.GeoDataFrame):
        self._blocks = blocks
        self._data = {}

    @property
    def blocks(self):
        return self._blocks.copy()

    @property
    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __setitem__(self, key: str, value: pd.DataFrame | gpd.GeoDataFrame):
        if not isinstance(key, str):
            raise KeyError("Key must be str")
        if key in self._data:
            logger.warning(f"Key {key} already exists. Replacing")

        is_df = isinstance(value, pd.DataFrame)
        is_gdf = isinstance(value, gpd.GeoDataFrame)
        if not is_df and not is_gdf:
            raise ValueError("Value must be an instance of DataFrame or GeoDataFrame")
        value = value.copy()
        if is_gdf:
            ensure_crs(self.blocks, value)
        self._data[key] = value

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError(f"Can't access data with such argument type {type(arg)}")

    @__getitem__.register(str)
    def _(self, key: str) -> gpd.GeoDataFrame:
        if key not in self._data:
            raise KeyError(f"No data found for key {key}")
        blocks = self.blocks
        df = self._data[key]
        return _join_dfs(blocks, df)

    @__getitem__.register(tuple)
    def _(self, keys: tuple[str]) -> gpd.GeoDataFrame:
        dfs = []

        for key in keys:
            dfs.append(self[key])

        df = dfs[0]
        for i in range(len(dfs) - 1):
            df = _join_dfs(dfs[i], dfs[i + 1])

        return df
