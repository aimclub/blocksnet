import pandera as pa
import pandas as pd
from loguru import logger
from pandera.typing import Index


class DfSchema(pa.DataFrameModel):
    idx: Index[int] = pa.Field(unique=True)

    class Config:
        strict = "filter"
        add_missing_columns = True
        coerce = True

    @classmethod
    def _check_instance(cls, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("An instance of DataFrame must be provided")

    @classmethod
    def _check_len(cls, df):
        if len(df) == 0:
            raise ValueError("Rows count must be greater than 0")

    @classmethod
    def _check_multi(cls, df):
        if df.index.nlevels > 1:
            raise ValueError("Index must not be multi-leveled")
        if df.columns.nlevels > 1:
            raise ValueError("Columns must not be multi-leveled")

    @classmethod
    def _reset_index_name(cls, df):
        index_name = df.index.name
        if index_name is not None:
            df.index.name = None
            # logger.warning(f'{cls.__name__} Got name in index : {index_name}. Resetting index name')

    @classmethod
    def _before_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @classmethod
    def _after_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @classmethod
    def validate(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        df = df.copy()
        cls._check_instance(df)
        cls._check_len(df)
        cls._check_multi(df)
        cls._reset_index_name(df)

        df = cls._before_validate(df)
        df = super().validate(df, **kwargs)
        df = cls._after_validate(df)
        return df.copy()

    @classmethod
    def _columns(cls) -> list:
        return list(cls.to_schema().columns.keys())

    @classmethod
    def columns_(cls) -> list:
        return cls._columns()

    @classmethod
    def create_empty(cls) -> pd.DataFrame:
        return pd.DataFrame([], columns=cls._columns())

    @pa.dataframe_parser
    @classmethod
    def _enforce_column_order(cls, df: pd.DataFrame):
        return df[cls._columns()]
