import pandera as pa
import pandas as pd
from loguru import logger
from pandera.typing import Index
from pandera.errors import SchemaErrors

TOP_N_ERRORS = 5


def _index_level_errors(cases_df: pd.DataFrame) -> list[str]:
    idx_df = cases_df[cases_df["schema_context"] == "Index"].copy()
    messages = []

    if not idx_df.empty:
        summary_df = (
            idx_df.groupby(["check"])
            .agg(
                n_cases=("failure_case", "size"),
                values=("failure_case", lambda x: list(pd.Series(x).head(TOP_N_ERRORS))),
            )
            .reset_index()
        )

        for _, row in summary_df.iterrows():
            values_str = ", ".join(map(str, row["values"]))
            if row["n_cases"] > TOP_N_ERRORS:
                values_str += ", ..."
            messages.append(f'{row["n_cases"]} index-level errors at check "{row["check"]}": {values_str}')

    return messages


def _dataframe_level_errors(cases_df: pd.DataFrame) -> list[str]:
    cases_df = cases_df[cases_df["schema_context"].isin(["DataFrame", "DataFrameSchema"])].copy()
    messages = []
    if not cases_df.empty:
        summary_df = (
            cases_df.groupby(["column", "check"]).agg(failure_case=("failure_case", lambda x: list(x))).reset_index()
        )
        for _, row in summary_df.iterrows():
            failure_cases = ", ".join(map(str, row["failure_case"]))
            message = f'{len(row["failure_case"])} dataframe-level errors at check "{row["check"]}": {failure_cases}'
            messages.append(message)
    return messages


def _column_level_errors(cases_df: pd.DataFrame) -> list[str]:
    cases_df = cases_df[cases_df["schema_context"] == "Column"].copy()
    messages = []
    if not cases_df.empty:
        summary_df = (
            cases_df.groupby(["column", "check"])
            .agg(n_cases=("index", "size"), cases=("index", lambda x: list(x.sort_values().head(TOP_N_ERRORS))))
            .reset_index()
        )
        for _, row in summary_df.iterrows():
            check = row["check"]
            column = row["column"]
            n_cases = row["n_cases"]
            cases = [str(case) for case in row["cases"]]
            message = f'{n_cases} column-level errors at column "{column}" at check "{check}": {str.join(", ",cases)}{", ..." if n_cases > TOP_N_ERRORS else ""}'
            messages.append(message)
    return messages


def _log_schema_errors(e: SchemaErrors):
    cases_df = e.failure_cases
    messages = ["Schema validation errors:"]
    messages.extend(_dataframe_level_errors(cases_df))
    messages.extend(_index_level_errors(cases_df))
    messages.extend(_column_level_errors(cases_df))
    logger.error(str.join("\n", messages))


class DfSchema(pa.DataFrameModel):
    """Base class for validating pandas DataFrames used across BlocksNet."""
    idx: Index[int] = pa.Field(unique=True)

    class Config:
        strict = "filter"
        add_missing_columns = True
        coerce = True

    def __new__(cls, *args, **kwargs) -> pd.DataFrame:
        return cls.validate(*args, **kwargs)

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

    @classmethod
    def _before_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @classmethod
    def _after_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @classmethod
    def validate(cls, df: pd.DataFrame, allow_empty: bool = False) -> pd.DataFrame:
        """Validate and coerce a DataFrame according to the schema.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to validate.
        allow_empty : bool, default=False
            Whether to allow empty dataframes to pass validation.

        Returns
        -------
        pandas.DataFrame
            Validated copy of the input dataframe.

        Raises
        ------
        ValueError
            If the dataframe fails schema validation or required structure
            checks.
        """

        df = df.copy()

        cls._check_instance(df)
        if not allow_empty:
            cls._check_len(df)
        cls._check_multi(df)
        cls._reset_index_name(df)

        df = cls._before_validate(df)
        try:
            df = super().to_schema().validate(df, lazy=True)
        except SchemaErrors as e:
            _log_schema_errors(e)
            raise ValueError(
                f"{e.schema.name} validation failed. Please check log and verify data according to schema"
            ) from None

        df = cls._enforce_columns_order(df)
        df = cls._after_validate(df)

        return df.copy()

    @classmethod
    def columns_(cls) -> list:
        """Return schema column names in order of definition."""

        return list(cls.to_schema().columns.keys())

    @classmethod
    def create_empty(cls) -> pd.DataFrame:
        """Create an empty dataframe that satisfies the schema."""

        return pd.DataFrame([], columns=cls.columns_())

    @classmethod
    def _enforce_columns_order(cls, df: pd.DataFrame):
        return df[cls.columns_()]
