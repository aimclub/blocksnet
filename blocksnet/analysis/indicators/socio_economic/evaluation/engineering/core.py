import pandas as pd
from .schemas import CountSchema
from .indicator import EngineeringIndicator


def evaluate_engineering(count_dfs: dict[EngineeringIndicator, pd.DataFrame]):
    count_dfs = {ind: CountSchema(df) for ind, df in count_dfs.items()}
    counts = {ind: int(df["count"].sum()) for ind, df in count_dfs.items()}
    return pd.DataFrame.from_dict(counts, orient="index")
