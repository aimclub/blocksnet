import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .schemas import BlocksSchema

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 11
N_INIT = "auto"
CLUSTER_COLUMN = "cluster"
MORPHOTYPE_COLUMN = "morphotype"
SEPARATOR = " "


def _get_interpretation_column(column):
    return f"{column}_interpretation"


INTERPRETATIONS = {
    "l": {0: "low-rise", 3: "mid-rise", 6: "high-mid-rise", 10: "high-rise", 17: "tower"},
    "fsi": {0: "low-density", 1: None, 2: "high-density"},
    "mxi": {0: "non-residential", 0.22: "mixed-use", 0.55: "residential"},
}


def _interpret_value(value: float, interpretation_dict: dict[float, str]):
    keys = [key for key in interpretation_dict.keys() if key <= value]
    return interpretation_dict[max(keys)]


def _interpret_cluster(series: pd.Series):
    interpretations_columns = [_get_interpretation_column(column) for column in INTERPRETATIONS.keys()]
    interpretations = [i for i in series[interpretations_columns] if i is not None]
    return str.join(SEPARATOR, interpretations)


def _interpret_clusters(clusters_df: pd.DataFrame) -> pd.DataFrame:
    for column, interpretation_dict in INTERPRETATIONS.items():
        clusters_df[_get_interpretation_column(column)] = clusters_df[column].apply(
            lambda v: _interpret_value(v, interpretation_dict)
        )
    clusters_df[MORPHOTYPE_COLUMN] = clusters_df.apply(_interpret_cluster, axis=1)
    return clusters_df


def _scale(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


def _clusterize(df: pd.DataFrame, n_clusters: int, random_state: int) -> pd.DataFrame:
    df = df.copy()
    df_scaled = _scale(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=N_INIT)
    df[CLUSTER_COLUMN] = kmeans.fit(df_scaled).labels_
    return df


def get_spacematrix_morphotypes(
    blocks_df: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS, random_state: int = DEFAULT_RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks_df = BlocksSchema(blocks_df)
    # clusterize blocks
    developed_blocks_df = blocks_df[blocks_df.fsi > 0]
    developed_blocks_df = _clusterize(developed_blocks_df, n_clusters, random_state)
    # interpret clusters
    clusters_df = developed_blocks_df.groupby(CLUSTER_COLUMN).median()
    clusters_df = _interpret_clusters(clusters_df)
    # merge results
    developed_blocks_df = developed_blocks_df.merge(
        clusters_df[[MORPHOTYPE_COLUMN]], left_on=CLUSTER_COLUMN, right_index=True
    )  # assign morphotypes
    blocks_df = blocks_df.join(developed_blocks_df[[CLUSTER_COLUMN, MORPHOTYPE_COLUMN]])  # return filtered blocks
    return blocks_df, clusters_df
