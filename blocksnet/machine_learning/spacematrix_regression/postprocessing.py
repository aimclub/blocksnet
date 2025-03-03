import torch
import pandas as pd
from .schemas import BlocksIndicatorsSchema


def out_to_df(blocks_df: pd.DataFrame, out: torch.Tensor) -> pd.DataFrame:
    index = blocks_df.index
    columns = BlocksIndicatorsSchema.to_schema().columns
    data = out.detach().numpy()
    return pd.DataFrame(data, index=index, columns=columns)
