import pandas as pd
from .schemas import BlocksSchema
from ....enums.land_use import LandUse


def land_use_shares(blocks_df: pd.DataFrame, area: float | None = None) -> dict[LandUse, float]:
    blocks_df = BlocksSchema(blocks_df)

    if area is None:
        area = blocks_df.site_area.sum()

    shares = {lu: 0.0 for lu in list(LandUse)}
    for lu in shares:
        df = blocks_df[blocks_df.land_use == lu]
        share = df.site_area.sum() / area
        shares[lu] = float(share)

    return shares
