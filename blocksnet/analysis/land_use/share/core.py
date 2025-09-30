import pandas as pd
from .schemas import BlocksSchema
from ....enums.land_use import LandUse


def calculate_land_use_shares(blocks_df: pd.DataFrame, area: float | None = None) -> dict[LandUse, float]:
    """Calculate proportional area occupied by each land-use category.

    Parameters
    ----------
    blocks_df : pandas.DataFrame
        Block dataframe validated by :class:`BlocksSchema`.
    area : float, optional
        Total area used to normalise shares. Defaults to the sum of
        ``site_area``.

    Returns
    -------
    dict of LandUse to float
        Mapping from land-use categories to their area share.

    Raises
    ------
    ValueError
        If ``blocks_df`` fails validation or total area is zero.
    """
    blocks_df = BlocksSchema(blocks_df)

    if area is None:
        area = blocks_df.site_area.sum()

    shares = {lu: 0.0 for lu in list(LandUse)}
    for lu in shares:
        df = blocks_df[blocks_df.land_use == lu]
        share = df.site_area.sum() / area
        shares[lu] = float(share)

    return shares
