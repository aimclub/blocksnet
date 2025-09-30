import random
from ...enums import LandUse
from ...config import land_use_config

LAND_USE_LIST = list(LandUse)


def direct_transform_lu(lu: LandUse | None) -> int:
    """Direct transform lu.

    Parameters
    ----------
    lu : LandUse | None
        Description.

    Returns
    -------
    int
        Description.

    """
    if isinstance(lu, LandUse):
        return LAND_USE_LIST.index(lu)
    return -1


def reverse_transform_lu(lu_id: int) -> LandUse | None:
    """Reverse transform lu.

    Parameters
    ----------
    lu_id : int
        Description.

    Returns
    -------
    LandUse | None
        Description.

    """
    if lu_id >= 0 and lu_id < len(LAND_USE_LIST):
        return LAND_USE_LIST[lu_id]
    return None


def _get_possible_land_use(lu: LandUse | None) -> list[LandUse]:
    """Get possible land use.

    Parameters
    ----------
    lu : LandUse | None
        Description.

    Returns
    -------
    list[LandUse]
        Description.

    """
    possible_lus = LAND_USE_LIST
    if isinstance(lu, LandUse):
        poss_mx = land_use_config.possibility_matrix
        poss_s = poss_mx.loc[lu]
        possible_lus = list(poss_s[poss_s].index)
    return possible_lus


def generate_gene_space(lus: list[LandUse | None]) -> list[list[int]]:
    """Generate gene space.

    Parameters
    ----------
    lus : list[LandUse | None]
        Description.

    Returns
    -------
    list[list[int]]
        Description.

    """
    return [[direct_transform_lu(plu) for plu in _get_possible_land_use(lu)] for lu in lus]
