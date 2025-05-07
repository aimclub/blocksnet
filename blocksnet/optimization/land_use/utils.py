import random
from ...enums import LandUse
from ...config import land_use_config

LAND_USE_LIST = list(LandUse)


def direct_transform_lu(lu: LandUse | None) -> int:
    if isinstance(lu, LandUse):
        return LAND_USE_LIST.index(lu)
    return -1


def reverse_transform_lu(lu_id: int) -> LandUse | None:
    if lu_id >= 0 and lu_id < len(LAND_USE_LIST):
        return LAND_USE_LIST[lu_id]
    return None


def _get_possible_land_use(lu: LandUse | None) -> list[LandUse]:
    possible_lus = LAND_USE_LIST
    if isinstance(lu, LandUse):
        poss_mx = land_use_config.possibility_matrix
        poss_s = poss_mx.loc[lu]
        possible_lus = list(poss_s[poss_s].index)
    return possible_lus


def generate_gene_space(lus: list[LandUse | None]) -> list[list[int]]:
    return [[direct_transform_lu(plu) for plu in _get_possible_land_use(lu)] for lu in lus]
