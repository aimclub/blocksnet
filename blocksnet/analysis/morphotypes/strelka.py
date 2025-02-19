from enum import Enum
import pandas as pd
from pandera.typing import Series
from pandera import Field
from ...utils.validation import DfSchema

MORPHOTYPE_COLUMN = "morphotype"


class LowRiseMorphotype(Enum):
    NON_RESIDENTIAL = "low-rise non-residential"
    INDIVIDUAL = "individual residential"
    LOW_RISE_MODEL = "low-rise model"


class MidRiseMorphotype(Enum):
    BASIC = "mid-rise"
    NON_RESIDENTIAL = "mid-rise non-residential"
    MICRODISTRICT = "mid-rise microdistrict"
    BLOCK = "mid-rise block"
    CENTRAL_MODEL = "central model"


class HighRiseMorphotype(Enum):
    NON_RESIDENTIAL = "high-rise non-residential"
    SOVIET_MICRODISTRICT = "high-rise soviet microdistrict"
    MODERN_MICRODISTRICT = "high-rise modern microdistrict"


class BlocksSchema(DfSchema):

    l: Series[float] = Field(ge=0, default=0)
    fsi: Series[float] = Field(ge=0, default=0)
    mxi: Series[float] = Field(ge=0, default=0)


def _interpret_high_rise(mxi: float, fsi: float) -> HighRiseMorphotype:
    if mxi < 0.1:
        return HighRiseMorphotype.NON_RESIDENTIAL
    if fsi <= 1.5:
        return HighRiseMorphotype.SOVIET_MICRODISTRICT
    return HighRiseMorphotype.MODERN_MICRODISTRICT


def _interpret_mid_rise(mxi: float, fsi: float) -> MidRiseMorphotype:
    if mxi < 0.2:
        return MidRiseMorphotype.NON_RESIDENTIAL
    if mxi < 0.45:
        if fsi <= 0.8:
            return MidRiseMorphotype.MICRODISTRICT
        return MidRiseMorphotype.BLOCK
    if mxi >= 0.6 and fsi > 1.5:
        return MidRiseMorphotype.CENTRAL_MODEL
    return MidRiseMorphotype.BASIC


def _interpret_low_rise(mxi: float, fsi: float) -> LowRiseMorphotype:
    if mxi < 0.05:
        return LowRiseMorphotype.NON_RESIDENTIAL
    if fsi <= 0.1:
        return LowRiseMorphotype.INDIVIDUAL
    return LowRiseMorphotype.LOW_RISE_MODEL


def _interpret_block(series: pd.Series) -> Enum:
    l, fsi, mxi = series[["l", "fsi", "mxi"]]
    if l >= 9:
        return _interpret_high_rise(mxi, fsi)
    if l >= 4:
        return _interpret_mid_rise(mxi, fsi)
    return _interpret_low_rise(mxi, fsi)


def get_morphotypes(blocks_df: pd.DataFrame) -> pd.DataFrame:
    blocks_df = BlocksSchema(blocks_df)
    # get morphotypes
    developed_blocks_df = blocks_df[blocks_df.fsi > 0].copy()
    developed_blocks_df[MORPHOTYPE_COLUMN] = developed_blocks_df.apply(lambda s: _interpret_block(s).value, axis=1)
    # merge results
    blocks_df = blocks_df.join(developed_blocks_df[[MORPHOTYPE_COLUMN]])  # return filtered blocks
    return blocks_df
