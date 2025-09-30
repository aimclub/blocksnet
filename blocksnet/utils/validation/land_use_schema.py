import pandas as pd
from pandera import Field, parser, check
from pandera.typing import Series
from .df_schema import DfSchema
from ...enums import LandUse


class LandUseSchema(DfSchema):
    """LandUseSchema class.

    """
    land_use: Series[object] = Field(nullable=True)

    @parser("land_use")
    @classmethod
    def _parse_land_use(cls, land_use: pd.Series) -> pd.Series:
        """Parse land use.

        Parameters
        ----------
        land_use : pd.Series
            Description.

        Returns
        -------
        pd.Series
            Description.

        """
        return land_use.apply(lambda lu: LandUse(lu.lower()) if isinstance(lu, str) else lu)

    @check("land_use")
    @classmethod
    def _check_land_use(cls, land_use: pd.Series) -> pd.Series:
        """Check land use.

        Parameters
        ----------
        land_use : pd.Series
            Description.

        Returns
        -------
        pd.Series
            Description.

        """
        return land_use.apply(lambda lu: True if lu is None else isinstance(lu, LandUse))
