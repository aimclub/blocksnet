from enum import Enum
from blocksnet.enums import LandUse
    
class LandUseCategory(Enum):
    """Categories of land-use types supported by BlocksNet."""
    URBAN = "urban"
    NON_URBAN = "non_urban"
    INDUSTRIAL = "industrial"
    _REVERSE_MAP = None

    @classmethod
    def from_land_use(cls, lu: LandUse) -> "LandUseCategory | None":
        """Map a :class:`LandUse` member to its broader category.

        Parameters
        ----------
        lu : LandUse
            Land-use value to convert.

        Returns
        -------
        LandUseCategory or None
            Matching category if a mapping exists, otherwise ``None``.
        """
        return LU_MAPPING.get(lu)

    def to_land_use(self) -> set[LandUse]:
        """Return :class:`LandUse` members belonging to the category.

        Returns
        -------
        set of LandUse
            Set of land-use values whose category equals ``self``.
        """
        if LandUseCategory._REVERSE_MAP is None:
            LandUseCategory._REVERSE_MAP = {}
            for k, v in LU_MAPPING.items():
                LandUseCategory._REVERSE_MAP.setdefault(v, set()).add(k)
        return LandUseCategory._REVERSE_MAP.get(self, set())

LU_MAPPING = {
    LandUse.RESIDENTIAL: LandUseCategory.URBAN,
    LandUse.BUSINESS: LandUseCategory.URBAN,
    LandUse.RECREATION: LandUseCategory.NON_URBAN,
    LandUse.TRANSPORT: LandUseCategory.NON_URBAN,
    LandUse.SPECIAL: LandUseCategory.NON_URBAN,
    LandUse.AGRICULTURE: LandUseCategory.NON_URBAN,
    LandUse.INDUSTRIAL: LandUseCategory.INDUSTRIAL,
}