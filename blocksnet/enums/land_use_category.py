from enum import Enum
from blocksnet.enums import LandUse
class LandUseCategory(Enum):
    """High-level land use categories used for prediction.
    Supported categories:
    - INDUSTRIAL
    - RECREATION
    - BUSINESS
    - RESIDENTIAL
    """
    RESIDENTIAL = "RESIDENTIAL"
    BUSINESS = "BUSINESS"
    RECREATION = "RECREATION"
    INDUSTRIAL = "INDUSTRIAL"

    _REVERSE_MAP = None

    @classmethod
    def from_land_use(cls, lu: LandUse) -> "LandUseCategory | None":
        """Convert a LandUse object to a LandUseCategory instance.
        
        Args:
            lu: A LandUse object to be converted to a category.
            
        Returns:
            The corresponding LandUseCategory instance if found, None otherwise.
        """
        return LU_MAPPING.get(lu)

    def to_land_use(self) -> set[LandUse]:
        """Convert the LandUseCategory to a set of LandUse objects.
        
        This method uses a reverse mapping (created on first call) to find all
        LandUse objects that belong to this category.
        
        Returns:
            A set of LandUse objects that belong to this category.
        """
        if LandUseCategory._REVERSE_MAP is None:
            LandUseCategory._REVERSE_MAP = {}
            for k, v in LU_MAPPING.items():
                LandUseCategory._REVERSE_MAP.setdefault(v, set()).add(k)
        return LandUseCategory._REVERSE_MAP.get(self, set())

LU_MAPPING = {
    LandUse.RESIDENTIAL: LandUseCategory.RESIDENTIAL,
    LandUse.BUSINESS: LandUseCategory.BUSINESS,
    LandUse.RECREATION: LandUseCategory.RECREATION,
    LandUse.AGRICULTURE: LandUseCategory.RECREATION,
    LandUse.SPECIAL: LandUseCategory.RECREATION,
    LandUse.INDUSTRIAL: LandUseCategory.INDUSTRIAL,
    LandUse.TRANSPORT: LandUseCategory.INDUSTRIAL,
    # Other LandUse values (e.g., TRANSPORT, SPECIAL, AGRICULTURE) map to None
}
