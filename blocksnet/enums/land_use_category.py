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
    INDUSTRIAL = "INDUSTRIAL"
    RECREATION = "RECREATION"
    BUSINESS = "BUSINESS"
    RESIDENTIAL = "RESIDENTIAL"
    AGRICULTURE = "AGRICULTURE"
    SPECIAL = "SPECIAL"
    TRANSPORT = "TRANSPORT"

    _REVERSE_MAP = None
    # MIXED_USE = "MIXED_USE"
    # LARGE_AREA = "LARGE_AREA"
    # ENGINEERING = "ENGINEERING"
    # RECREATION = "RECREATION"
    # INDUSTRIAL = "INDUSTRIAL"
    # LIVING = "LIVING"
    # NOT_LIVING = "NOT_LIVING"

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

# LU_MAPPING = { # 1 СЦЕНАРИЙ - ПРОСТОР 
#     LandUse.RESIDENTIAL: LandUseCategory.RESIDENTIAL,
#     LandUse.BUSINESS: LandUseCategory.BUSINESS,
#     LandUse.RECREATION: LandUseCategory.RECREATION,
#     LandUse.AGRICULTURE: LandUseCategory.RECREATION,
#     LandUse.SPECIAL: LandUseCategory.RECREATION,
#     LandUse.INDUSTRIAL: LandUseCategory.INDUSTRIAL,
#     LandUse.TRANSPORT: LandUseCategory.INDUSTRIAL,
#     # Other LandUse values (e.g., TRANSPORT, SPECIAL, AGRICULTURE) map to None
# }

# LU_MAPPING = { # 2 сценарий - по размеру
#     LandUse.RESIDENTIAL: LandUseCategory.MIXED_USE,
#     LandUse.BUSINESS: LandUseCategory.MIXED_USE,
#     LandUse.RECREATION: LandUseCategory.LARGE_AREA,
#     LandUse.AGRICULTURE: LandUseCategory.LARGE_AREA,
#     LandUse.SPECIAL: LandUseCategory.ENGINEERING,
#     LandUse.INDUSTRIAL: LandUseCategory.LARGE_AREA,
#     LandUse.TRANSPORT: LandUseCategory.ENGINEERING,
#     # Other LandUse values (e.g., TRANSPORT, SPECIAL, AGRICULTURE) map to None
# }

# LU_MAPPING = { # 3 сценарий - по функции
#     LandUse.RESIDENTIAL: LandUseCategory.MIXED_USE,
#     LandUse.BUSINESS: LandUseCategory.MIXED_USE,
#     LandUse.RECREATION: LandUseCategory.RECREATION,
#     LandUse.AGRICULTURE: LandUseCategory.RECREATION,
#     LandUse.SPECIAL: LandUseCategory.ENGINEERING,
#     LandUse.INDUSTRIAL: LandUseCategory.ENGINEERING,
#     LandUse.TRANSPORT: LandUseCategory.INDUSTRIAL,
#     # Other LandUse values (e.g., TRANSPORT, SPECIAL, AGRICULTURE) map to None
# }

# LU_MAPPING = { # 4 сценарий - по жилой/нежилой
#     LandUse.RESIDENTIAL: LandUseCategory.LIVING,
#     LandUse.BUSINESS: LandUseCategory.NOT_LIVING,
#     LandUse.RECREATION: LandUseCategory.NOT_LIVING,
#     LandUse.AGRICULTURE: LandUseCategory.NOT_LIVING,
#     LandUse.SPECIAL: LandUseCategory.NOT_LIVING,
#     LandUse.INDUSTRIAL: LandUseCategory.NOT_LIVING,
#     LandUse.TRANSPORT: LandUseCategory.NOT_LIVING,
#     # Other LandUse values (e.g., TRANSPORT, SPECIAL, AGRICULTURE) map to None
# }

LU_MAPPING = { # 0 сценарий - все типы
    LandUse.RESIDENTIAL: LandUseCategory.RESIDENTIAL,
    LandUse.BUSINESS: LandUseCategory.BUSINESS,
    LandUse.RECREATION: LandUseCategory.RECREATION,
    LandUse.AGRICULTURE: LandUseCategory.AGRICULTURE,
    LandUse.SPECIAL: LandUseCategory.SPECIAL,
    LandUse.INDUSTRIAL: LandUseCategory.INDUSTRIAL,
    LandUse.TRANSPORT: LandUseCategory.TRANSPORT,
    # Other LandUse values (e.g., TRANSPORT, SPECIAL, AGRICULTURE) map to None
}