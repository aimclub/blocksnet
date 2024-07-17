from enum import Enum


class LandUse(Enum):
    """
    Enumeration for different types of land use.

    Attributes
    ----------
    RESIDENTIAL : str
        Represents residential areas.

    BUSINESS : str
        Represents business or commercial areas.

    RECREATION : str
        Represents recreational areas.

    SPECIAL : str
        Represents special areas with unique purposes.

    INDUSTRIAL : str
        Represents industrial areas.

    AGRICULTURE : str
        Represents agricultural areas.

    TRANSPORT : str
        Represents transport-related areas.
    """

    RESIDENTIAL = "residential"
    BUSINESS = "business"
    RECREATION = "recreation"
    SPECIAL = "special"
    INDUSTRIAL = "industrial"
    AGRICULTURE = "agriculture"
    TRANSPORT = "transport"
