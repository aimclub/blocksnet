from enum import Enum


class LandUse(Enum):
    """Fine-grained land-use categories used throughout BlocksNet."""

    RESIDENTIAL = "residential"
    BUSINESS = "business"
    RECREATION = "recreation"
    INDUSTRIAL = "industrial"
    TRANSPORT = "transport"
    SPECIAL = "special"
    AGRICULTURE = "agriculture"

    def to_one_hot(self):
        """Represent the land-use as a one-hot encoded dictionary.

        Returns
        -------
        dict[str, int]
            Mapping from land-use values to ``0``/``1`` indicators.
        """

        return {lu.value: int(self == lu) for lu in LandUse}
