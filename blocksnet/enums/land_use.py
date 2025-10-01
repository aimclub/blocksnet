from enum import Enum


class LandUse(Enum):

    RESIDENTIAL = "residential"
    BUSINESS = "business"
    RECREATION = "recreation"
    INDUSTRIAL = "industrial"
    TRANSPORT = "transport"
    SPECIAL = "special"
    AGRICULTURE = "agriculture"

    def to_one_hot(self):
        return {lu.value: int(self == lu) for lu in LandUse}
