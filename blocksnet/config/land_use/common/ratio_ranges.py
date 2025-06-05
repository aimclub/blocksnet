import pandas as pd
from ....enums import LandUse

ratio_ranges = {
    LandUse.RESIDENTIAL: (1, 3),
    LandUse.BUSINESS: (1, 4),
    LandUse.RECREATION: (1, 7),
    LandUse.SPECIAL: (1, 6),
    LandUse.INDUSTRIAL: (1, 5),
    LandUse.AGRICULTURE: (1, 4),
    LandUse.TRANSPORT: (1, 7),
}

RATIO_RANGES = pd.DataFrame.from_dict(ratio_ranges, orient="index", columns=["lower", "upper"])
