import pandas as pd
from ....enums import LandUse

area_ranges = {
    LandUse.RESIDENTIAL: (2_000, 100_000),
    LandUse.BUSINESS: (50_000, 150_000),
    LandUse.RECREATION: (10_000, 1_000_000),
    LandUse.SPECIAL: (50_000, 500_000),
    LandUse.INDUSTRIAL: (10_000, 800_000),
    LandUse.AGRICULTURE: (300_000, 1_000_000),
    LandUse.TRANSPORT: (50_000, 500_000),
}

AREA_RANGES = pd.DataFrame.from_dict(area_ranges, orient="index", columns=["lower", "upper"])
