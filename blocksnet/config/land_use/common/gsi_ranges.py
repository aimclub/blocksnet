import pandas as pd

from ....enums import LandUse


gsi_ranges = {
    LandUse.RESIDENTIAL: (0.2, 0.8),
    LandUse.BUSINESS: (0.0, 0.8),
    LandUse.RECREATION: (0.0, 0.3),
    LandUse.SPECIAL: (0.05, 0.15),
    LandUse.INDUSTRIAL: (0.2, 0.8),
    LandUse.AGRICULTURE: (0.0, 0.6),
    LandUse.TRANSPORT: (0.0, 0.8),
}

GSI_RANGES = pd.DataFrame.from_dict(gsi_ranges, orient="index", columns=["fsi_min", "fsi_max"])
