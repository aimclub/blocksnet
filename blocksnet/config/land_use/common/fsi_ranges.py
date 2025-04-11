import pandas as pd
from ....enums import LandUse

fsi_ranges = {
    LandUse.RESIDENTIAL: (0.5, 3.0),
    LandUse.BUSINESS: (1.0, 3.0),
    LandUse.RECREATION: (0.05, 0.2),
    LandUse.SPECIAL: (0.05, 0.2),
    LandUse.INDUSTRIAL: (0.3, 1.5),
    LandUse.AGRICULTURE: (0.1, 0.2),
    LandUse.TRANSPORT: (0.2, 1.0),
}

FSI_RANGES = pd.DataFrame.from_dict(fsi_ranges, orient="index", columns=["fsi_min", "fsi_max"])
