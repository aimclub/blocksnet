import pandas as pd
from ....enums import LandUse

transition_probabilities = {
    LandUse.RESIDENTIAL: [0.7, 0.15, 0.02, 0.03, 0.05, 0.03, 0.02],
    LandUse.BUSINESS: [0.1, 0.75, 0.03, 0.05, 0.02, 0.02, 0.03],
    LandUse.RECREATION: [0.02, 0.04, 0.85, 0.02, 0.01, 0.05, 0.01],
    LandUse.SPECIAL: [0.03, 0.03, 0.04, 0.8, 0.04, 0.03, 0.03],
    LandUse.INDUSTRIAL: [0.05, 0.1, 0.01, 0.05, 0.65, 0.06, 0.08],
    LandUse.AGRICULTURE: [0.03, 0.02, 0.01, 0.03, 0.05, 0.8, 0.06],
    LandUse.TRANSPORT: [0.02, 0.04, 0.01, 0.02, 0.07, 0.06, 0.78],
}

PROBABILITY_MATRIX = pd.DataFrame(transition_probabilities, index=list(LandUse), columns=list(LandUse)).transpose()
