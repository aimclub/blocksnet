"""Pick 5 residential-ish target blocks for the services-optimizer verification run."""

import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd
import pandas as pd

SELECTED_PATH = "examples/optimization/services/5blocks_audit_2026_07_17/data/_selected_blocks.csv"

b = gpd.read_file("data/blocks_with_buildings_and_services.gpkg")

# 5 representative residential-ish blocks: vary in size and current population
# to exercise the optimizer across different scales.
candidates = b[(b.is_living.fillna(0) > 0) & (b.population.fillna(0) > 0)].copy()
candidates["area"] = candidates.geometry.area
candidates = candidates.sort_values("area", ascending=False).reset_index()
print(f"Residential candidates: {len(candidates)}")

# Pick blocks across different area quartiles so the optimizer sees
# different BFA budgets and population mixes.
sizes = candidates.area.describe()
print("Area distribution:")
print(sizes)

# Pick top-1, top-1/4, top-1/2, top-3/4, median-1.
n = len(candidates)
picks = []
for frac in [0.999, 0.75, 0.5, 0.25, 0.05]:
    idx = int(n * (1 - frac))
    picks.append(candidates.iloc[idx])
picks_df = pd.DataFrame(picks)
picks_df = picks_df[["index", "area", "population", "build_floor_area", "is_living", "living_area"]]
print("Selected target blocks:")
print(picks_df.to_string(index=False))

# Persist the block IDs for the runner script.
picks_df.to_csv(SELECTED_PATH, index=False)
print(f"Saved block ids to {SELECTED_PATH}")
