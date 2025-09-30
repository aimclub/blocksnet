"""Pre-configured classification strategy artefacts."""

from pathlib import Path

from blocksnet.machine_learning.strategy.catboost import CatBoostClassificationStrategy

CURRENT_DIRECTORY = Path(__file__).parent
ARTIFACTS_DIRECTORY = str(CURRENT_DIRECTORY / "artifacts")

strategy = CatBoostClassificationStrategy()
strategy.load(ARTIFACTS_DIRECTORY)
