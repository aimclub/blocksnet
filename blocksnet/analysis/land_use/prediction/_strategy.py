import os
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from pathlib import Path
from blocksnet.machine_learning.strategy.sklearn.ensemble.voting.classification_strategy import SKLearnVotingClassificationStrategy

CURRENT_DIRECTORY = Path(__file__).parent
ARTIFACTS_DIRECTORY = str(CURRENT_DIRECTORY / "artifacts")

BASE_PARAMS = {"random_state": 42, "n_jobs": -1}
CPU = max(1, min(8, os.cpu_count() or 1))
MODEL_PARAMS = {
    "rf": {
        "n_estimators": 120,          
        "max_depth": 7,
        "class_weight": "balanced",
        "max_samples": 0.25,          
        "min_samples_leaf": 10,       
        **BASE_PARAMS
    },
    "xgb": {
        "n_estimators": 150,          
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,             
        "colsample_bytree": 0.8,
        "tree_method": "hist",        
        "n_jobs": CPU                 
    },
    "lgb": {
        "n_estimators": 200,
        "max_depth": 7,
        "learning_rate": 0.05,
        "class_weight": "balanced",
        "num_threads": CPU            
    },
    "hgb": {
        "max_iter": 200,
        "max_depth": 7,
        "learning_rate": 0.05,
        "random_state": 42
    }
}
estimators = [
    ("rf",  RandomForestClassifier(**MODEL_PARAMS["rf"])),
    ("xgb", XGBClassifier(**MODEL_PARAMS["xgb"])),
    ("lgb", LGBMClassifier(**MODEL_PARAMS["lgb"])),
    ("hgb", HistGradientBoostingClassifier(**MODEL_PARAMS["hgb"])),
]

strategy = SKLearnVotingClassificationStrategy(estimators)
strategy.load(ARTIFACTS_DIRECTORY)