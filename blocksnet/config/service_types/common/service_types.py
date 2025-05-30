from pathlib import Path

import pandas as pd


CURRENT_DIRECTORY = Path(__file__).parent
JSON_FILENAME = "default.json"
JSON_PATH = CURRENT_DIRECTORY / JSON_FILENAME

SERVICE_TYPES = pd.read_json(JSON_PATH).set_index("name")
