import pandas as pd
from .service_types import SERVICE_TYPES

data = [
    {"service_type": service_type_name, **unit}
    for service_type_name, units in SERVICE_TYPES["units"].items()
    for unit in units
]
UNITS = pd.DataFrame(data)
