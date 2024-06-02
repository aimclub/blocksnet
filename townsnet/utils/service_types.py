from .basic_plus_service_types import BASIC_PLUS_SERVICE_TYPES
from .basic_service_types import BASIC_SERVICE_TYPES
from .comfort_service_types import COMFORT_SERVICE_TYPES


SERVICE_TYPES = [
    *[
        {"category": "basic", "infrastructure": infrastructure, **st}
        for infrastructure, stypes in BASIC_SERVICE_TYPES.items()
        for st in stypes
    ],
    *[
        {"category": "basic_plus", "infrastructure": infrastructure, **st}
        for infrastructure, stypes in BASIC_PLUS_SERVICE_TYPES.items()
        for st in stypes
    ],
    *[
        {"category": "comfort", "infrastructure": infrastructure, **st}
        for infrastructure, stypes in COMFORT_SERVICE_TYPES.items()
        for st in stypes
    ],
]
