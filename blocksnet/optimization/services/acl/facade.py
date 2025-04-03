import copy
from typing import Dict, List, Set
from functools import reduce
import sys

import geopandas as gpd
import numpy as np
from numpy.typing import ArrayLike

from blocksnet.method.annealing_optimizer import Indicator, LandUse
from blocksnet.models import City, ServiceType

from .chekers.area_checker import AreaChecker
# from .provision_adapter import ProvisionAdapter
from .provision_adapter import DiversityProvisionsAdapter
from .variable_adapters.variable_adapter import VariableAdapter

VACANT_AREA_COEF = 0.8


class BlocksNetFacade:
    """
    The BlocksnetFacade class serves as a high-level interface for managing various
    domain-related operations in an urban planning context. 

    Responsibilities
    ----------------
  

    Attributes
    ----------
   
    """

    def __init__(
        self,
        blocks_lu: Dict[int, LandUse],
        chosen_service_types: List[str],
        variable_adapter: VariableAdapter,
    ) -> None:
       pass
