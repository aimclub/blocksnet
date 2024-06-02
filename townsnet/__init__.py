"""
Provision assessment for cities of an urban region
"""

import importlib

__version__ = importlib.metadata.version("townsnet")
__author__ = ""
__email__ = ""
__credits__ = []
__license__ = "BSD-3"

from .method import *
from .models import *
from .utils import *