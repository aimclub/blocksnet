try:
    import pymoo
except:
    raise ImportError("pymoo package is required but not installed")

from .core import LandUseOptimizer
