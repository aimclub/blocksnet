"""
Class holding parameters for blocks cutter is defined here.
"""
from pydantic import BaseModel, Field


class BlocksCutterParameters(BaseModel):
    """
    Blocks cutter parameters
    """

    roads_buffer: int = Field(3, ge=0)
    """roads geometry buffer in meters"""
    block_cutoff_ratio: float = Field(0.15, ge=0)
    """block polygon perimeter to area ratio. Objects with bigger ratio will be dropped."""
    block_cutoff_area: int = Field(1_400, ge=0)
    """block polygon area filter in meters. Objects with smaller area will be dropped."""
    park_cutoff_area: int = Field(10_000, ge=0)
    """park polygon area filter in meters. Objects with smaller area will be dropped."""
    filter_lu: bool = False
    """filter landuse """
