"""Validation schema for connectivity indicators."""

from pandera.typing import Series
from pandera import Field

from blocksnet.utils.validation import DfSchema


class BlocksAccessibilitySchema(DfSchema):
    """Ensure accessibility values are non-negative prior to inversion."""

    accessibility: Series[float] = Field(ge=0)
