from enum import Enum


class BlockCategory(Enum):
    """Classification labels assigned to generated blocks."""

    INVALID = "invalid"
    NORMAL = "normal"
    LARGE = "large"
