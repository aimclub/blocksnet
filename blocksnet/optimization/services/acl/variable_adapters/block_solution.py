from .variable_adapter import VariableAdapter
from blocksnet.enums import LandUse


class BlockSolution(VariableAdapter):
    """
    Class for adapting a solution vector specifically for block-level data in the city.
    """

    def __init__(
        self,
        blocks_lu: Dict[int, LandUse],
        service_types: Optional[Set[ServiceType]] = None
    ):
        super().__init__()
        pass
