import pandas as pd
from pydantic import BaseModel, Field


class Variable(BaseModel):

    """Variable class.

    """
    block_id: int
    service_type: str
    capacity: int = Field(ge=0)
    site_area: float = Field(ge=0)
    build_floor_area: float = Field(ge=0)
    count: int = Field(ge=0, default=0)

    @property
    def total_capacity(self):
        """Total capacity.

        """
        return self.count * self.capacity

    @property
    def total_site_area(self):
        """Total site area.

        """
        return self.count * self.site_area

    @property
    def total_build_floor_area(self):
        """Total build floor area.

        """
        return self.count * self.build_floor_area

    def to_dict(self) -> dict:
        """To dict.

        Returns
        -------
        dict
            Description.

        """
        return {
            **self.model_dump(),
            "total_capacity": self.total_capacity,
            "total_site_area": self.total_site_area,
            "total_build_floor_area": self.total_build_floor_area,
        }
