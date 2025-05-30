from pydantic import BaseModel, Field
import pandas as pd


class Variable(BaseModel):

    block_id: int
    service_type: str
    capacity: int = Field(ge=0)
    site_area: float = Field(ge=0)
    build_floor_area: float = Field(ge=0)
    count: int = Field(ge=0, default=0)

    @property
    def total_capacity(self):
        return self.count * self.capacity

    @property
    def total_site_area(self):
        return self.count * self.site_area

    @property
    def total_build_floor_area(self):
        return self.count * self.build_floor_area

    def to_dict(self) -> dict:
        return {
            **self.model_dump(),
            "total_capacity": self.total_capacity,
            "total_site_area": self.total_site_area,
            "total_build_floor_area": self.total_build_floor_area,
        }
