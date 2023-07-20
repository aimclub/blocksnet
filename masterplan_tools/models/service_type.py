from pydantic import BaseModel, Field


class ServiceType(BaseModel):
    name: str = Field(min_length=1)
    """name of service type as unique identifier"""
    demand: int = Field(default=0, ge=0)
    """demand per 1000 population"""
    accessibility: int = Field(default=0, ge=0)
    """service accessibility in minutes"""
