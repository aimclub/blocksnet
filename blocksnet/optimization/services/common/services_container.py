from pydantic import BaseModel, Field, InstanceOf
import pandas as pd


class ServicesContainer(BaseModel):

    name: str
    weight: float = Field(ge=0, le=1)
    services_df: InstanceOf[pd.DataFrame]
