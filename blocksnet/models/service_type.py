import math

from pydantic import BaseModel, Field, field_validator

from .land_use import LandUse


class ServiceBrick(BaseModel):
    """
    Represents a typical service module or unit used in urban development.

    Parameters
    ----------
    capacity : int
        Maximum number of people that can be accommodated or served by this service module.
    area : float
        Area occupied by the service module in square meters.
    is_integrated : bool
        Indicates whether the service module is integrated within a living building.
    parking_area : float
        Required parking area in square meters for the service module.
    """

    capacity: int
    area: float
    is_integrated: bool
    parking_area: float


class ServiceType(BaseModel):
    """
    Represents a category of service entity, such as schools, and its parameters.

    Parameters
    ----------
    code : str
        Code identifying the service type.
    name : str
        Unique name of the service type.
    accessibility : int, optional
        Average accessibility time in minutes to reach the service unit. Must be equal or greater than 0.
    demand : int, optional
        Demand per 1000 population for the service type. Must be equal or greater than 0.
    land_use : list[LandUse], optional
        List of land use categories associated with the service type. Default is an empty list.
    bricks : list[ServiceBrick], optional
        List of service bricks or modules available for the service type. Default is an empty list.
    """

    code: str
    name: str
    accessibility: int = Field(ge=0)
    demand: int = Field(ge=0)
    land_use: list[LandUse] = []
    bricks: list[ServiceBrick] = []

    def get_bricks(self, is_integrated: bool = False) -> list[ServiceBrick]:
        """
        Retrieve service bricks or modules associated with the service type.

        Parameters
        ----------
        is_integrated : bool, optional
            Filter by whether the brick may be integrated within a living building. Default is False.

        Returns
        -------
        list[ServiceBrick]
            List of ServiceBrick objects matching the filter criteria.
        """
        filtered_bricks = filter(lambda b: b.is_integrated == is_integrated, self.bricks)
        return list(filtered_bricks)

    @field_validator("bricks", mode="before")
    def validate_bricks(value: list) -> list[ServiceBrick]:
        """
        Validate and convert the `bricks` field into a list of `ServiceBrick` objects.

        Parameters
        ----------
        value : list
            List of dictionaries or `ServiceBrick` objects.

        Returns
        -------
        list[ServiceBrick]
            List of validated `ServiceBrick` objects.
        """
        bricks = [sb if isinstance(sb, ServiceBrick) else ServiceBrick(**sb) for sb in value]
        return bricks

    @field_validator("land_use", mode="before")
    def validate_land_use(value: list) -> list[LandUse]:
        """
        Validate and convert the `land_use` field into a list of `LandUse` objects.

        Parameters
        ----------
        value : list
            List of strings or LandUse objects.

        Returns
        -------
        list[LandUse]
            List of validated LandUse objects.
        """
        land_uses = [lu if isinstance(lu, LandUse) else LandUse[lu.upper()] for lu in value]
        return land_uses

    def calculate_in_need(self, population: int) -> int:
        """
        Calculate the estimated number of people in need of this service type based on population.

        Parameters
        ----------
        population : int
            Total population.

        Returns
        -------
        int
            Estimated number of people in need of this service type.
        """
        return math.ceil(population / 1000 * self.demand)

    def __hash__(self) -> int:
        """
        Compute the hash value of the ServiceType object based on its name.

        Returns
        -------
        int
            Hash value of the ServiceType object.
        """
        return hash(self.name)

    def __str__(self) -> str:
        """
        Return a string representation of the ServiceType object.

        Returns
        -------
        str
            String containing basic information about the ServiceType object.
        """
        accessibility = f"{self.accessibility} min"
        demand = f"{self.demand}/1000 population"
        return f"{self.code.ljust(10)} {self.name.ljust(20)} {accessibility.ljust(10)} {demand.ljust(20)}"

    def to_dict(self):
        """
        Convert the ServiceType object into a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the ServiceType object.
        """
        return self.model_dump()
