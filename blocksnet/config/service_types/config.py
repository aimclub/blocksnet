from functools import singledispatchmethod
import pandas as pd
from .common import SERVICE_TYPES, UNITS
from .schemas import ServiceTypesSchema, UnitsSchema, LandUseSchema
from ...enums import LandUse


class ServiceTypesConfig:
    """Access service type metadata and land-use mappings.

    Parameters
    ----------
    service_types : pandas.DataFrame
        Tabular description of service types validated by
        :class:`ServiceTypesSchema`.
    units : pandas.DataFrame
        DataFrame of service units validated by :class:`UnitsSchema`.
    land_use : pandas.DataFrame
        Mapping between land-use categories and service availability validated
        by :class:`LandUseSchema`.
    """

    def __init__(self, service_types: pd.DataFrame, units: pd.DataFrame, land_use: pd.DataFrame):
        self.service_types = ServiceTypesSchema(service_types)
        self.units = UnitsSchema(units)
        self.land_use = LandUseSchema(land_use)

    @singledispatchmethod
    def __getitem__(self, arg):
        """Retrieve configuration data using a dynamic key type.

        Parameters
        ----------
        arg : object
            Either a service type name or a :class:`blocksnet.enums.LandUse`
            instance.

        Returns
        -------
        dict or list of str
            Detailed service type information when *arg* is a ``str`` or a
            list of service type identifiers when *arg* is a land-use value.

        Raises
        ------
        NotImplementedError
            If the provided key type is not supported.
        """

        raise NotImplementedError(f"Cant access object with such argument type : {type(arg)}")

    @__getitem__.register(str)
    def _(self, service_type: str) -> dict:
        """Return metadata for a single service type.

        Parameters
        ----------
        service_type : str
            Identifier of the service type.

        Returns
        -------
        dict
            Column values describing the requested service type.
        """

        result = self.service_types.loc[service_type].to_dict()
        return result

    @__getitem__.register(LandUse)
    def _(self, land_use: LandUse) -> list[str]:
        """List service types compatible with a land-use category.

        Parameters
        ----------
        land_use : LandUse
            Land-use category to inspect.

        Returns
        -------
        list of str
            Service type identifiers applicable to the land-use.
        """

        row = self.land_use[land_use]
        service_types = row[row].index
        return list(service_types)

    def __iter__(self):
        """Iterate over configured service type identifiers.

        Returns
        -------
        iterator of str
            Iterator over the service type names.
        """

        return iter(self.service_types.index)

    def get_service_type(self, service_type: str) -> dict:
        """Fetch service type metadata as a dictionary.

        Parameters
        ----------
        service_type : str
            Identifier of the service type.

        Returns
        -------
        dict
            Column values describing the requested service type.
        """

        row = self.service_types.loc[service_type]
        return row.to_dict()


service_types_config = ServiceTypesConfig(SERVICE_TYPES, UNITS, SERVICE_TYPES)
