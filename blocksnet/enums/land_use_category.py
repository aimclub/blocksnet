from enum import Enum

# TODO: Оставить 1 вариант
class LandUseCategory(Enum):
    RECREATION = ("RECREATION", "sp_ag_rec")
    SPECIAL = ("SPECIAL", "sp_ag_rec")
    AGRICULTURE = ("AGRICULTURE", "sp_ag_rec")
    BUSINESS = ("BUSINESS", "bus_res")
    RESIDENTIAL = ("RESIDENTIAL", "bus_res")
    INDUSTRIAL = ("INDUSTRIAL", "industrial")
    TRANSPORT = ("TRANSPORT", None)

    def __str__(self):
        return self.value[1]  # Вернёт 'sp_ag_rec', и т.д.

    @property
    def tag(self):
        return self.value[1]
    
class LandUseCategory(Enum):
    SP_AG_REC = "sp_ag_rec"
    BUS_RES = "bus_res"
    INDUSTRIAL = "industrial"

