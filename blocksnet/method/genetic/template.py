import geopandas as gpd
import pandas as pd
from ..base_method import BaseMethod
from ...models import Block
from ..provision import Provision


class Genetic(BaseMethod):
    # у тебя здесь уже есть city_model из BaseMethod, можешь обращаться к ней через self.city_model
    # а инициализировать как gen = Genetic(city_model=city_model)

    # инит тебе переписывать не нужно (и даже нельзя) потому что это уже делает BaseModel в BaseMethod (см. pydantic)

    # это твой входной метод, юзер просто хочет его вызвать и дать туда какие-то ограничения на количество кварталов или что-то еще что ты придумаешь
    # главное максимально инкапсулировать его на уровень "я хочу обновить три любых квартала" или "вот мои три квартала обнови их" в общем на твое усмотрение:
    # gen.calculate(scenario) - это например для всех кварталов
    def calculate(
        self, scenario, selected_blocks: list[Block] = []
    ) -> gpd.GeoDataFrame:  # верни здесь какой-нибудь гдф или дф который будет содержать "постройку" сервиса и мощности
        """
        Если selected_blocks тебе не отдали, то делай оптимизацию по всему городу
        selected_blocks принимает не айдишники кварталов, а сами кварталы, их ты можешь выцепить через city_model.blocks (это прям экземпляры класса Block)
        поэтому ты из этих Block можешь выцепить всю нужную инфу (смотри class Block в models/city) там есть area и прочее
        ----
        Scenario принимай как было раньше:
        {'schools':0.5, 'kindergartens':0.5} и т.д.
        """
        city_model = self.city_model  # город у тебя уже инициализирован
        blocks = self.city_model.blocks  # можешь вот так выцепить список всех Block
        blocks[0].area  # площадь квартала на нулевом месте в списке
        city_model[0].area  # площадь квартала с id=0
        prov = Provision(
            city_model=city_model
        )  # вот твой стабильный пров через который считается обеспеченность и который уже знает свой город
        #
        #
        # как задавать апдейт по кварталам (если неудобно или будут идеи лучше, предлагай, я ориентировался на то что по стандарту мы работаем в дф/гдф)
        update = {
            148: {
                "population": 1000,
            },
            150: {"schools": 1000},
        }
        update_df = pd.DataFrame.from_dict(update, orient="index")
        # как считать провижен сценария: отдаешь ему свой scenario и дф с обновлениями, из пришедшего тупла заберешь вторую чиселку: total, её максимизируешь
        _, total = prov.calculate_scenario(scenario, update_df)

    def plot(
        gdf: gpd.GeoDataFrame,
    ):  # здесь по желанию метод, в который ты можешь отдать гдф из твоего calculate и нарисовать что-то на карте
        ...
