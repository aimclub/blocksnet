import pickle
import geopandas as gpd
from .schemas import BlocksSchema
from .common import BlocksOrganizer


class DataOrganizer(BlocksOrganizer):
    def __init__(self, blocks: gpd.GeoDataFrame):
        BlocksOrganizer.__init__(self, BlocksSchema(blocks))

    @staticmethod
    def from_pickle(file_path: str):
        state = None
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        return state

    def to_pickle(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
