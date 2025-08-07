import os
import numpy as np
import geopandas as gpd
from pathlib import Path
from sklearn.model_selection import train_test_split

from blocksnet.enums import LandUseCategory, LandUse
from blocksnet.machine_learning.context import BaseContext
from blocksnet.machine_learning.strategy import BaseStrategy
from ._strategy import strategy
from .schemas import BlocksInputSchema
from .preprocessing import DataProcessor

# land_use (str) → LandUse
def str_to_land_use(lu_str: str) -> LandUse:
    return LandUse(lu_str.lower()) if isinstance(lu_str, str) else lu_str

# LandUse → LandUseCategory
def land_use_to_category(lu: LandUse) -> LandUseCategory | None:
    return LandUseCategory.from_land_use(lu)

def category_to_index(val) -> int | None:
    """Convert category (enum or str) to index."""
    if isinstance(val, LandUseCategory):
        return CATEGORY_TO_INDEX.get(val)
    if isinstance(val, str):
        try:
            enum_val = LandUseCategory(val.lower())
            return CATEGORY_TO_INDEX.get(enum_val)
        except ValueError:
            return None
    return None


CATEGORY_TO_INDEX = {cat: i for i, cat in enumerate(LandUseCategory)}
INDEX_TO_CATEGORY = {i: cat for cat, i in CATEGORY_TO_INDEX.items()}


class SpatialClassifier(BaseContext):
    def __init__(
        self,
        strategy: BaseStrategy,
        buffer_distance: float = 1000,
        k_neighbors: int = 5,
    ):
        """
        Классификатор с учетом пространственных характеристик.

        Args:
            estimators: список моделей (name, model)
            model_params: параметры для VotingClassifier
            buffer_distance: Расстояние для буферизации
            k_neighbors: Количество соседей для KNN
        """
        super().__init__(strategy=strategy)

        self.data_processor = DataProcessor(buffer_distance=buffer_distance, k_neighbors=k_neighbors)
        self.feature_cols = None
        self.target_col = 'target_label'
        self.is_fitted = False
        self.class_names_ = None
        self.train_gdf_for_rec_zones = None
        self.processed_train_for_context = None

    def train(self, train_gdf: gpd.GeoDataFrame,) -> None:

        train_gdf = BlocksInputSchema(train_gdf)
        self.train_gdf_for_rec_zones = train_gdf.copy()
        
        processed_train = self.data_processor.prepare_data(train_gdf, is_train=True)
        self.processed_train_for_context = processed_train
        
        excluded_columns = ['geometry', 'category', 'city', 'city_center']
        excluded_columns += self.data_processor.columns_to_log
        self.feature_cols = [c for c in processed_train.columns if c not in excluded_columns]

        processed_train["target_label"] = processed_train["category"].map(category_to_index)

        if processed_train["target_label"].isnull().any():
            raise ValueError("Некоторые значения land_use не удалось замапить в категорию.")
        
        X = processed_train[self.feature_cols].values
        y = processed_train[self.target_col].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.strategy.train(x_train, y_train, x_test, y_test)

        self.is_fitted = True
        
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        self.class_names_ = [INDEX_TO_CATEGORY[i] for i in unique_labels]

    def predict(self, new_gdf: gpd.GeoDataFrame) -> np.ndarray:
        processed_new = self._prepare_for_prediction(new_gdf)
        X = processed_new[self.feature_cols].values
        return self.strategy.predict(X)

    def predict_proba(self, new_gdf: gpd.GeoDataFrame) -> np.ndarray:
        processed_new = self._prepare_for_prediction(new_gdf)
        X = processed_new[self.feature_cols].values
        return self.strategy.predict_proba(X)
    
    def run(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        predictions = self.predict(gdf)
        probabilities = self.predict_proba(gdf)
        
        result = gdf.copy()
        result['pred_class'] = predictions
        result['pred_name'] = [INDEX_TO_CATEGORY[c].value for c in predictions]

        for i, cls in enumerate(self.class_names_):
            result[f'prob_{cls.value}'] = probabilities[:, i]

        return result

    @classmethod
    def default(cls) -> "SpatialClassifier":
        pass
        # return cls(strategy)


    def _prepare_for_prediction(self, new_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if not self.is_fitted:
            raise RuntimeError("Firstly, you need to call train() method")
            
        return self.data_processor.prepare_data(
            new_gdf,
            is_train=False,
            train_gdf_for_rec_zones=self.processed_train_for_context,
            knn_model=self.data_processor.knn_model
        )

    def save_train_data(self, filename: str) -> None:

        if not self.is_fitted:
            raise RuntimeError("Модель не обучена")
        self._save_geojson(self.train_gdf_for_rec_zones, filename)

    def save_test_data(self, test_gdf: gpd.GeoDataFrame, filename: str) -> None:

        self._save_geojson(test_gdf, filename)

    def save_mistakes(self, test_gdf: gpd.GeoDataFrame, 
                    predictions: np.ndarray,
                    filename: str) -> None:

        test_data = test_gdf.copy()
        test_data['pred_class'] = predictions
        test_data['pred_category'] = [self.class_names_[c].value for c in predictions]

        actual_category = test_data["land_use"].map(str_to_land_use).map(land_use_to_category)
        actual_index = actual_category.map(CATEGORY_TO_INDEX)
        test_data['true_category'] = actual_category
        test_data['true_category_name'] = actual_category.map(lambda c: c.value if c else "unknown")

        mistakes = test_data[actual_index != test_data['pred_class']]
        mistakes['mismatch'] = mistakes.apply(
            lambda row: f"{row['true_category_name']} -> {row['pred_category']}", axis=1
        )

        self._save_geojson(mistakes, filename)


    def save_predictions_to_geojson(self, gdf: gpd.GeoDataFrame, 
                                predictions: np.ndarray,
                                probabilities: np.ndarray,
                                filename: str) -> None:
        
        result = gdf.copy()
        result['pred_class'] = predictions
        result['pred_category'] = [INDEX_TO_CATEGORY[i].value for i in predictions]
        
        for i, cls in enumerate(self.class_names_):
            result[f'prob_{cls.value}'] = probabilities[:, i]
        
        self._save_geojson(result.round(4), filename)

    def _save_geojson(self, gdf: gpd.GeoDataFrame, filename: str) -> None:

        try:
            filepath = Path(filename)
            os.makedirs(filepath.parent, exist_ok=True)
            
            save_gdf = gdf.copy()
            geom_cols = [col for col in save_gdf.columns if save_gdf[col].dtype == 'geometry']
            
            if len(geom_cols) > 1:
                main_geom = geom_cols[0]
                for col in geom_cols[1:]:
                    save_gdf[col + '_wkt'] = save_gdf[col].to_wkt()
                    save_gdf = save_gdf.drop(columns=[col])
            
            for col in save_gdf.columns:
                if save_gdf[col].dtype == 'object' and col != save_gdf.geometry.name:
                    save_gdf[col] = save_gdf[col].astype(str)
            
            save_gdf.to_file(filename, driver='GeoJSON', encoding='utf-8')
            
        except Exception as e:
            raise f"Ошибка сохранения {filename}: {str(e)}"


