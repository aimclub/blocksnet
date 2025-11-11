import os
from pathlib import Path
from typing import Iterable, Union, List
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

from shapely import Point, unary_union
from sklearn.preprocessing import RobustScaler
from blocksnet.enums import LandUseCategory, LandUse
from blocksnet.machine_learning.context import BaseContext
from blocksnet.machine_learning.strategy import BaseStrategy
from ._strategy import get_default_strategy
from ._strategy import ARTIFACTS_DIRECTORY as DEFAULT_ARTIFACTS_DIR
from .schemas import BlocksInputSchema
from .preprocessing import DataProcessor


# land_use (str) → LandUse
def str_to_land_use(lu_str: str) -> LandUse:
    """
    Convert a string to a LandUse enum value.
    
    This function takes a string representing a land use type and converts it to the corresponding
    LandUse enum value. If the input is already a LandUse enum value, it is returned unchanged.
    
    Args:
        lu_str (str): The string to be converted to a LandUse enum value.
        
    Returns:
        LandUse: The corresponding LandUse enum value.
    """
    return LandUse(lu_str.lower()) if isinstance(lu_str, str) else lu_str

# LandUse → LandUseCategory
def land_use_to_category(lu: LandUse) -> LandUseCategory | None:
    return LandUseCategory.from_land_use(lu)

def category_to_index(val) -> int | None:
    """Convert category (enum or str) to index (case-insensitive for strings)."""
    if isinstance(val, LandUseCategory):
        return CATEGORY_TO_INDEX.get(val)
    if isinstance(val, str):
        s = val.strip()
        enum_val = None
        try:
            enum_val = LandUseCategory(s)
        except Exception:
            try:
                enum_val = next(v for v in LandUseCategory if v.value.lower() == s.lower())
            except Exception:
                try:
                    enum_val = LandUseCategory[s.upper()]
                except Exception:
                    enum_val = None
        return CATEGORY_TO_INDEX.get(enum_val) if enum_val is not None else None
    return None


CATEGORY_TO_INDEX = {cat: i for i, cat in enumerate(LandUseCategory)}
INDEX_TO_CATEGORY = {i: cat for cat, i in CATEGORY_TO_INDEX.items()}

PREFERRED_FEATURE_ORDER = [
    "area",
    "perimeter",
    "compactness",
    "solidity",
    "bbox_width",
    "elongation",
    "mrr_height",
    "mrr_area",
    "mrr_aspect_ratio",
    "rectangularity_index",
    "shape_index",
    "fractal_dimension",
    "median_dist",
    "mean_k3_dist",
    "deg_10m",
    "clust_10m",
    "avg_n_deg_10m",
    "component_id_10m",
    "component_size_10m",
    "pagerank_10m",
    "h3_density",
    "h3_mean_area",
    "h3_entropy",
]


class SpatialClassifier(BaseContext):
    def __init__(
        self,
        strategy: BaseStrategy,
        buffer_distance: float = 1000,
        k_neighbors: int = 5,
    ):
        """
        A classifier that takes into account spatial characteristics.

        Args:
            strategy (BaseStrategy): The strategy to use for classification
            buffer_distance (float, optional): Distance for buffer analysis. Defaults to 1000.
            k_neighbors (int, optional): Number of neighbors to consider. Defaults to 5.
        """
        super().__init__(strategy=strategy)

        self.data_processor = DataProcessor(buffer_distance=buffer_distance, k_neighbors=k_neighbors)
        self.feature_cols: list[str] | None = None
        self.target_col = 'target_label'
        self.is_fitted = False
        self.class_names_: list[LandUseCategory] | None = None

        # context for inference (if need to store train processing)
        self.processed_train_for_context: gpd.GeoDataFrame | None = None
        # keep normalized training gdf to compute nearby_* against known zones
        self.known_gdf_for_rec_zones: gpd.GeoDataFrame | None = None
        self._last_normalized_train: gpd.GeoDataFrame | None = None
        self.scaler: RobustScaler | None = None

    # ---------- auxiliary input normalization methods ----------

    @staticmethod
    def _stack_city_list(city_gdfs: Iterable[gpd.GeoDataFrame],
                         *,
                         start: int = 0,
                         name_fmt: str = "{:03d}") -> gpd.GeoDataFrame:
        """
        Merges a list of GeoDataFrames, adding 'city' column with numbering like '000','001',...

        Args:
            city_gdfs (Iterable[gpd.GeoDataFrame]): List of GeoDataFrames to merge
            start (int, optional): Starting number for city naming. Defaults to 0.
            name_fmt (str, optional): Format string for city names. Defaults to "{:03d}".

        Returns:
            gpd.GeoDataFrame: Merged GeoDataFrame with city numbering

        Raises:
            TypeError: If any element in the list is not a GeoDataFrame
        """
        city_gdfs = list(city_gdfs)
        if not city_gdfs:
            return gpd.GeoDataFrame()

        # common set of columns
        all_cols: set[str] = set()
        for g in city_gdfs:
            if not isinstance(g, gpd.GeoDataFrame):
                raise TypeError("All elements in the list must be GeoDataFrame")
            cols = set(g.columns)
            cols.add("city")
            all_cols |= cols

        ref_crs = city_gdfs[0].crs
        parts: list[gpd.GeoDataFrame] = []
        name_counter = start
        for g in city_gdfs:
            gi = g.copy()
            if ref_crs is not None and gi.crs not in (None, ref_crs):
                gi = gi.to_crs(ref_crs)

            if "city" not in gi.columns:
                gi["city"] = name_fmt.format(name_counter)
                name_counter += 1
            else:
                missing_mask = gi["city"].isna()
                if missing_mask.any():
                    gi.loc[missing_mask, "city"] = name_fmt.format(name_counter)
                    name_counter += 1

            gi = gi.reindex(columns=sorted(all_cols))
            parts.append(gi)
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=ref_crs)

    @staticmethod
    def _ensure_city_and_center(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        After validation: ensures presence of 'city' and 'city_center' columns.

        Args:
            df (gpd.GeoDataFrame): Input GeoDataFrame

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with guaranteed city and city_center columns

        Raises:
            ValueError: If unable to compute city_center for any city
        """
        out = df.copy()

        # city
        if "city" not in out.columns:
            out["city"] = "__one__"

        # city_center
        if ("city_center" not in out.columns) or out["city_center"].isna().any():
            centers: dict[str, Point | None] = {}
            for city, geom_s in out.groupby("city")["geometry"]:
                geom_valid = geom_s[geom_s.notna() & ~geom_s.is_empty]
                if geom_valid.empty:
                    centers[city] = None
                    continue
                try:
                    u = unary_union(geom_valid.values)
                    c = u.centroid
                    if not isinstance(c, Point) or c.is_empty:
                        pts = geom_valid.centroid
                        c = Point(float(pts.x.mean()), float(pts.y.mean()))
                except Exception:
                    pts = geom_valid.centroid
                    c = Point(float(pts.x.mean()), float(pts.y.mean()))
                centers[city] = c

            if "city_center" in out.columns:
                miss = out["city_center"].isna()
                out.loc[miss, "city_center"] = out.loc[miss, "city"].map(centers)
            else:
                out["city_center"] = out["city"].map(centers)

            if out["city_center"].isna().any():
                bad = out.loc[out["city_center"].isna(), "city"].unique().tolist()
                raise ValueError(f"Failed to compute 'city_center' for cities: {bad}. Check geometry.")
        return out

    def _select_feature_columns(self, df: gpd.GeoDataFrame) -> list[str]:
        """
        Determine feature columns (numeric only)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {'category', 'category_true', 'city', 'city_center', self.target_col}
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if not numeric_cols:
            raise ValueError("No numeric feature columns available after preprocessing.")

        preferred = [col for col in PREFERRED_FEATURE_ORDER if col in numeric_cols]
        return preferred

        remaining = [col for col in numeric_cols if col not in preferred]
        return preferred + sorted(remaining)

    def _ensure_scaler(self, fit_scaler: bool, feature_frame: pd.DataFrame) -> None:
        """
        Fit scaler when required and make sure feature columns are scaled.
        """
        if self.feature_cols is None:
            raise ValueError("Feature columns are not initialized.")

        if fit_scaler or self.scaler is None:
            self.scaler = RobustScaler()
            self.scaler.fit(feature_frame[self.feature_cols])
        if self.scaler is None:
            raise ValueError("Scaler is not initialized. Train the classifier or load a scaler state.")

        transformed = self.scaler.transform(feature_frame[self.feature_cols])
        feature_frame.loc[:, self.feature_cols] = transformed

    def preprocess_data(
        self,
        data: gpd.GeoDataFrame | list | tuple,
        *,
        fit_scaler: bool = False,
        require_strict_categories: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Unified preprocessing pipeline for both training and inference data.

        Args:
            data: GeoDataFrame or list/tuple of GeoDataFrames.
            fit_scaler: When True, fit RobustScaler on the resulting features (training mode).
            require_strict_categories: Whether to enforce category presence.

        Returns:
            GeoDataFrame with engineered (and scaled) features.
        """
        normalized = self._normalize_input(data)
        if fit_scaler:
            self._last_normalized_train = normalized.copy()
            self.known_gdf_for_rec_zones = normalized.copy()
            reference = normalized
        else:
            reference = self.known_gdf_for_rec_zones
            if reference is None:
                reference = self._last_normalized_train
            if reference is None:
                reference = normalized

        processed = self.data_processor.prepare_data(
            normalized,
            known_gdf_for_rec_zones=reference,
            require_strict_categories=require_strict_categories,
        )

        processed_df = processed.copy()
        if 'category' not in processed_df and 'category' in normalized.columns:
            processed_df['category'] = normalized['category'].values
        if 'city' in normalized.columns:
            processed_df['city'] = normalized['city'].values
        if 'city_center' in normalized.columns:
            processed_df['city_center'] = normalized['city_center'].values
        processed_df['geometry'] = normalized.geometry.values

        processed_gdf = gpd.GeoDataFrame(processed_df, geometry='geometry', crs=normalized.crs)

        if fit_scaler or self.feature_cols is None:
            self.feature_cols = self._select_feature_columns(processed_gdf)
        if self.feature_cols is None:
            raise ValueError("Feature columns are not initialized. Train the classifier first.")

        for feature in self.feature_cols:
            if feature not in processed_gdf.columns:
                processed_gdf[feature] = 0.0

        self._ensure_scaler(fit_scaler, processed_gdf)

        meta_cols = [c for c in ['geometry', 'category', 'category_true', 'city', 'city_center'] if c in processed_gdf.columns]
        other_cols = [c for c in processed_gdf.columns if c not in self.feature_cols + meta_cols]
        ordered_cols = self.feature_cols + other_cols + meta_cols
        processed_gdf = processed_gdf[ordered_cols]

        return processed_gdf

    def _normalize_input(self, data: gpd.GeoDataFrame | list | tuple) -> gpd.GeoDataFrame:
        """
        1) Validates input(s) using BlocksInputSchema (without additional checks).
        2) Adds/guarantees 'city' and 'city_center' columns.

        Args:
            data (gpd.GeoDataFrame | list | tuple): Input data as GeoDataFrame or list/tuple of GeoDataFrames

        Returns:
            gpd.GeoDataFrame: Normalized GeoDataFrame

        Raises:
            TypeError: If input is not GeoDataFrame, list[GeoDataFrame] or tuple[GeoDataFrame]
        """
        if isinstance(data, (list, tuple)):
            validated = [BlocksInputSchema(g) for g in data]
            merged = self._stack_city_list(validated, start=0, name_fmt="{:03d}")
            return self._ensure_city_and_center(merged)
        elif isinstance(data, gpd.GeoDataFrame):
            g = BlocksInputSchema(data)          # validation
            return self._ensure_city_and_center(g)
        else:
            raise TypeError("Expecting GeoDataFrame, list[GeoDataFrame] or tuple[GeoDataFrame].")



    # ---------------------- train / predict ----------------------

    def preprocess_training_data(
        self,
        train_gdf: gpd.GeoDataFrame | list | tuple,
        *,
        require_strict_categories: bool = True,
        **_,
    ) -> gpd.GeoDataFrame:
        """
        Backward-compatible wrapper around ``preprocess_data`` for training datasets.
        """
        return self.preprocess_data(
            train_gdf,
            fit_scaler=True,
            require_strict_categories=require_strict_categories,
        )

    def preprocess_run_data(
        self,
        gdf: gpd.GeoDataFrame | list | tuple,
        *,
        require_strict_categories: bool = False,
        **_,
    ) -> gpd.GeoDataFrame:
        """
        Backward-compatible wrapper around ``preprocess_data`` for inference datasets.
        """
        return self.preprocess_data(
            gdf,
            fit_scaler=False,
            require_strict_categories=require_strict_categories,
        )

    def train(
        self,
        train_gdf: gpd.GeoDataFrame | list | tuple,
    ) -> float:
        """
        Train the spatial classifier using all available labeled data.

        Args:
            train_gdf: Raw training data (GeoDataFrame or list of GeoDataFrames).

        Returns:
            float: Training score computed on the same dataset (for reference).
        """
        # processed_train = self.preprocess_data(
        #     train_gdf,
        #     fit_scaler=True,
        #     require_strict_categories=True,
        # )

        self.processed_train_for_context = train_gdf.copy()

        if 'category' not in train_gdf.columns:
            raise ValueError("Training data must contain the 'category' column.")

        y_series = train_gdf['category'].map(category_to_index)
        if y_series.isnull().any():
            missing = train_gdf.loc[y_series.isnull(), 'category'].unique()
            raise ValueError(f"Unknown categories encountered during training: {missing}")

        X = train_gdf[self.feature_cols].values
        y = y_series.to_numpy(dtype=int)

        score = self.strategy.train(X, y, X, y)
        self.is_fitted = True
        self.classes_ = list(self.strategy.model.classes_)
        self.class_names_ = [INDEX_TO_CATEGORY[i] for i in self.classes_]
        return score

    def _prepare_for_prediction(
        self,
        new_data: gpd.GeoDataFrame | list | tuple,
        *,
        require_strict_categories: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Common input normalizer + preprocessing for inference.

        Args:
            new_data (gpd.GeoDataFrame | list | tuple): New data to prepare
            require_strict_categories (bool, optional): Whether category validation is strict.

        Returns:
            gpd.GeoDataFrame: Processed data ready for prediction
        """
        if isinstance(new_data, gpd.GeoDataFrame):
            if self.feature_cols and all(col in new_data.columns for col in self.feature_cols):
                return new_data.copy()
        return self.preprocess_run_data(new_data, require_strict_categories=require_strict_categories)

    def _as_list(self, gdf_or_list) -> tuple[list[gpd.GeoDataFrame], bool]:
        """
        Converts input to list of GDFs. Returns (list, was_list).

        Args:
            gdf_or_list: Input data as GDF or list/tuple of GDFs

        Returns:
            tuple[list[gpd.GeoDataFrame], bool]: List of GDFs and whether input was a list

        Raises:
            TypeError: If input is not GeoDataFrame or list/tuple of GeoDataFrame
        """
        if isinstance(gdf_or_list, (list, tuple)):
            return list(gdf_or_list), True
        if isinstance(gdf_or_list, gpd.GeoDataFrame):
            return [gdf_or_list], False
        raise TypeError("Expected GeoDataFrame or list/tuple of GeoDataFrame")

    def predict(self, new_gdf: Union[gpd.GeoDataFrame, list, tuple]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Makes predictions for new data.

        Args:
            new_gdf (Union[gpd.GeoDataFrame, list, tuple]): New data for prediction

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Predictions as array or list of arrays
        """
        items, was_list = self._as_list(new_gdf)
        out: List[np.ndarray] = []

        for g in items:
            processed_new = self._prepare_for_prediction(g, require_strict_categories=False)
            X = processed_new[self.feature_cols].values
            out.append(self.strategy.predict(X))

        return out if was_list else out[0]

    def predict_proba(self, new_gdf: Union[gpd.GeoDataFrame, list, tuple]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Makes probability predictions for new data.

        Args:
            new_gdf (Union[gpd.GeoDataFrame, list, tuple]): New data for prediction

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Probability predictions as array or list of arrays
        """
        items, was_list = self._as_list(new_gdf)
        out: List[np.ndarray] = []

        for g in items:
            processed_new = self._prepare_for_prediction(g, require_strict_categories=False)
            X = processed_new[self.feature_cols].values
            out.append(self.strategy.predict_proba(X))

        return out if was_list else out[0]

    def run(self, gdf: Union[gpd.GeoDataFrame, list, tuple]) -> Union[gpd.GeoDataFrame, List[gpd.GeoDataFrame]]:
        """
        Runs complete prediction pipeline including class names and probabilities.

        Args:
            gdf (Union[gpd.GeoDataFrame, list, tuple]): Input data

        Returns:
            Union[gpd.GeoDataFrame, List[gpd.GeoDataFrame]]: Results as GDF or list of GDFs
        """
        items, was_list = self._as_list(gdf)
        results: List[gpd.GeoDataFrame] = []

        # ensure class order for prob columns
        if self.class_names_ is None:
            self.classes_ = list(self.strategy.model.classes_)
            # optional guard: ensure current model is trained with our category set size
            if len(self.classes_) != len(LandUseCategory) - 1:
                raise ValueError(
                    "Loaded model classes do not match current LandUseCategory set. "
                    "Please retrain the model with updated categories."
                )
            self.class_names_ = [INDEX_TO_CATEGORY[c] for c in self.classes_]

        for g in items:
            processed_new = self._prepare_for_prediction(g, require_strict_categories=False)
            X = processed_new[self.feature_cols].values
            preds = self.strategy.predict(X)
            probs = self.strategy.predict_proba(X)  # shape: (n, n_classes) in model.classes_ order

            if not isinstance(processed_new, gpd.GeoDataFrame):
                gdf_out = gpd.GeoDataFrame(processed_new.copy(), geometry='geometry')
            else:
                gdf_out = processed_new.copy()

            # labels and names
            gdf_out['pred_class'] = preds
            gdf_out['pred_name'] = [INDEX_TO_CATEGORY[c].value for c in preds]

            # probabilities - strictly in self.classes_/self.class_names_ order
            for j, cls in enumerate(self.class_names_):
                gdf_out[f'prob_{cls.value}'] = probs[:, j] if probs.size else np.array([], dtype=float)

            prob_cols = [f'prob_{cls.value}' for cls in self.class_names_]
            base_cols = ['geometry']
            if 'category' in gdf_out.columns:
                base_cols.append('category')
            base_cols.append('pred_name')
            results.append(gdf_out[base_cols + prob_cols])

        return results if was_list else results[0]

    def save_scaler(self, path: str | Path) -> None:
        """
        Persist the fitted scaler and feature list to disk.
        """
        if self.scaler is None or not self.feature_cols:
            raise ValueError("Scaler is not initialized. Train the classifier before saving.")
        state = {
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
        }
        joblib.dump(state, str(path))

    def load_scaler(self, path: str | Path) -> None:
        """
        Load a previously saved scaler + feature column order.
        """
        state = joblib.load(str(path))
        self.scaler = state.get("scaler")
        self.feature_cols = state.get("feature_cols")
        if self.scaler is None or not self.feature_cols:
            raise ValueError("Loaded scaler state is incomplete.")

    @classmethod
    def default(cls) -> "SpatialClassifier":
        """
        Creates a default instance of SpatialClassifier.

        Returns:
            SpatialClassifier: Default classifier instance
        """
        inst = cls(get_default_strategy())
        try:
            p = Path(DEFAULT_ARTIFACTS_DIR) / "scaler.joblib"
            if p.exists():
                inst.load_scaler(p)
        except Exception:
            pass
        return inst

    def save_mistakes(self, test_gdf: gpd.GeoDataFrame, 
                      predictions: np.ndarray,
                      filename: str) -> None:
        """
        Saves prediction mistakes to a GeoJSON file.

        Args:
            test_gdf (gpd.GeoDataFrame): Test data with true labels
            predictions (np.ndarray): Model predictions
            filename (str): Output file path
        """
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
    # ---------------------- persistence ----------------------

    def save(self, path: str | Path) -> None:
        """
        Save strategy (model) and scaler state in the same directory.

        - strategy artifacts are managed by the strategy itself
        - scaler state (scaler + feature_cols) is saved via joblib in scaler.joblib
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save strategy artifacts
        self.strategy.save(str(path))
        # Save scaler state
        scaler_path = path / "scaler.joblib"
        self.save_scaler(scaler_path)

    def load(self, path: str | Path) -> None:
        """
        Load strategy (model) and scaler state from the same directory.
        """
        path = Path(path)
        # Load strategy artifacts
        self.strategy.load(str(path))
        # Load scaler state if present
        scaler_path = path / "scaler.joblib"
        if scaler_path.exists():
            self.load_scaler(scaler_path)
    def save_predictions_to_geojson(self, gdf: gpd.GeoDataFrame, 
                                    predictions: np.ndarray,
                                    probabilities: np.ndarray,
                                    filename: str) -> None:
        """
        Saves predictions with probabilities to a GeoJSON file.

        Args:
            gdf (gpd.GeoDataFrame): Input data
            predictions (np.ndarray): Model predictions
            probabilities (np.ndarray): Prediction probabilities
            filename (str): Output file path
        """
        result = gdf.copy()
        result['pred_class'] = predictions
        result['pred_category'] = [INDEX_TO_CATEGORY[i].value for i in predictions]
        
        for i, cls in enumerate(self.class_names_ or []):
            result[f'prob_{cls.value}'] = probabilities[:, i]
        
        self._save_geojson(result.round(4), filename)

    def _save_geojson(self, gdf: gpd.GeoDataFrame, filename: str) -> None:
        """
        Internal method to save GeoDataFrame to GeoJSON.

        Args:
            gdf (gpd.GeoDataFrame): Data to save
            filename (str): Output file path

        Raises:
            RuntimeError: If saving fails
        """
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
            raise RuntimeError(f"Error saving {filename}: {str(e)}") from e


