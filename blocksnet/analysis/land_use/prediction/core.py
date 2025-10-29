import os
from pathlib import Path
from typing import Iterable, Union, List
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import Point, unary_union
from blocksnet.enums import LandUseCategory, LandUse
from blocksnet.machine_learning.context import BaseContext
from blocksnet.machine_learning.strategy import BaseStrategy
from ._strategy import get_default_strategy
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

TRAIN_ASSIGN_COLUMN = "__train_split__"


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
        self.feature_cols: list[str] | None = ['x_local',
                                                'y_local',
                                                'compactness',
                                                'fractal_dimension',
                                                'rectangularity_index',
                                                'squareness_index',
                                                'shape_index_log',
                                                'mbr_area_log',
                                                'mbr_aspect_ratio_log',
                                                'solidity_log',
                                                'asymmetry_x_log',
                                                'asymmetry_y_log',
                                                'nbr_mean_x_local',
                                                'nbr_mean_y_local',
                                                'nbr_mean_compactness',
                                                'nbr_mean_fractal_dimension',
                                                'nbr_mean_shape_index',
                                                'nbr_mean_mbr_area',
                                                'nbr_mean_rectangularity_index',
                                                'nbr_mean_mbr_aspect_ratio',
                                                'nbr_mean_squareness_index',
                                                'nbr_mean_solidity',
                                                'nbr_mean_asymmetry_x',
                                                'nbr_mean_asymmetry_y',
                                                'nbr_mean_shape_index_log',
                                                'nbr_mean_mbr_area_log',
                                                'nbr_mean_mbr_aspect_ratio_log',
                                                'nbr_mean_solidity_log',
                                                'nbr_mean_asymmetry_x_log',
                                                'nbr_mean_asymmetry_y_log',
                                                'nearby_rec_spec_agri_count',
                                                'nearby_bus_res_count',
                                                'nearby_industrial_count',
                                                'nearby_transport_count'
                                                ]
        self.target_col = 'target_label'
        self.is_fitted = False
        self.class_names_: list[LandUseCategory] | None = None

        # context for inference (if need to store train processing)
        self.processed_train_for_context: gpd.GeoDataFrame | None = None
        # keep normalized training gdf to compute nearby_* against known zones
        self.known_gdf_for_rec_zones: gpd.GeoDataFrame | None = None
        self._last_normalized_train: gpd.GeoDataFrame | None = None
        self._last_train_assignments: pd.Series | None = None
        self._train_assign_column: str = TRAIN_ASSIGN_COLUMN

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
            all_cols |= set(g.columns)
        all_cols |= {"city"}  # ensure presence

        ref_crs = city_gdfs[0].crs
        parts: list[gpd.GeoDataFrame] = []
        for i, g in enumerate(city_gdfs, start):
            gi = g.copy()
            if ref_crs is not None and gi.crs != ref_crs:
                gi = gi.to_crs(ref_crs)
            # overwrite/create 'city'
            if "city" in gi.columns:
                gi = gi.drop(columns=["city"])
            gi["city"] = name_fmt.format(i)
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
            # validate EACH GDF
            validated = [BlocksInputSchema(g) for g in data]
            merged = self._stack_city_list(validated, start=0, name_fmt="{:03d}")
            return self._ensure_city_and_center(merged)
        elif isinstance(data, gpd.GeoDataFrame):
            g = BlocksInputSchema(data)          # validation
            return self._ensure_city_and_center(g)
        else:
            raise TypeError("Expecting GeoDataFrame, list[GeoDataFrame] or tuple[GeoDataFrame].")

    # ---------------------- L/U split ----------------------

    def split_l_u_per_city(
        self,
        gdf: gpd.GeoDataFrame,
        target_col: str = 'category',
        test_size: float = 0.2,
        random_state = None
    ) -> pd.Series:
        """
        Returns pd.Series (index=gdf.index) with values:
        'L' — labeled part for training,
        'U' — hidden part for validation within city,
        'X' — initially unlabeled (if any).
        Splitting is done independently for each city.

        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame
            target_col (str, optional): Name of target column. Defaults to 'category'.
            test_size (float, optional): Proportion of data for validation. Defaults to 0.2.
            random_state (int, optional): Random seed. Defaults to 42.

        Returns:
            pd.Series: Series with split assignments
        """
        if 'city' not in gdf.columns:
            gdf = gdf.assign(city="__one__")

        assign = pd.Series('X', index=gdf.index, dtype=object)
        rng = np.random.default_rng(random_state)
        labeled_all = gdf.index[gdf[target_col].notna()]

        for _, idx in gdf.groupby('city').indices.items():
            idx = pd.Index(idx)

            labeled_idx = idx.intersection(labeled_all)
            n_lab = len(labeled_idx)
            if n_lab == 0:
                continue

            n_val = int(round(test_size * n_lab))
            n_val = max(1, min(n_val, n_lab))

            if n_lab == 1:
                val_idx = labeled_idx
                train_idx = idx.difference(val_idx)
            else:
                val_idx = pd.Index(rng.choice(labeled_idx.to_numpy(), size=n_val, replace=False))
                train_idx = labeled_idx.difference(val_idx)

            assign.loc[train_idx] = 'L'
            assign.loc[val_idx]   = 'U'

        return assign

    # ---------------------- train / predict ----------------------

    def preprocess_training_data(
        self,
        train_gdf: gpd.GeoDataFrame | list | tuple,
        *,
        target_col: str = 'category',
        test_size: float = 0.2,
        random_state: int | None = None,
        assign_column: str = TRAIN_ASSIGN_COLUMN,
    ) -> gpd.GeoDataFrame:
        """
        Run full preprocessing pipeline for training data and return a GeoDataFrame ready for training.

        The returned GeoDataFrame contains engineered features together with original geometry/category
        columns and an additional column specified by ``assign_column`` holding the L/U split labels.

        Args:
            train_gdf (gpd.GeoDataFrame | list | tuple): Raw training data.
            target_col (str, optional): Target column used for split. Defaults to 'category'.
            test_size (float, optional): Fraction for validation split. Defaults to 0.2.
            random_state (int | None, optional): RNG seed for split. Defaults to None.
            assign_column (str, optional): Column name to store L/U assignments. Defaults to TRAIN_ASSIGN_COLUMN.

        Returns:
            gpd.GeoDataFrame: Preprocessed training data with split labels stored in ``assign_column``.
        """
        normalized_train = self._normalize_input(train_gdf)
        assignments = self.split_l_u_per_city(
            normalized_train,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
        )

        processed_train = self.data_processor.prepare_data(
            normalized_train,
            known_gdf_for_rec_zones=normalized_train,
            require_strict_categories=True,
        )

        processed = processed_train.copy()
        if 'category' not in processed.columns and 'category' in normalized_train.columns:
            processed['category'] = normalized_train['category'].values
        if 'city' in normalized_train.columns:
            processed['city'] = normalized_train['city'].values
        if 'city_center' in normalized_train.columns:
            processed['city_center'] = normalized_train['city_center'].values
        processed[assign_column] = assignments.values
        processed['geometry'] = normalized_train.geometry.values

        processed_gdf = gpd.GeoDataFrame(processed, geometry='geometry', crs=normalized_train.crs)

        self._last_normalized_train = normalized_train.copy()
        self._last_train_assignments = assignments.copy()
        self._train_assign_column = assign_column

        return processed_gdf

    def preprocess_run_data(
        self,
        gdf: gpd.GeoDataFrame | list | tuple,
        *,
        require_strict_categories: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Run full preprocessing pipeline for inference data and return a GeoDataFrame ready for prediction.

        The returned GeoDataFrame contains engineered features alongside original geometry/category columns.
        Missing feature columns (w.r.t. ``self.feature_cols``) are appended with zeros to keep compatibility
        with the trained model.

        Args:
            gdf (gpd.GeoDataFrame | list | tuple): Raw inference data.
            require_strict_categories (bool, optional): Whether to enforce category presence.
                Defaults to False.

        Returns:
            gpd.GeoDataFrame: Preprocessed inference data suitable for ``predict``/``run``.
        """
        normalized = self._normalize_input(gdf)
        processed = self.data_processor.prepare_data(
            normalized,
            known_gdf_for_rec_zones=self.known_gdf_for_rec_zones,
            require_strict_categories=require_strict_categories,
        )

        processed_df = processed.copy()
        if 'category' not in processed_df.columns and 'category' in normalized.columns:
            processed_df['category'] = normalized['category'].values
        if 'city' in normalized.columns:
            processed_df['city'] = normalized['city'].values
        if 'city_center' in normalized.columns:
            processed_df['city_center'] = normalized['city_center'].values
        processed_df['geometry'] = normalized.geometry.values

        processed_gdf = gpd.GeoDataFrame(processed_df, geometry='geometry', crs=normalized.crs)

        if self.feature_cols:
            for feature in self.feature_cols:
                if feature not in processed_gdf.columns:
                    processed_gdf[feature] = 0.0
            feature_set = [c for c in self.feature_cols if c in processed_gdf.columns]
            meta_cols = [c for c in ['geometry', 'category', 'city', 'city_center'] if c in processed_gdf.columns]
            other_cols = [c for c in processed_gdf.columns if c not in feature_set + meta_cols]
            processed_gdf = processed_gdf[feature_set + other_cols + meta_cols]

        return processed_gdf

    def train(
        self,
        train_gdf: gpd.GeoDataFrame | list | tuple,
        *,
        assign_column: str | None = None,
        normalized_gdf: gpd.GeoDataFrame | None = None,
    ) -> None:
        """
        Training with city-wise L/U split:
        - Uses provided preprocessed data if available (identified by presence of the assignment column)
        - Otherwise runs preprocessing automatically.

        Args:
            train_gdf (gpd.GeoDataFrame | list | tuple): Training data
            assign_column (str | None, optional): Column name with L/U assignments when data is preprocessed.
                Defaults to ``TRAIN_ASSIGN_COLUMN``.
            normalized_gdf (gpd.GeoDataFrame | None, optional): Normalized training GeoDataFrame providing
                spatial context for inference. Required if ``train_gdf`` is already preprocessed and no prior
                call to ``preprocess_training_data`` has been made.

        Returns:
            float: Training score
        """
        assign_col = assign_column or self._train_assign_column or TRAIN_ASSIGN_COLUMN

        if isinstance(train_gdf, gpd.GeoDataFrame) and assign_col in train_gdf.columns:
            processed_train = train_gdf.copy()
            assignments = processed_train[assign_col]
            normalized_train = normalized_gdf
            if normalized_train is None:
                normalized_train = self._last_normalized_train
            if normalized_train is None:
                normalized_train = self.known_gdf_for_rec_zones
            if normalized_train is None:
                raise ValueError(
                    "Normalized training data is required. "
                    "Call preprocess_training_data first or pass normalized_gdf."
                )
        else:
            processed_train = self.preprocess_training_data(
                train_gdf,
                assign_column=assign_col,
            )
            assignments = self._last_train_assignments
            normalized_train = self._last_normalized_train
            if assignments is None or normalized_train is None:
                raise RuntimeError("Failed to obtain preprocessing context for training.")

        # keep context for inference/downstream needs
        normalized_copy = normalized_train.copy()
        self.known_gdf_for_rec_zones = normalized_copy
        self.processed_train_for_context = processed_train.copy()
        self._train_assign_column = assign_col
        self._last_normalized_train = normalized_copy

        excluded_columns = {'geometry', 'category', 'city', 'city_center', assign_col}
        excluded_columns |= set(getattr(self.data_processor, 'columns_to_log', []))
        self.feature_cols = [c for c in processed_train.columns if c not in excluded_columns]

        train_frame = processed_train.copy()
        train_frame[self.target_col] = train_frame["category"].map(category_to_index)

        # align assign and processed_train
        assignments = assignments.reindex(train_frame.index)
        self._last_train_assignments = assignments.copy()
        is_L = (assignments == 'L')
        is_U = (assignments == 'U')
        used_mask = is_L | is_U

        # Only require mapped indices for rows actually used for training/validation (L or U)
        if train_frame.loc[used_mask, self.target_col].isnull().any():
            raise ValueError("Some category values used for training/validation could not be mapped to index.")

        X_L = train_frame.loc[is_L, self.feature_cols].values
        y_L = train_frame.loc[is_L, self.target_col].values

        if is_U.any():
            X_U = train_frame.loc[is_U, self.feature_cols].values
            y_U = train_frame.loc[is_U, self.target_col].values
        else:
            X_U = np.empty((0, len(self.feature_cols)), dtype=float)
            y_U = np.empty((0,), dtype=int)

        score = self.strategy.train(X_L, y_L, X_U, y_U)
        # Ensure class_names_ strictly follows the model.classes_ order
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
            if len(self.classes_) != len(LandUseCategory):
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

    @classmethod
    def default(cls) -> "SpatialClassifier":
        """
        Creates a default instance of SpatialClassifier.

        Returns:
            SpatialClassifier: Default classifier instance
        """
        return cls(get_default_strategy())

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
