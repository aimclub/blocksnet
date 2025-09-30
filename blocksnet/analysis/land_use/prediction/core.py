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
from ._strategy import strategy
from .schemas import BlocksInputSchema
from .preprocessing import DataProcessor


# land_use (str) → LandUse
def str_to_land_use(lu_str: str) -> LandUse:
    """Convert a string label to a :class:`LandUse` enum.

    Parameters
    ----------
    lu_str : str | LandUse
        Land-use label or existing enum member.

    Returns
    -------
    LandUse
        Normalised land-use enum value.
    """
    return LandUse(lu_str.lower()) if isinstance(lu_str, str) else lu_str

# LandUse → LandUseCategory
def land_use_to_category(lu: LandUse) -> LandUseCategory | None:
    """Map a land-use type to its broader :class:`LandUseCategory`.

    Parameters
    ----------
    lu : LandUse
        Land-use enum to convert.

    Returns
    -------
    LandUseCategory or None
        Category corresponding to the land-use, or ``None`` when undefined.
    """
    return LandUseCategory.from_land_use(lu)


def category_to_index(val) -> int | None:
    """Convert category label or enum into its numeric index.

    Parameters
    ----------
    val : LandUseCategory | str
        Category value to map to the classifier index.

    Returns
    -------
    int or None
        Index matching :data:`CATEGORY_TO_INDEX`, or ``None`` if conversion
        fails.
    """
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
    """Predict land-use categories using spatial features and context.

    Parameters
    ----------
    strategy : blocksnet.machine_learning.strategy.BaseStrategy
        Strategy responsible for fitting and inference.
    buffer_distance : float, optional
        Buffer distance used when engineering neighbor-based features.
        Defaults to ``1000``.
    k_neighbors : int, optional
        Number of neighbours considered during feature aggregation. Defaults
        to ``5``.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        buffer_distance: float = 1000,
        k_neighbors: int = 5,
    ):
        """Initialise the spatial classifier with preprocessing context."""
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
                                                'nbr_mean_asymmetry_y_log']
        self.target_col = 'target_label'
        self.is_fitted = False
        self.class_names_: list[LandUseCategory] | None = None

        # context for inference (if need to store train processing)
        self.processed_train_for_context: gpd.GeoDataFrame | None = None

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
        random_state: int = 42
    ) -> pd.Series:
        """Split labelled data into train and validation subsets per city.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input dataframe containing city and target columns.
        target_col : str, optional
            Name of the column storing target categories. Defaults to
            ``"category"``.
        test_size : float, optional
            Proportion of labelled observations assigned to validation within
            each city. Defaults to ``0.2``.
        random_state : int, optional
            Seed controlling the random split. Defaults to ``42``.

        Returns
        -------
        pandas.Series
            Series indexed by ``gdf`` with values ``"L"`` (train), ``"U"``
            (validation), or ``"X"`` (unlabelled).
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

    def train(self, train_gdf: gpd.GeoDataFrame | list | tuple) -> None:
        """Fit the classifier using city-specific labelled and unlabelled data.

        Parameters
        ----------
        train_gdf : geopandas.GeoDataFrame or list or tuple
            Training data in a single dataframe or a collection of dataframes.

        Returns
        -------
        float
            Score reported by the strategy on the validation subset.

        Raises
        ------
        ValueError
            If category values cannot be mapped to classifier indices.
        TypeError
            If the input type is unsupported.
        """
        # 1) input validation and normalization (city + city_center)
        train_gdf = self._normalize_input(train_gdf)

        # 2) L/U labeling by cities
        assign = self.split_l_u_per_city(train_gdf, target_col='category', test_size=0.2, random_state=42)

        # 3) preprocessing
        processed_train = self.data_processor.prepare_data(train_gdf)
        self.processed_train_for_context = processed_train

        excluded_columns = ['geometry', 'category', 'city', 'city_center']
        excluded_columns += getattr(self.data_processor, 'columns_to_log', [])
        self.feature_cols = [c for c in processed_train.columns if c not in excluded_columns]

        processed_train[self.target_col] = processed_train["category"].map(category_to_index)
        if processed_train[self.target_col].isnull().any():
            raise ValueError("Some category values could not be mapped to index.")

        # align assign and processed_train
        assign = assign.reindex(processed_train.index)
        is_L = (assign == 'L')
        is_U = (assign == 'U')

        X_L = processed_train.loc[is_L, self.feature_cols].values
        y_L = processed_train.loc[is_L, self.target_col].values

        if is_U.any():
            X_U = processed_train.loc[is_U, self.feature_cols].values
            y_U = processed_train.loc[is_U, self.target_col].values
        else:
            X_U = np.empty((0, len(self.feature_cols)), dtype=float)
            y_U = np.empty((0,), dtype=int)

        score = self.strategy.train(X_L, y_L, X_U, y_U)
        unique_labels = np.unique(np.concatenate([y_L, y_U]) if y_U.size else y_L)
        self.class_names_ = [INDEX_TO_CATEGORY[i] for i in unique_labels]
        return score

    def _prepare_for_prediction(self, new_data: gpd.GeoDataFrame | list | tuple) -> gpd.GeoDataFrame:
        """
        Common input normalizer + preprocessing for inference.

        Args:
            new_data (gpd.GeoDataFrame | list | tuple): New data to prepare

        Returns:
            gpd.GeoDataFrame: Processed data ready for prediction
        """
        # 1) validation (BlocksInputSchema) and adding city/city_center
        gdf_norm = self._normalize_input(new_data)

        # 2) preprocessing (features); can pass context if needed
        processed_new = self.data_processor.prepare_data(gdf_norm)

        # 3) add missing features to match training set
        if self.feature_cols:
            for c in self.feature_cols:
                if c not in processed_new.columns:
                    processed_new[c] = 0.0
            processed_new = processed_new[self.feature_cols + [col for col in processed_new.columns if col not in self.feature_cols]]

        return processed_new

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
        """Predict land-use categories for new geometries.

        Parameters
        ----------
        new_gdf : geopandas.GeoDataFrame or list or tuple
            Data to classify, optionally provided as multiple city datasets.

        Returns
        -------
        numpy.ndarray or list[numpy.ndarray]
            Predicted category indices aligned with :data:`INDEX_TO_CATEGORY`.
        """
        items, was_list = self._as_list(new_gdf)
        out: List[np.ndarray] = []

        for g in items:
            processed_new = self._prepare_for_prediction(g)
            X = processed_new[self.feature_cols].values
            out.append(self.strategy.predict(X))

        return out if was_list else out[0]

    def predict_proba(self, new_gdf: Union[gpd.GeoDataFrame, list, tuple]) -> Union[np.ndarray, List[np.ndarray]]:
        """Estimate class membership probabilities for new data.

        Parameters
        ----------
        new_gdf : geopandas.GeoDataFrame or list or tuple
            Data to classify, optionally provided in multiple batches.

        Returns
        -------
        numpy.ndarray or list[numpy.ndarray]
            Probability distributions matching the class order used during
            training.
        """
        items, was_list = self._as_list(new_gdf)
        out: List[np.ndarray] = []

        for g in items:
            processed_new = self._prepare_for_prediction(g)
            X = processed_new[self.feature_cols].values
            out.append(self.strategy.predict_proba(X))

        return out if was_list else out[0]

    def run(self, gdf: Union[gpd.GeoDataFrame, list, tuple]) -> Union[gpd.GeoDataFrame, List[gpd.GeoDataFrame]]:
        """Execute the full prediction pipeline with metadata restoration.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame or list or tuple
            Input data as a single dataset or list of datasets.

        Returns
        -------
        geopandas.GeoDataFrame or list[geopandas.GeoDataFrame]
            Dataframes containing original geometry, predicted labels, and
            probability columns.
        """
        items, was_list = self._as_list(gdf)
        results: List[gpd.GeoDataFrame] = []

        # ensure class order for prob columns
        if self.class_names_ is None:
            self.classes_ = list(self.strategy.model.classes_)
            self.class_names_ = [INDEX_TO_CATEGORY[c] for c in self.classes_]

        for g in items:
            processed_new = self._prepare_for_prediction(g)
            X = processed_new[self.feature_cols].values

            preds = self.strategy.predict(X)
            probs = self.strategy.predict_proba(X)  # shape: (n, n_classes) in model.classes_ order

            # restore original GeoDataFrame (with 'city', 'city_center' after normalize)
            gdf_norm = self._normalize_input(g).copy()

            # labels and names
            gdf_norm['pred_class'] = preds
            gdf_norm['pred_name'] = [INDEX_TO_CATEGORY[c].value for c in preds]

            # probabilities — strictly in self.classes_/self.class_names_ order
            for j, cls in enumerate(self.class_names_):
                gdf_norm[f'prob_{cls.value}'] = probs[:, j] if probs.size else np.array([], dtype=float)

            results.append(gdf_norm[['geometry','category','pred_name','prob_urban','prob_non_urban','prob_industrial']])

        return results if was_list else results[0]

    @classmethod
    def default(cls) -> "SpatialClassifier":
        """Create a :class:`SpatialClassifier` with the bundled strategy.

        Returns
        -------
        SpatialClassifier
            Configured classifier ready for training.
        """
        return cls(strategy)

    def save_mistakes(self, test_gdf: gpd.GeoDataFrame,
                      predictions: np.ndarray,
                      filename: str) -> None:
        """Persist misclassified observations to GeoJSON for inspection.

        Parameters
        ----------
        test_gdf : geopandas.GeoDataFrame
            Dataset containing ground-truth land-use information.
        predictions : numpy.ndarray
            Predicted class indices aligned with ``test_gdf``.
        filename : str
            Destination file path.
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
        """Persist predicted classes and probabilities to GeoJSON.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input geometries and metadata.
        predictions : numpy.ndarray
            Predicted class indices.
        probabilities : numpy.ndarray
            Probability estimates for each class.
        filename : str
            Destination file path.
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
