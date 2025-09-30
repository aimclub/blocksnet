import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Optional

from scipy import sparse
from shapely import make_valid
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph


class DataProcessor:
    """Create spatial features for land-use classification."""

    def __init__(self, buffer_distance: float = 1000, k_neighbors: int = 5):
        """Initialise processor with neighbourhood and buffering parameters.

        Parameters
        ----------
        buffer_distance : float, optional
            Buffer radius (in metres) applied when counting nearby zones.
            Defaults to ``1000``.
        k_neighbors : int, optional
            Number of nearest neighbours used when constructing KNN graphs.
            Defaults to ``5``.
        """
        self.buffer_distance = buffer_distance
        self.k_neighbors = k_neighbors
        self.knn_model = None
        
        # Feature lists for spatial context and logarithm transformation
        self.feature_names_for_spatial_context = [
            'mbr_area', 'solidity', 'compactness', 'shape_index', 
            'mbr_aspect_ratio', 'squareness_index', 'fractal_dimension',
            'rectangularity_index', 'nearby_bus_res_count', 
            'nearby_transport_count', 'nearby_industrial_count',
            'nearby_rec_spec_agri_count'
        ]
        
        self.columns_to_log = [
            'shape_index', 'mbr_area', 'mbr_aspect_ratio', 
            'solidity', 'asymmetry_x', 'asymmetry_y'
        ]

    def build_city_graph(self, city_gdf: gpd.GeoDataFrame,
                        mode: str = "radius",
                        radius: float = 1000.0,
                        k: int = 8) -> sparse.csr_matrix:
        """Build a sparse adjacency matrix from city polygons.

        Parameters
        ----------
        city_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing city geometries.
        mode : {"radius", "knn"}, optional
            Strategy for constructing connections. Defaults to ``"radius"``.
        radius : float, optional
            Search radius (metres) when ``mode="radius"``. Defaults to ``1000``.
        k : int, optional
            Number of neighbours when ``mode="knn"``. Defaults to ``8``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Connectivity matrix where non-zero entries denote adjacency.

        Notes
        -----
        Self-loops are removed and special cases such as empty datasets or
        ``k`` larger than the node count are handled gracefully.
        """
        centroids = city_gdf.geometry.centroid
        coords = np.c_[centroids.x.values, centroids.y.values]
        n = len(coords)

        if n == 0:
            return sparse.csr_matrix((0, 0), dtype=np.float32)
        if n == 1:
            return sparse.csr_matrix((1, 1), dtype=np.float32)

        if mode == "radius":
            A = radius_neighbors_graph(coords, radius=radius, mode="connectivity",
                                    include_self=False)
            return A.tocsr()

        k_eff = min(k, n - 1)
        if k_eff <= 0:
            return sparse.csr_matrix((n, n), dtype=np.float32)

        A = kneighbors_graph(coords, n_neighbors=k_eff, mode="connectivity",
                            include_self=False)
        return A.tocsr()

    def neighbor_geom_aggregates(self, A: sparse.csr_matrix,
                                feats_df: pd.DataFrame,
                                agg: str = "mean") -> pd.DataFrame:
        """Aggregate numeric node features across neighbourhoods.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Adjacency matrix describing neighbourhood relationships.
        feats_df : pandas.DataFrame
            Node feature dataframe containing numeric descriptors.
        agg : str, optional
            Aggregation strategy. Currently only ``"mean"`` is supported.

        Returns
        -------
        pandas.DataFrame
            Dataframe of aggregated features prefixed with ``nbr_mean_`` and
            indexed like ``feats_df``.

        Notes
        -----
        Non-numeric columns are ignored and isolated nodes yield zero-valued
        aggregations.
        """
        if A.shape[0] == 0:
            return pd.DataFrame(index=feats_df.index)

        feats_num = feats_df.select_dtypes(include=[np.number, 'bool'])
        if feats_num.shape[1] == 0:
            return pd.DataFrame(index=feats_df.index)

        X = feats_num.to_numpy(dtype=float)
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg_safe = np.maximum(deg, 1.0)

        # mean
        nbr = (A @ X) / deg_safe[:, None]

        out = pd.DataFrame(
            nbr,
            index=feats_df.index,
            columns=[f"nbr_mean_{c}" for c in feats_num.columns]
        )
        return out

    def neighbor_label_counts_and_proportions(self,
                                              A: sparse.csr_matrix,
                                              labels: pd.Series,
                                              classes_: np.ndarray) -> pd.DataFrame:
        """Summarise counts and proportions of labelled neighbour classes.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Adjacency matrix describing neighbourhood relationships.
        labels : pandas.Series
            Series of node labels with ``NaN`` for unlabelled entries.
        classes_ : numpy.ndarray
            Unique class identifiers expected in ``labels``.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing ``nbr_count_*`` and ``nbr_prop_*`` columns for
            each class, indexed by the original nodes.

        Notes
        -----
        Nodes without labelled neighbours return zeros for counts and
        proportions. Matrix multiplication is used for efficient aggregation.
        """
        n = A.shape[0]
        if n == 0:
            return pd.DataFrame(index=labels.index)

        L_mask = labels.notna().to_numpy()
        if not L_mask.any():
            cols = []
            for cls in classes_:
                cols += [f"nbr_count_{cls}", f"nbr_prop_{cls}"]
            return pd.DataFrame(0.0, index=labels.index, columns=cols)

        idx_L = np.where(L_mask)[0]
        A_L = A[:, idx_L]                               # (n x |L|)
        y_L = labels.iloc[idx_L].to_numpy()

        cls2col = {cls: i for i, cls in enumerate(classes_)}
        Y = np.zeros((len(idx_L), len(classes_)), dtype=np.float32)
        for i, cls in enumerate(y_L):
            Y[i, cls2col[cls]] = 1.0

        counts = A_L @ Y                                 # (n x C)
        counts = np.asarray(counts, dtype=float)

        labeled_deg = np.asarray(A_L.sum(axis=1)).ravel()
        labeled_deg_safe = np.maximum(labeled_deg, 1.0)
        props = counts / labeled_deg_safe[:, None]

        df_counts = pd.DataFrame(counts, index=labels.index,
                                 columns=[f"nbr_count_{c}" for c in classes_])
        df_props  = pd.DataFrame(props, index=labels.index,
                                 columns=[f"nbr_prop_{c}" for c in classes_])
        return pd.concat([df_counts, df_props], axis=1)

    def calc_polygon_features(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Compute geometric descriptors for each polygon.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing polygon geometries.

        Returns
        -------
        pandas.DataFrame
            Dataframe with compactness, fractal dimension, bounding-box, and
            asymmetry metrics aligned with ``gdf``.
        """
        try:
            geo = gdf.geometry
            area = geo.area.to_numpy()
            length = geo.length.to_numpy()
            centroids = geo.centroid
            cx, cy = centroids.x.to_numpy(), centroids.y.to_numpy()
            
            min_env = geo.minimum_rotated_rectangle()
            mbr_area = min_env.area.to_numpy()
            convex = geo.convex_hull
            convex_area = convex.area.to_numpy()
            
            compactness = np.where(length>0, 4*np.pi*area/length**2, 0)
            fractal_dim = np.where((area>0)&(length>0)&(np.log(length)!=0), np.log(area)/np.log(length), 0)
            rectangularity = np.where(mbr_area>0, area/mbr_area, 0)
            
            bounds = min_env.bounds
            dx = bounds.maxx - bounds.minx
            dy = bounds.maxy - bounds.miny
            
            aspect_ratio = np.where((dx>0)&(dy>0), np.maximum(dx,dy)/np.minimum(dx,dy), 0)
            squareness = np.where(np.maximum(dx,dy)>0, np.minimum(dx,dy)/np.maximum(dx,dy), 0)
            shape_index = np.where(length>0, area / length, 0)
            solidity = np.where(convex_area>0, area/convex_area, 0)
            
            asym_x = np.abs((bounds.minx+bounds.maxx)/2 - cx)
            asym_y = np.abs((bounds.miny+bounds.maxy)/2 - cy)
            
            result = pd.DataFrame({
                'compactness': compactness,
                'fractal_dimension': fractal_dim,
                'shape_index': shape_index,
                'mbr_area': mbr_area,
                'rectangularity_index': rectangularity,
                'mbr_aspect_ratio': aspect_ratio,
                'squareness_index': squareness,
                'solidity': solidity,
                'asymmetry_x': asym_x,
                'asymmetry_y': asym_y,
            }, index=gdf.index)
            return result
            
        except Exception as e:
            raise e

    def count_nearby_zones(self, gdf: gpd.GeoDataFrame, rec_gdf: Optional[gpd.GeoDataFrame],
                          buffer_distance: float) -> pd.Series:
        """Count reference zones within buffered distances around geometries.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Source geometries for which surrounding zones are tallied.
        rec_gdf : geopandas.GeoDataFrame, optional
            Zones to count; may be ``None`` to skip counting.
        buffer_distance : float
            Buffer radius, expressed in the CRS units.

        Returns
        -------
        pandas.Series
            Counts indexed by ``gdf`` rows with zeros where no zones intersect.
        """

        if rec_gdf is None or rec_gdf.empty:
            return pd.Series(0, index=gdf.index)
            
        try:
            # Check and transform CRS if needed
            if gdf.crs != rec_gdf.crs:
                rec = rec_gdf.to_crs(gdf.crs)
            else:
                rec = rec_gdf
                
            buffers = gdf.geometry.buffer(buffer_distance)
            buff_gdf = gpd.GeoDataFrame(geometry=buffers, crs=gdf.crs)
            
            joined = gpd.sjoin(buff_gdf, rec[['geometry']], how='left', predicate='intersects')
            counts = joined.groupby(joined.index).size()
            
            result = counts.reindex(gdf.index, fill_value=0).astype(int)
            return result
            
        except Exception as e:
            raise e

    def transform_features(self, gdf, known_gdf_for_rec_zones=None):
        """Generate core feature set for a GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input geographic data to process.
        known_gdf_for_rec_zones : geopandas.GeoDataFrame, optional
            Reference zones used for neighbourhood counts.

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of ``gdf`` with cleaned geometries, local coordinates, and
            engineered features.
        """
        
        gdf = gdf.copy()
        
        gdf.geometry = gdf.geometry.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
        
        centroids = gdf.geometry.centroid
        gdf['x_local'] = 0.0
        gdf['y_local'] = 0.0
        
        if 'city' in gdf:
            
            def get_city_center(group):
                """
                Calculate the center point of a city from its geometries.
                
                Args:
                    group (Series): Group of geometries belonging to a city
                
                Returns:
                    Point: Center point of the city
                """
                try:
                    valid_geoms = group.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
                    union = valid_geoms.unary_union
                    if not union.is_valid:
                        union = make_valid(union)
                    return union.centroid
                except Exception as e:
                    return group.iloc[0].centroid
            
            cc_geom = gdf.groupby('city')['geometry'].apply(get_city_center)
            ccdf = cc_geom.apply(lambda p: pd.Series({'x': p.x, 'y': p.y}))
            
            gdf = gdf.join(ccdf, on='city')
            gdf['x_local'] = centroids.x - gdf['x']
            gdf['y_local'] = centroids.y - gdf['y']
            gdf = gdf.drop(columns=['x', 'y'])
        
        calc = self.calc_polygon_features(gdf)
        gdf = pd.concat([gdf, calc], axis=1)
        
        for z in ['rec_spec_agri','bus_res','industrial','transport']:
            if known_gdf_for_rec_zones is not None and 'land_use' in known_gdf_for_rec_zones:
                rec_z = known_gdf_for_rec_zones[known_gdf_for_rec_zones['land_use']==z]
            else:
                rec_z = None
                
            gdf[f'nearby_{z}_count'] = self.count_nearby_zones(gdf, rec_z, self.buffer_distance)
        
        return gdf

    def prepare_data(self, gdf: gpd.GeoDataFrame,
                    target_col: str = 'land_use_code',
                    radius: float = 1000.0,
                    k_neighbors: int = None,
                    classes_: np.ndarray = None) -> pd.DataFrame:
        """
        Prepare feature DataFrame from input GeoDataFrame by computing node features and neighbor aggregates.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame containing geometries and target values
        target_col : str, optional
            Name of the target column (default: 'land_use_code')
        radius : float, optional
            Search radius for neighbor detection in meters (default: 1000.0)
        k_neighbors : int, optional
            Number of neighbors to consider if using KNN mode (default: None)
        classes_ : np.ndarray, optional
            Array of class labels used for neighbor label features (default: None)
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing computed features without target column
            
        Notes
        -----
        The function performs the following steps:
        1. Computes basic node features
        2. Applies log transformation to specified columns
        3. For each city (or entire dataset if no city column):
           - Builds spatial graph using radius or KNN
           - Computes geometric feature aggregates from neighbors
           - If classes_ provided, computes neighbor label counts and proportions
        4. Combines all features into final DataFrame
        """
        gdf = gdf.copy()
        gdf.reset_index(drop=True, inplace=True)
        base = self.transform_features(gdf, known_gdf_for_rec_zones=None)  

        for col in self.columns_to_log:
            if col in base.columns:
                base[f'{col}_log'] = np.log1p(base[col])

        pieces = []
        for city, idx in gdf.groupby('city').indices.items() if 'city' in gdf.columns else {None: gdf.index}.items():
            city_idx = pd.Index(idx)
            city_base = base.loc[city_idx]

            A = self.build_city_graph(city_base, mode="radius" if k_neighbors is None else "knn",
                                    radius=radius, k=k_neighbors or 8)
            geom_cols = [c for c in city_base.columns
                        if c not in ('geometry', target_col, 'land_use', 'city', 'city_center')]
            geom_cols = [c for c in geom_cols if not c.startswith('nbr_') and not c.startswith('prob_') and not c.startswith('nearby_')]
            geom_df = city_base[geom_cols]

            nbr_geom = self.neighbor_geom_aggregates(A, geom_df, agg="mean")

            block = pd.concat([city_base[geom_cols], nbr_geom], axis=1)

            if classes_ is not None:
                labels = gdf.loc[city_idx, target_col]
                nbr_lbl = self.neighbor_label_counts_and_proportions(A, labels, classes_=classes_)
                block = pd.concat([block, nbr_lbl], axis=1)

            pieces.append(block)

        feats = pd.concat(pieces, axis=0).loc[gdf.index]
        return feats
