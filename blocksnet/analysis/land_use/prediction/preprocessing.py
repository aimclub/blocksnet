import h3
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from typing import Optional

from scipy import sparse
from scipy.spatial.distance import cdist
from shapely import make_valid
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from blocksnet.enums import LandUse, LandUseCategory



class DataProcessor:
    def __init__(self, buffer_distance: float = 1000, k_neighbors: int = 5):
        """Initialize the data processor with buffer distance and number of neighbors settings.
        
        Args:
            buffer_distance: Distance for buffering when counting nearby zones
            k_neighbors: Number of neighbors for KNN
        """
        self.buffer_distance = buffer_distance
        self.k_neighbors = k_neighbors
        self.knn_model = None
        
        # Feature lists for spatial context and logarithm transformation
        self.feature_names_for_spatial_context = [
            'mbr_area', 'solidity', 'compactness', 'shape_index', 
            'mbr_aspect_ratio', 'squareness_index', 'fractal_dimension',
            'rectangularity_index',
            'nearby_residential_count', 'nearby_business_count',
            'nearby_recreation_count', 'nearby_industrial_count',
        ]
        
        self.columns_to_log = [
            'shape_index', 'mbr_area', 'mbr_aspect_ratio', 
            'solidity', 'asymmetry_x', 'asymmetry_y'
        ]

        # Advanced feature settings inspired by prediction_v2.ipynb
        self.rings: tuple[tuple[int, int], ...] = ((0, 150), (150, 500), (500, 1000))
        self.graph_neighbor_buffer: float = 10.0
        self.distance_batch_size: int = 512
        self.h3_resolution: int = 9

    def build_city_graph(self, city_gdf: gpd.GeoDataFrame,
                        mode: str = "radius",
                        radius: float = 1000.0,
                        k: int = 8) -> sparse.csr_matrix:
        """Build a spatial adjacency graph from city geometries.

        Args:
            city_gdf (gpd.GeoDataFrame): GeoDataFrame containing city geometries
            mode (str, optional): Graph construction mode. Either "radius" or "knn". 
                Defaults to "radius".
            radius (float, optional): Connection radius in meters when mode="radius". 
                Defaults to 1000.0.
            k (int, optional): Number of nearest neighbors when mode="knn". 
                Defaults to 8.

        Returns:
            sparse.csr_matrix: Sparse adjacency matrix representing the spatial graph.
                The matrix has shape (n, n) where n is the number of geometries.
                A[i,j] = 1 indicates that geometries i and j are connected.

        Notes:
            - For radius mode, geometries within the given radius are connected
            - For knn mode, each geometry is connected to its k nearest neighbors
            - Self-connections are excluded
            - Returns empty matrix for empty input
            - Handles edge cases when n=1 or k >= n
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
        """
        Calculate neighbor aggregation features for nodes in a graph.
        
        This function computes aggregated features from neighboring nodes for each node
        in the graph, using only numeric features from the input feature matrix.

        Parameters
        ----------
        A : sparse.csr_matrix
            Adjacency matrix in CSR format with shape (n x n), where n is the number of nodes
        feats_df : pd.DataFrame
            Node features DataFrame with shape (n x F), where F is the number of features
        agg : str, optional
            Aggregation method to use (default is "mean")
            Note: Currently only "mean" is implemented

        Returns
        -------
        pd.DataFrame
            Aggregated neighbor features with shape (n x F), containing only numeric features.
            Non-numeric columns from input are ignored.
            Column names are prefixed with "nbr_mean_" to indicate neighbor mean aggregation.
            Returns empty DataFrame with same index if no numeric features are found or if
            adjacency matrix is empty.

        Notes
        -----
        - Only numeric features (including bool/Int64/Float64 types) are considered
        - Non-numeric columns are automatically ignored
        - For nodes with no neighbors, uses 1.0 as denominator to avoid division by zero
        - The output DataFrame maintains the same index as the input feats_df
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
        """
        Calculate for EACH node:
          - count_<cls>: number of neighbors of class <cls> among LABELED neighbors (L)
          - prop_<cls>: proportion of such neighbors relative to the NUMBER of LABELED neighbors
        
        Parameters
        ----------
        A : sparse.csr_matrix
            Adjacency matrix in CSR format representing the graph structure
        labels : pd.Series
            Series containing node labels, with NaN values for unlabeled nodes
        classes_ : np.ndarray
            Array containing all possible class labels
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing two types of columns for each class:
            - nbr_count_<cls>: count of labeled neighbors belonging to class <cls>
            - nbr_prop_<cls>: proportion of labeled neighbors belonging to class <cls>
            The DataFrame index matches the input labels index.
            
        Notes
        -----
        - If there are no labeled neighbors, all counts and proportions are set to 0.0
        - Proportions are calculated safely to avoid division by zero
        - The computation uses matrix multiplication for efficiency
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
        """
        Compute geometric descriptors for polygons (area, perimeter, shape indices, etc.).
        """
        if gdf.empty:
            return pd.DataFrame(index=gdf.index)

        eps = 1e-12
        geo = gdf.geometry
        area = np.nan_to_num(geo.area.to_numpy(), nan=0.0)
        perimeter = np.nan_to_num(geo.length.to_numpy(), nan=0.0)
        convex_area = np.nan_to_num(geo.convex_hull.area.to_numpy(), nan=0.0)
        centroids = geo.centroid
        cx = np.nan_to_num(centroids.x.to_numpy(), nan=0.0)
        cy = np.nan_to_num(centroids.y.to_numpy(), nan=0.0)

        bounds = geo.bounds
        minx = bounds["minx"].to_numpy()
        maxx = bounds["maxx"].to_numpy()
        miny = bounds["miny"].to_numpy()
        maxy = bounds["maxy"].to_numpy()
        bbox_width = np.nan_to_num(maxx - minx, nan=0.0)
        bbox_height = np.nan_to_num(maxy - miny, nan=0.0)

        compactness = np.where(
            perimeter > 0,
            (4.0 * np.pi * area) / (np.square(perimeter) + eps),
            0.0,
        )
        solidity = np.where(convex_area > 0, area / (convex_area + eps), 0.0)
        shape_index = np.where(
            area > 0,
            0.25 * perimeter / np.sqrt(area + eps),
            0.0,
        )
        fractal_dimension = np.where(
            (area > 0) & (perimeter > 0),
            2.0 * np.log((perimeter + eps) / 4.0) / np.log(area + eps),
            0.0,
        )
        fractal_dimension = np.nan_to_num(fractal_dimension, nan=0.0, posinf=0.0, neginf=0.0)
        fractal_dimension = np.clip(fractal_dimension, 0.0, 2.5)

        asym_x = np.abs(((minx + maxx) / 2.0) - cx)
        asym_y = np.abs(((miny + maxy) / 2.0) - cy)
        elongation = np.where(bbox_height > 0, bbox_width / (bbox_height + eps), 1.0)

        mrr_metrics = geo.apply(self._mrr_metrics_single)
        mrr_width = np.array([vals[0] for vals in mrr_metrics], dtype=float)
        mrr_height = np.array([vals[1] for vals in mrr_metrics], dtype=float)
        mrr_area = np.array([vals[2] for vals in mrr_metrics], dtype=float)
        mrr_aspect_ratio = np.array([vals[3] for vals in mrr_metrics], dtype=float)
        rectangularity_index = np.where(mrr_area > 0, area / (mrr_area + eps), 0.0)

        result = pd.DataFrame({
            'area': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'solidity': solidity,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'elongation': elongation,
            'mrr_height': mrr_height,
            'mrr_area': mrr_area,
            'mrr_aspect_ratio': mrr_aspect_ratio,
            'rectangularity_index': rectangularity_index,
            'shape_index': shape_index,
            'fractal_dimension': fractal_dimension,
            'asymmetry_x': asym_x,
            'asymmetry_y': asym_y,
        }, index=gdf.index)

        # Maintain legacy feature names for downstream compatibility
        result['mbr_area'] = result['mrr_area']
        result['mbr_aspect_ratio'] = result['mrr_aspect_ratio']
        result['squareness_index'] = np.where(
            result['mrr_aspect_ratio'] > 0,
            1.0 / (result['mrr_aspect_ratio'] + eps),
            0.0,
        )

        return result

    @staticmethod
    def _mrr_metrics_single(geom):
        """
        Return width, height, area and aspect ratio of the minimum rotated rectangle.
        """
        try:
            if geom is None or geom.is_empty:
                return (0.0, 0.0, 0.0, 1.0)
            mrr = geom.minimum_rotated_rectangle
            coords = np.array(mrr.exterior.coords)
            edges = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(axis=1))
            lengths = np.sort(np.unique(np.round(edges, 12)))
            if len(lengths) >= 2:
                width, height = float(lengths[-1]), float(lengths[-2])
            elif len(lengths) == 1:
                width = height = float(lengths[0])
            else:
                width = height = 0.0
            area = float(mrr.area)
            w, h = (width, height) if width >= height else (height, width)
            aspect = (w / (h + 1e-12)) if (w > 0 and h > 0) else 1.0
            return w, h, area, aspect
        except Exception:
            return (0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def _label_slug(value) -> str:
        if value is None:
            return "__none__"
        if isinstance(value, LandUseCategory):
            return value.value.lower()
        if isinstance(value, LandUse):
            cat = LandUseCategory.from_land_use(value)
            return cat.value.lower() if cat else value.value.lower()
        if isinstance(value, str):
            return value.strip().lower()
        return str(value).lower()

    @staticmethod
    def _to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.copy()
        try:
            utm_crs = gdf.estimate_utm_crs()
        except Exception:
            utm_crs = None
        if utm_crs:
            return gdf.to_crs(utm_crs)
        return gdf.copy()

    def _compute_ring_statistics(self, city_gdf: gpd.GeoDataFrame, target_col: str) -> pd.DataFrame:
        out = pd.DataFrame(index=city_gdf.index)
        if city_gdf.empty or city_gdf.geometry.is_empty.all():
            return out

        working = city_gdf.copy()
        if target_col not in working.columns:
            working[target_col] = None

        for r_min, r_max in self.rings:
            buf_max = working.geometry.buffer(r_max)
            if r_min > 0:
                buf_min = working.geometry.buffer(r_min)
                ring_geom = buf_max.difference(buf_min)
            else:
                ring_geom = buf_max

            right = gpd.GeoDataFrame(
                working[[target_col, "area"]].copy(),
                geometry=ring_geom,
                crs=working.crs,
            )
            joined = gpd.sjoin(
                working[[target_col, "geometry", "area"]],
                right,
                how="left",
                predicate="intersects",
                lsuffix="A",
                rsuffix="B",
            ).reset_index()

            left_index = "index"
            right_index = "index_B"
            if left_index not in joined.columns and joined.index.name:
                joined = joined.reset_index(names="index_left")
                left_index = "index_left"
            if right_index not in joined.columns:
                right_index = "index_right"

            joined = joined[joined[right_index].notna()]
            joined = joined[joined[left_index] != joined[right_index]]
            if joined.empty:
                band = f"{int(r_min)}_{int(r_max)}m"
                for col in (
                    f"n_neighbors_{band}m",
                    f"avg_area_neighbors_{band}m",
                    f"n_neighbors_none_{band}m",
                    f"avg_area_neighbors_none_{band}m",
                ):
                    out[col] = 0 if "avg" not in col else 0.0
                continue

            joined[f"{target_col}_B"] = joined[f"{target_col}_B"].fillna("__none__")
            joined["_neighbor_class"] = joined[f"{target_col}_B"].map(self._label_slug)
            band = f"{int(r_min)}_{int(r_max)}m"

            n_neighbors = joined.groupby(left_index).size()
            avg_area_neighbors = joined.groupby(left_index)["area_B"].mean()

            none_mask = joined["_neighbor_class"] == "__none__"
            n_neighbors_none = none_mask.groupby(joined[left_index]).sum()
            avg_area_none = (
                joined.loc[none_mask].groupby(left_index)["area_B"].mean()
                if none_mask.any()
                else pd.Series(dtype=float)
            )

            out[f"n_neighbors_{band}m"] = out.index.map(n_neighbors).fillna(0).astype(int)
            out[f"avg_area_neighbors_{band}m"] = out.index.map(avg_area_neighbors).fillna(0.0)
            out[f"n_neighbors_none_{band}m"] = out.index.map(n_neighbors_none).fillna(0).astype(int)
            out[f"avg_area_neighbors_none_{band}m"] = out.index.map(avg_area_none).fillna(0.0)

            pct_by_class = (
                joined[joined["_neighbor_class"] != "__none__"]
                .groupby([left_index, "_neighbor_class"])
                .size()
                .unstack(fill_value=0)
            )
            if pct_by_class.empty:
                continue
            pct_by_class = pct_by_class.div(pct_by_class.sum(axis=1), axis=0)
            for cls in pct_by_class.columns:
                slug = self._label_slug(cls)
                if slug == "__none__":
                    continue
                col_name = f"pct_neighbor_{slug}_{band}m"
                out[col_name] = out.index.map(pct_by_class[cls]).fillna(0.0)

        return out

    def _compute_distance_metrics(self, city_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=city_gdf.index)
        n = len(city_gdf)
        if n == 0:
            out["median_dist"] = pd.Series(dtype=float)
            out["mean_k3_dist"] = pd.Series(dtype=float)
            return out

        reps = city_gdf.geometry.representative_point()
        coords = np.column_stack((reps.x.values, reps.y.values)).astype(float)
        median_dist = np.zeros(n, dtype=float)
        mean_k3_dist = np.zeros(n, dtype=float)

        if n > 1:
            for start in range(0, n, self.distance_batch_size):
                end = min(start + self.distance_batch_size, n)
                batch = coords[start:end]
                dist_block = cdist(batch, coords)
                for local_row, global_idx in enumerate(range(start, end)):
                    dist_block[local_row, global_idx] = np.nan
                median_dist[start:end] = np.nanmedian(dist_block, axis=1)
                sorted_block = np.sort(dist_block, axis=1)
                k_lim = min(3, sorted_block.shape[1] - 1)
                if k_lim > 0:
                    mean_k3_dist[start:end] = np.nanmean(sorted_block[:, :k_lim], axis=1)
        median_dist = np.nan_to_num(median_dist, nan=0.0)
        mean_k3_dist = np.nan_to_num(mean_k3_dist, nan=0.0)
        out["median_dist"] = median_dist
        out["mean_k3_dist"] = mean_k3_dist
        return out

    def _compute_local_graph_features(self, city_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=city_gdf.index)
        if city_gdf.empty:
            return out

        buffers = city_gdf.geometry.buffer(self.graph_neighbor_buffer)
        right = gpd.GeoDataFrame(index=city_gdf.index, geometry=buffers, crs=city_gdf.crs)
        pairs = gpd.sjoin(
            city_gdf[["geometry"]],
            right,
            how="left",
            predicate="intersects",
            lsuffix="L",
            rsuffix="R",
        ).reset_index()

        left_index = "index"
        right_index = "index_R"
        if left_index not in pairs.columns and pairs.index.name:
            pairs = pairs.reset_index(names="index_left")
            left_index = "index_left"
        if right_index not in pairs.columns:
            right_index = "index_right"

        pairs = pairs[pairs[right_index].notna()]
        pairs = pairs[pairs[left_index] != pairs[right_index]]
        edges = list(zip(pairs[left_index], pairs[right_index]))

        G = nx.Graph()
        G.add_nodes_from(city_gdf.index.tolist())
        if edges:
            G.add_edges_from(edges)

        deg_dict = dict(G.degree())
        clust_dict = nx.clustering(G)
        avg_neighbor = nx.average_neighbor_degree(G) if G.number_of_edges() > 0 else {node: 0.0 for node in G.nodes()}

        comp_id = {}
        comp_size = {}
        for cid, comp_nodes in enumerate(nx.connected_components(G), start=1):
            size = len(comp_nodes)
            for node in comp_nodes:
                comp_id[node] = cid
                comp_size[node] = size

        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)
        except Exception:
            pagerank = {node: 0.0 for node in G.nodes()}

        out["deg_10m"] = out.index.map(deg_dict).fillna(0).astype(int)
        out["clust_10m"] = out.index.map(clust_dict).fillna(0.0)
        out["avg_n_deg_10m"] = out.index.map(avg_neighbor).fillna(0.0)
        out["component_id_10m"] = out.index.map(comp_id).fillna(0).astype(int)
        out["component_size_10m"] = out.index.map(comp_size).fillna(1).astype(int)
        out["pagerank_10m"] = out.index.map(pagerank).fillna(0.0)
        return out

    def _compute_h3_features(self, city_gdf: gpd.GeoDataFrame, target_col: str) -> pd.DataFrame:
        if h3 is None:
            raise ImportError("h3 package is required to compute hexagonal features. Install via `pip install h3`.")  # pragma: no cover

        out = pd.DataFrame(index=city_gdf.index)
        if city_gdf.empty:
            return out

        latlon = city_gdf.to_crs(4326)
        centroids = latlon.geometry.centroid

        def _cell(point):
            if point is None or point.is_empty:
                return None
            return h3.latlng_to_cell(point.y, point.x, self.h3_resolution)

        h3_index = centroids.apply(_cell)
        h3_series = pd.Series(h3_index.to_numpy(), index=city_gdf.index)
        valid = h3_series.dropna()

        density = h3_series.map(valid.value_counts())
        area_mean = h3_series.map(city_gdf.groupby(h3_series)["area"].mean())

        if target_col in city_gdf.columns:
            targets = city_gdf[target_col].apply(self._label_slug)
        else:
            targets = pd.Series("__none__", index=city_gdf.index)

        def _entropy(series: pd.Series) -> float:
            freq = series.value_counts(normalize=True)
            return float(-(freq * np.log(freq + 1e-12)).sum())

        entropy = targets.groupby(h3_series).apply(_entropy)
        out["h3_density"] = density.reindex(city_gdf.index).fillna(0).astype(int)
        out["h3_mean_area"] = area_mean.reindex(city_gdf.index).fillna(0.0)
        out["h3_entropy"] = h3_series.map(entropy).fillna(0.0)
        return out

    def _compute_city_features(self, city_gdf: gpd.GeoDataFrame, target_col: str) -> pd.DataFrame:
        metric = self._to_metric(city_gdf)
        if metric.empty:
            return pd.DataFrame(index=city_gdf.index)

        poly_feats = self.calc_polygon_features(metric)
        metric = metric.join(poly_feats)

        ring_feats = self._compute_ring_statistics(metric, target_col=target_col)
        dist_feats = self._compute_distance_metrics(metric)
        graph_feats = self._compute_local_graph_features(metric)
        h3_feats = self._compute_h3_features(metric, target_col=target_col)

        components = [poly_feats, ring_feats, dist_feats, graph_feats, h3_feats]
        combined = pd.concat(components, axis=1)
        return combined

    def count_nearby_zones(self, gdf: gpd.GeoDataFrame, rec_gdf: Optional[gpd.GeoDataFrame], 
                          buffer_distance: float) -> pd.Series:
        """Counts the number of zones of a specific type within a buffer around each object.
        
        This method creates a buffer around each geometry in the main GeoDataFrame and counts
        how many zones from the second GeoDataFrame intersect with each buffer. The method handles
        CRS transformations automatically and ensures proper indexing in the result.
        
        Args:
            gdf: Main GeoDataFrame containing the geometries to create buffers around
            rec_gdf: GeoDataFrame containing zones to be counted within buffers
            buffer_distance: Distance for buffer creation in the units of the CRS
            
        Returns:
            pd.Series: Series containing the count of zones for each object, indexed by the
                      original GeoDataFrame's index. Returns zeros for objects with no
                      intersecting zones if rec_gdf is None or empty.
                      
        Raises:
            Exception: Propagates any exceptions that occur during spatial operations
            
        Note:
            - The method automatically handles CRS transformation if input GeoDataFrames
              have different coordinate reference systems
            - If rec_gdf is None or empty, returns a Series of zeros
            - The result maintains the same index as the input gdf
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

    def _map_land_use_to_category(self, ser: pd.Series) -> pd.Series:
        """Map raw land_use values to LandUseCategory using enum mapping.

        Accepts strings or LandUse; returns LandUseCategory or NaN if unmapped.
        """
        def to_category(v):
            try:
                if isinstance(v, LandUse):
                    lu = v
                elif isinstance(v, str):
                    lu = LandUse(v.lower())
                else:
                    return np.nan
                cat = LandUseCategory.from_land_use(lu)
                return cat if cat is not None else np.nan
            except Exception:
                return np.nan
        return ser.map(to_category)

    def count_nearby_by_category(
        self,
        gdf: gpd.GeoDataFrame,
        known_gdf: Optional[gpd.GeoDataFrame],
        buffer_distance: float,
        exclude_self: bool = True,
        require_strict: bool = False,
    ) -> pd.DataFrame:
        """Count nearby zones per LandUseCategory within a buffer.

        Returns columns: nearby_<category>_count for each category in LandUseCategory
        where <category> is lowercased (e.g., nearby_industrial_count).
        If known_gdf is None or lacks land_use, returns zeros.
        """
        # Helper to build safe suffix from category
        def _cat_suffix(c):
            val = getattr(c, 'value', c)
            return str(val).lower() if val is not None else 'unknown'

        # Prepare zero frame in fallback cases
        zero_cols = [f"nearby_{_cat_suffix(c)}_count" for c in LandUseCategory]
        if known_gdf is None or len(known_gdf) == 0 or 'land_use' not in known_gdf.columns:
            return pd.DataFrame(0, index=gdf.index, columns=zero_cols)

        # Ensure same CRS
        if gdf.crs != known_gdf.crs:
            rec = known_gdf.to_crs(gdf.crs)
        else:
            rec = known_gdf

        # Map to categories
        rec = rec.copy()
        rec['__lu_cat__'] = self._map_land_use_to_category(rec['land_use'])
        if require_strict and rec['__lu_cat__'].isna().any():
            bad_vals = (
                rec.loc[rec['__lu_cat__'].isna(), 'land_use']
                .astype(str)
                .value_counts()
                .head(5)
                .to_dict()
            )
            raise ValueError(f"Unmapped land_use to LandUseCategory in training data: {bad_vals}")
        # If all NaN, return zeros
        if rec['__lu_cat__'].isna().all():
            return pd.DataFrame(0, index=gdf.index, columns=zero_cols)

        # Build buffers once
        buffers = gdf.geometry.buffer(buffer_distance)
        buff_gdf = gpd.GeoDataFrame(geometry=buffers, crs=gdf.crs)

        out = {}
        for cat in LandUseCategory:
            cat_mask = rec['__lu_cat__'] == cat
            rec_cat = rec.loc[cat_mask, ['geometry']]
            if rec_cat.empty:
                out[f"nearby_{_cat_suffix(cat)}_count"] = pd.Series(0, index=gdf.index)
                continue

            joined = gpd.sjoin(buff_gdf, rec_cat, how='left', predicate='intersects')
            if exclude_self and 'index_right' in joined.columns:
                # Drop self-joins for overlapping indices
                overlap = rec_cat.index.intersection(gdf.index)
                if len(overlap) > 0:
                    joined = joined[joined.index != joined['index_right']]
            counts = joined.groupby(joined.index).size()
            out[f"nearby_{_cat_suffix(cat)}_count"] = counts.reindex(gdf.index, fill_value=0).astype(int)

        return pd.DataFrame(out)

    def transform_features(self, gdf, target_col: str = 'category',
                           known_gdf_for_rec_zones=None,
                           require_strict_categories: bool = False):
        """
        Transform features of a GeoDataFrame with invalid geometry handling.
        
        This method performs several transformations on the input GeoDataFrame:
        1. Validates and fixes invalid geometries
        2. Calculates local coordinates relative to city centers
        3. Computes advanced geometric, network, and contextual features
        4. Counts nearby zones of different types (using known_gdf_for_rec_zones)
        
        Args:
            gdf (GeoDataFrame): Input geographic data to be transformed
            target_col (str): Column describing current class labels (default: 'category')
            known_gdf_for_rec_zones (GeoDataFrame, optional): Reference data for 
                counting nearby zones. Defaults to None.
        
        Returns:
            GeoDataFrame: Transformed geographic data with new features including:
                - Validated geometries
                - Local coordinates (x_local, y_local)
                - Geometric features
                - Counts of nearby zones by type
        """
        
        gdf = gdf.copy()
        
        gdf.geometry = gdf.geometry.apply(
            lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
        )
        
        # centroids = gdf.geometry.centroid
        # gdf['x_local'] = 0.0
        # gdf['y_local'] = 0.0
        
        # if 'city' in gdf:
            
        #     def get_city_center(group):
        #         """
        #         Calculate the center point of a city from its geometries.
                
        #         Args:
        #             group (Series): Group of geometries belonging to a city
                
        #         Returns:
        #             Point: Center point of the city
        #         """
        #         try:
        #             valid_geoms = group.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
        #             union = valid_geoms.unary_union
        #             if not union.is_valid:
        #                 union = make_valid(union)
        #             return union.centroid
        #         except Exception as e:
        #             return group.iloc[0].centroid
            
        #     cc_geom = gdf.groupby('city')['geometry'].apply(get_city_center)
        #     ccdf = cc_geom.apply(lambda p: pd.Series({'x': p.x, 'y': p.y}))
            
        #     gdf = gdf.join(ccdf, on='city')
        #     gdf['x_local'] = centroids.x - gdf['x']
        #     gdf['y_local'] = centroids.y - gdf['y']
        #     gdf = gdf.drop(columns=['x', 'y'])
        
        feature_blocks = []
        if 'city' in gdf.columns:
            city_iter = gdf.groupby('city').indices.items()
        else:
            city_iter = [(None, gdf.index)]
        for _, idx in city_iter:
            idx = pd.Index(idx)
            city_slice = gdf.loc[idx].copy()
            city_features = self._compute_city_features(city_slice, target_col=target_col)
            feature_blocks.append(city_features)
        if feature_blocks:
            features_df = pd.concat(feature_blocks, axis=0)
            features_df = features_df.reindex(gdf.index)
            gdf = pd.concat([gdf, features_df], axis=1)
        
        # Nearby counts by LandUseCategory
        # try:
        #     nearby_df = self.count_nearby_by_category(
        #         gdf,
        #         known_gdf_for_rec_zones,
        #         buffer_distance=self.buffer_distance,
        #         exclude_self=True,
        #         require_strict=require_strict_categories,
        #     )
        # except Exception:
        #     # If something goes wrong, fall back to zeros to avoid breaking pipeline
        #     nearby_df = pd.DataFrame(
        #         0,
        #         index=gdf.index,
        #         columns=[f"nearby_{c.value.lower()}_count" for c in LandUseCategory],
        #     )

        # gdf = pd.concat([gdf, nearby_df], axis=1)

        return gdf

    def prepare_data(self, gdf: gpd.GeoDataFrame,
                    target_col: str = 'category',
                    radius: float = 1000.0,
                    k_neighbors: int = None,
                    classes_: np.ndarray = None,
                    known_gdf_for_rec_zones: Optional[gpd.GeoDataFrame] = None,
                    require_strict_categories: bool = False) -> pd.DataFrame:
        """
        Prepare feature DataFrame from input GeoDataFrame by computing node features and neighbor aggregates.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame containing geometries and target values
        target_col : str, optional
            Name of the target column (default: 'category')
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
        base = self.transform_features(
            gdf,
            target_col=target_col,
            known_gdf_for_rec_zones=known_gdf_for_rec_zones,
            require_strict_categories=require_strict_categories,
        )  

        pieces = []
        for city, idx in gdf.groupby('city').indices.items() if 'city' in gdf.columns else {None: gdf.index}.items():
            city_idx = pd.Index(idx)
            city_base = base.loc[city_idx]

            A = self.build_city_graph(city_base, mode="radius" if k_neighbors is None else "knn",
                                    radius=radius, k=k_neighbors or 8)
            # Base feature columns used for modeling and neighbor aggregation.
            # We exclude service columns and already-derived neighbor/probability columns,
            # but we KEEP nearby_* counts so they are included in the model and can also
            # be aggregated to nbr_mean_nearby_* if numeric.
            geom_cols = [c for c in city_base.columns
                        if c not in ('geometry', target_col, 'land_use', 'city', 'city_center')]
            geom_cols = [c for c in geom_cols if not c.startswith('nbr_') and not c.startswith('prob_')]
            block = city_base[geom_cols]

            # nbr_geom = self.neighbor_geom_aggregates(A, geom_df, agg="mean")
            # block = pd.concat([city_base[geom_cols], nbr_geom], axis=1)

            if classes_ is not None:
                labels = gdf.loc[city_idx, target_col]
                nbr_lbl = self.neighbor_label_counts_and_proportions(A, labels, classes_=classes_)
                block = pd.concat([block, nbr_lbl], axis=1)

            pieces.append(block)

        feats = pd.concat(pieces, axis=0).loc[gdf.index]
        return feats
