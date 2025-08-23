import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as st
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import CCA
import scipy.linalg
from tqdm import tqdm
from scipy import sparse

def preprocess_adata(adata, normalize_flag=False):
    """
    Preprocess AnnData object for PV analysis.
    
    Parameters:
    adata: AnnData object
    normalize_flag: bool, whether to normalize and log-transform the data
    
    Returns:
    adata: preprocessed AnnData object
    """
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_counts=1)

    if normalize_flag:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
   
    return adata

import harmonypy as hm
def get_harmony_embedding(adata_sp, adata_sc, n_pcs=100):
    adata_sp_raw = adata_sp.copy()
    adata_sc_raw = adata_sc.copy()

    batch_key="batch"
    adata = adata_sp.concatenate(adata_sc, batch_key = batch_key)
    adata = preprocess_adata(adata)
    sc.tl.pca(adata, n_comps=n_pcs)
    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, batch_key)
    adata_sp_raw.obsm['X_pca'] = adata.obsm['X_pca'][:adata_sp_raw.n_obs]
    adata_sc_raw.obsm['X_pca'] = adata.obsm['X_pca'][adata_sp_raw.n_obs:]
    adata_sp_raw.obsm['X_shared'] = ho.Z_corr.T[:adata_sp_raw.n_obs]
    adata_sc_raw.obsm['X_shared'] = ho.Z_corr.T[adata_sp_raw.n_obs:]
    
    return adata_sp_raw, adata_sc_raw

def get_common_embedding(
    adata_sp: sc.AnnData,
    adata_sc: sc.AnnData,
    emb: str = "pca",
    method: str = "svd",
    n_components: int = 10,
    cos_threshold: float = 0.3,
    project_on: str = "sc",
    normalize_flag: bool = True,
    zscore_flag: bool = True,
):
    """
    Compute shared embedding for spatial & single-cell data.

    Parameters
    ----------
    emb : "pca", "raw", or "nmf"
    method : "svd" or "cca"
    project_on : "sc", "sp", or "both"
    normalize_flag : bool, whether to normalize data before processing
    """
    if emb not in ["pca", "raw", "nmf", "cca"]:
        raise ValueError("emb must be 'pca', 'cca' ,'raw', or 'nmf'")
    if method not in ["svd", "cca"]:
        raise ValueError("method must be 'svd' or 'cca'")

    adata_sp_raw = adata_sp.copy()
    adata_sc_raw = adata_sc.copy()
    adata_sp = adata_sp.copy()
    adata_sc = adata_sc.copy()

    adata_sp = preprocess_adata(adata_sp, normalize_flag)
    adata_sc = preprocess_adata(adata_sc, normalize_flag)

    common_genes = np.intersect1d(adata_sp.var_names, adata_sc.var_names)
    sp_data = adata_sp[:, common_genes].X.toarray()
    sc_data = adata_sc[:, common_genes].X.toarray()
    sc_data_all_genes = adata_sc.X.toarray()
    if zscore_flag:
        sp_data = st.zscore(sp_data, axis=0)
        sc_data = st.zscore(sc_data, axis=0)


    # Adjust n_components based on data dimensions
    max_components = min(sp_data.shape[1], n_components)
    min_sample_num = min(sp_data.shape[0], sc_data.shape[0])
    if max_components > min_sample_num:
        max_components = min_sample_num
        print(f"Warning: max_components is adjusted to {max_components} based on data dimensions")

    if emb == "raw":
        max_components = min(max_components, sp_data.shape[1])

    # Embedding step
    print(f"begin to run {emb}")
    if emb == "pca":
        pca_sp = PCA(n_components=max_components)
        pca_sc = PCA(n_components=max_components)
        sp_factors = pca_sp.fit(sp_data).components_
        sc_factors = pca_sc.fit(sc_data).components_
    if emb == "cca":
        pca_sp = PCA(n_components=max_components)
        cca_sc = CCA(n_components=max_components)
        sp_factors = pca_sp.fit(sp_data).components_
        sc_factors = cca_sc.fit(sc_data_all_genes, sc_data).y_weights_
    elif emb == "raw":
        sp_factors = sp_data.T
        sc_factors = sc_data.T
    elif emb == "nmf":
        # Shift data to non-negative for NMF
        min_sp = np.min(sp_data)
        min_sc = np.min(sc_data)
        shift = min(0, min_sp, min_sc)
        sp_data_shifted = sp_data - shift
        sc_data_shifted = sc_data - shift
        nmf_model = NMF(n_components=max_components, init='nndsvd',
                        max_iter=200, random_state=0)
        sp_factors = nmf_model.fit(sp_data_shifted).components_
        sc_factors = nmf_model.fit(sc_data_shifted).components_

    print(f"begin to run orth")
    sp_factors = scipy.linalg.orth(sp_factors.T).T
    sc_factors = scipy.linalg.orth(sc_factors.T).T

    print(f"begin to run decomposition {method}")
    # Decomposition step
    if method == "svd":
        # SVD on domain-specific factors (n_components, n_genes)
        u, sigma, v = np.linalg.svd(sc_factors @ sp_factors.T)
        # Compute principal vectors in gene space
        sc_pv = u.T @ sc_factors  # (n_components, n_genes)
        sp_pv = v @ sp_factors    # (n_components, n_genes)
        # Normalize
        sc_pv = normalize(sc_pv, axis=1)
        sp_pv = normalize(sp_pv, axis=1)
        # Cosine similarity
        cos_similarity = sc_pv @ sp_pv.T
        cos_similarity = np.diag(cos_similarity)
        print(f"Max: {np.max(cos_similarity):.4f}, Min: {np.min(cos_similarity):.4f}, Median: {np.median(cos_similarity):.4f}")
        # print(cos_similarity)
        effective_n_pv = np.sum(cos_similarity > cos_threshold)
        print(f"PV: Selected {effective_n_pv} effective components based on cosine similarity threshold {cos_threshold}")
        sc_pv = sc_pv[:effective_n_pv]
        sp_pv = sp_pv[:effective_n_pv]
    elif method == "cca":
        cca = CCA(n_components=max_components, scale=False)
        cca.fit(sp_data, sc_data)
        sc_pv = cca.y_weights_.T  # (n_components, n_genes)
        sp_pv = cca.x_weights_.T  # (n_components, n_genes)
        sc_pv = normalize(sc_pv, axis=1)
        sp_pv = normalize(sp_pv, axis=1)
        correlations = np.diag(sc_pv @ sp_pv.T)
        effective_n_pv = np.sum(correlations > cos_threshold)
        print(f"PV: Selected {effective_n_pv} effective components based on correlation threshold {cos_threshold}")
        sc_pv = sc_pv[:effective_n_pv]
        sp_pv = sp_pv[:effective_n_pv]

    # Project data: use standardized original data and PVs
    # sp_data, sc_data: (n_cells, n_genes)
    # sc_pv, sp_pv: (effective_n_pv, n_genes)
    print(f"begin to project on {project_on}")
    if project_on == "sc":
        adata_sp_raw.obsm["X_shared"] = sp_data @ sc_pv.T  # (n_spatial_cells, effective_n_pv)
        adata_sc_raw.obsm["X_shared"] = sc_data @ sc_pv.T  # (n_single_cells, effective_n_pv)
        # adata_sp_raw.varm["PV"] = sc_pv.T
        # adata_sc_raw.varm["PV"] = sc_pv.T
    elif project_on == "sp":
        adata_sp_raw.obsm["X_shared"] = sp_data @ sp_pv.T
        adata_sc_raw.obsm["X_shared"] = sc_data @ sp_pv.T
        # adata_sp_raw.varm["PV"] = sp_pv.T
        # adata_sc_raw.varm["PV"] = sp_pv.T
    elif project_on == "both":
        adata_sp_raw.obsm["X_shared_sc"] = sp_data @ sc_pv.T
        adata_sc_raw.obsm["X_shared_sc"] = sc_data @ sc_pv.T
        adata_sp_raw.obsm["X_shared_sp"] = sp_data @ sp_pv.T
        adata_sc_raw.obsm["X_shared_sp"] = sc_data @ sp_pv.T
    else:
        raise ValueError("project_on must be 'sc', 'sp', or 'both'")

    adata_sp_raw.uns["shared"] = {"method": "PV", "project_on": project_on}
    adata_sc_raw.uns["shared"] = {"method": "PV", "project_on": project_on}

    return adata_sp_raw, adata_sc_raw

def get_imputed_adata(
    adata_sp,
    adata_sc,
    genes_to_predict=None,
    n_neighbors=50,
    method="NearestNeighbors",
    ot_mapping=None
):
    """
    Impute spatial gene expression using sc data.

    Parameters
    ----------
    method : "NearestNeighbors" or "MNN"
    """
    if method == "NearestNeighbors":
        return _get_impute_NearestNeighbors(adata_sp, adata_sc, genes_to_predict, n_neighbors)
    elif method == "MNN":
        return _get_impute_MNN(adata_sp, adata_sc, genes_to_predict, n_neighbors)
    elif method == "OT":
        return _get_impute_OT(adata_sp, adata_sc, ot_mapping, genes_to_predict, n_neighbors)
    else:
        raise ValueError("method must be 'NearestNeighbors' or 'MNN'")


def _get_impute_OT(
        adata_sp,
        adata_sc,
        ot_mapping,
        genes_to_predict=None,
        n_neighbors=50,
        mode='mean'
):
    if "X_shared" not in adata_sp.obsm or "X_shared" not in adata_sc.obsm:
        if "X_shared_sc" in adata_sp.obsm and "X_shared_sc" in adata_sc.obsm:
            adata_sp.obsm["X_shared"] = adata_sp.obsm["X_shared_sc"]
            adata_sc.obsm["X_shared"] = adata_sc.obsm["X_shared_sc"]
        else:
            raise ValueError("Run get_common_embedding first or ensure X_shared is available!")

    if genes_to_predict is None:
        genes_to_predict = adata_sc.var_names
    else:
        genes_to_predict = [g for g in genes_to_predict if g in adata_sc.var_names]
    print(f"number of genes to predict: {len(genes_to_predict)}")

    if len(genes_to_predict) == 0:
        raise ValueError("genes_to_predict is empty!")

    sc_view = adata_sc[:, genes_to_predict]
    if sparse.issparse(sc_view.X):
        sc_expr = sc_view.X.tocsr().astype(np.float32)
    else:
        sc_expr = sparse.csr_matrix(np.asarray(sc_view.X, dtype=np.float32))

    n_spots, n_cells = ot_mapping.shape
    if n_cells == 0 or n_spots == 0:
        raise ValueError("ot_mapping 对齐后为空，请检查 spots / cells 名称是否匹配。")

    k = int(min(max(n_neighbors, 1), n_cells))
    M = ot_mapping.to_numpy(dtype=np.float32, copy=False)
    np.nan_to_num(M, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    M[M < 0] = 0.0

    imputed = np.zeros((n_spots, len(genes_to_predict)), dtype=np.float32)
    used_count = np.zeros(n_cells, dtype=int)
    used_weight_sum = np.zeros(n_cells, dtype=np.float32)

    for i in tqdm(range(n_spots)):
        row = M[i]
        topk = np.argpartition(row, -k)[-k:]
        topk = topk[row[topk] > 0]

        if topk.size == 0:
            fallback_idx = np.argmax(row)
            topk = np.array([fallback_idx])
            weights = np.array([1.0])
            imputed[i] = sc_expr[topk].mean(axis=0)
            used_count[fallback_idx] += 1
            used_weight_sum[fallback_idx] += 1.0
            continue

        expr_subset = sc_expr[topk]

        if mode == 'mean':
            imputed[i] = expr_subset.mean(axis=0)
            weights = np.ones(topk.size)
        elif mode == 'weighted':
            weights = row[topk]
            weights = weights / weights.sum()
            imputed[i] = weights @ expr_subset
        elif mode == 'knn_weighted':
            weights = row[topk]
            w = 1 - weights / np.sum(weights)
            denom = max(1, len(w) - 1)  # avoid div by zero
            weights = w / denom
            imputed[i] = weights @ expr_subset
        else:
            raise ValueError("mode must be 'weighted', 'mean', or 'knn_weighted'")

        for idx, w in zip(topk, weights):
            used_count[idx] += 1
            used_weight_sum[idx] += w

    used_weight_mean = np.divide(
        used_weight_sum, used_count,
        out=np.zeros_like(used_weight_sum),
        where=used_count > 0
    )

    cell_types = adata_sc.obs['Level1'].values if 'Level1' in adata_sc.obs.columns else np.array(['NA'] * n_cells)
    df_stat = pd.DataFrame({
        'cell_type': cell_types,
        'used_count': used_count,
        'used_weight_sum': used_weight_sum,
        'used_weight_mean': used_weight_mean,
    }, index=adata_sc.obs_names)

    adata_imputed = sc.AnnData(
        X=imputed,
        obs=adata_sp.obs.copy(),
        var=pd.DataFrame(index=pd.Index(genes_to_predict))
    )
    adata_imputed.var_names = pd.Index(genes_to_predict)

    return adata_imputed, df_stat


def _get_impute_NearestNeighbors(
    adata_sp,
    adata_sc,
    genes_to_predict=None,
    n_neighbors=50
):
    if "X_shared" not in adata_sp.obsm or "X_shared" not in adata_sc.obsm:
        if "X_shared_sc" in adata_sp.obsm and "X_shared_sc" in adata_sc.obsm:
            adata_sp.obsm["X_shared"] = adata_sp.obsm["X_shared_sc"]
            adata_sc.obsm["X_shared"] = adata_sc.obsm["X_shared_sc"]
        else:
            raise ValueError("Run get_common_embedding first or ensure X_shared is available!")

    if genes_to_predict is None:
        genes_to_predict = adata_sc.var_names

    sc_data = pd.DataFrame(
        adata_sc.X.toarray(),
        index=adata_sc.obs_names,
        columns=adata_sc.var_names
    )

    imputed_data = pd.DataFrame(
        np.zeros((adata_sp.n_obs, len(genes_to_predict))),
        index=adata_sp.obs_names,
        columns=genes_to_predict
    )

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(adata_sc.obsm["X_shared"])
    distances, indices = nbrs.kneighbors(adata_sp.obsm["X_shared"])

    arr = sc_data[genes_to_predict].values  # shape: (n_sc_cells, n_genes_to_predict)
    imputed_arr = np.zeros((adata_sp.n_obs, len(genes_to_predict)))

    # 新增：统计每个sc cell被用的次数和权重和
    n_sc_cells = adata_sc.n_obs
    used_count = np.zeros(n_sc_cells, dtype=int)
    used_weight_sum = np.zeros(n_sc_cells, dtype=float)

    for i in tqdm(range(adata_sp.n_obs), desc="Imputing with NearestNeighbors"):
        valid_dist = distances[i, distances[i] < 1]
        valid_indices = indices[i, distances[i] < 1]
        if len(valid_dist) == 0:
            valid_indices = indices[i, :1]
            imputed_arr[i, :] = arr[valid_indices].mean(axis=0)
            # fallback时，均匀分配权重
            for idx in valid_indices:
                used_count[idx] += 1
                used_weight_sum[idx] += 1.0
            continue
        weights = 1 - valid_dist / np.sum(valid_dist)
        weights = weights / (len(weights) - 1)
        imputed_arr[i, :] = np.dot(weights, arr[valid_indices])
        # 统计被用次数和权重
        for idx, w in zip(valid_indices, weights):
            used_count[idx] += 1
            used_weight_sum[idx] += w

    # 统计每个cell type的分布
    cell_types = adata_sc.obs['Level1'].values
    df_stat = pd.DataFrame({
        'cell_type': cell_types,
        'used_count': used_count,
        'used_weight_sum': used_weight_sum,
        'used_weight_mean': np.divide(used_weight_sum, used_count, out=np.zeros_like(used_weight_sum), where=used_count>0)
    }, index=adata_sc.obs_names)

    imputed_data = pd.DataFrame(imputed_arr, index=adata_sp.obs_names, columns=genes_to_predict)

    adata_imputed = sc.AnnData(imputed_data, obs=adata_sp.obs.copy())
    adata_imputed.var_names = genes_to_predict

    return adata_imputed, df_stat

def _get_impute_MNN(
    adata_sp: sc.AnnData,
    adata_sc: sc.AnnData,
    genes_to_predict: list = None,
    n_neighbors: int = 50,
    fallback_k: int = 5  # 新增参数，默认5
):
    if "X_shared" not in adata_sp.obsm or "X_shared" not in adata_sc.obsm:
        if "X_shared_sc" in adata_sp.obsm and "X_shared_sc" in adata_sc.obsm:
            adata_sp.obsm["X_shared"] = adata_sp.obsm["X_shared_sc"]
            adata_sc.obsm["X_shared"] = adata_sc.obsm["X_shared_sc"]
        else:
            raise ValueError("Run get_common_embedding first or ensure X_shared is available!")

    if genes_to_predict is None:
        genes_to_predict = adata_sc.var_names

    sc_data = pd.DataFrame(
        adata_sc.X.toarray(),
        index=adata_sc.obs_names,
        columns=adata_sc.var_names
    )

    arr = sc_data[genes_to_predict].values  # shape: (n_sc_cells, n_genes_to_predict)
    imputed_arr = np.zeros((adata_sp.n_obs, len(genes_to_predict)), dtype=np.float32)

    nbrs_sp = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(adata_sp.obsm["X_shared"])
    nbrs_sc = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(adata_sc.obsm["X_shared"])

    _, indices_sp2sc = nbrs_sc.kneighbors(adata_sp.obsm["X_shared"])
    _, indices_sc2sp = nbrs_sp.kneighbors(adata_sc.obsm["X_shared"])

    for i, sp_neighbors in tqdm(enumerate(indices_sp2sc), desc="Imputing with MNN"):
        mnn = []
        for j in sp_neighbors:
            if i in indices_sc2sp[j]:
                mnn.append(j)
        if not mnn:
            # fallback: 用前 fallback_k 个最近邻的均值
            fallback_indices = sp_neighbors[:fallback_k]
            imputed_arr[i, :] = arr[fallback_indices].mean(axis=0)
            continue
        imputed_arr[i, :] = arr[mnn].mean(axis=0)

    imputed_data = pd.DataFrame(imputed_arr, index=adata_sp.obs_names, columns=genes_to_predict)

    if imputed_data.isnull().values.any():
        print("Warning: imputed_data contains NaN, will fill with 0")
        imputed_data = imputed_data.fillna(0)
    imputed_data = imputed_data.astype(np.float32)

    adata_imputed = sc.AnnData(imputed_data, obs=adata_sp.obs.copy())
    adata_imputed.var_names = genes_to_predict

    return adata_imputed, None
