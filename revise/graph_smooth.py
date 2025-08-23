import scanpy as sc
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from scipy import sparse


def get_expression_graph(adata, n_pca=30, n_neighbors=15):
    """

    Args:
        adata:
        n_pca:
        n_neighbors:

    Returns:

    """
    adata = adata.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    print(f"data of building expression graph: {adata.shape}")
    sc.pp.pca(adata, n_comps=n_pca)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
    print("Graph of sc expression are saved in obsp['connectivities'].")
    return adata

def get_spatial_graph(
    adata,
    n_neighbors=15,
    weight_mode="gaussian",   # 'gaussian' | 'inverse' | 'linear'
    bandwidth=None,           # 高斯核带宽；默认用每个点第k邻居距离的中位数
    symmetric="mean",         # 'max' | 'mean' | 'sum'  对称化策略
    row_normalize=True,       # 是否把每行归一化为和为1（随机游走归一化）
    save_key_prefix="spatial", # obsp 保存的前缀
    use_cell_size=False,      # 是否考虑细胞大小影响
    sc_pixel=None,               # 单细胞直径，单位为像素
):
    """

    Args:
        adata:
        n_neighbors:
        weight_mode:
        bandwidth:
        symmetric:
        row_normalize:
        save_key_prefix:
        use_cell_size:
        sc_pixel:
    Returns:

    """
    if 'spatial' not in adata.obsm:
        raise ValueError("`adata_sp.obsm['spatial']` does not exist. Please ensure spatial coordinates are available.")

    adata = adata.copy()
    coords = adata.obsm['spatial'].copy()
    n = adata.n_obs

    if use_cell_size and sc_pixel:
        kdtree = KDTree(coords)
        neighbors = kdtree.query_ball_point(coords, sc_pixel)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coords)
    distances, indices = nbrs.kneighbors(coords)  # shape: (n, k)

    if weight_mode == "gaussian":
        # calculate bandwidth if not provided
        if bandwidth is None:
            kth = distances[:, -1]
            bandwidth = np.median(kth[kth > 0]) if np.any(kth > 0) else 1.0
        print(f"Using Gaussian kernel with bandwidth h = {bandwidth:.4f}")
        # w_ij = exp( - d_ij^2 / (2 * h^2) )
        weights = np.exp(-(distances ** 2) / (2 * (bandwidth ** 2)))
    elif weight_mode == "inverse":
        # w_ij = 1 / (d_ij + eps)
        eps = 1e-8
        weights = 1.0 / (distances + eps)
    elif weight_mode == "linear":
        # w_ij = max(0, 1 - d_ij / d_k(i))
        dk = distances[:, -1][:, None] + 1e-8
        weights = np.maximum(0.0, 1.0 - distances / dk)
    else:
        raise ValueError("weight_mode have to be in ('gaussian', 'inverse', 'linear')")

    # remove self-loops by setting self-weights to 0
    self_mask = (indices == np.arange(n)[:, None])
    weights = np.where(self_mask, 0.0, weights)

    # sparse adjacency matrix
    row_idx = np.repeat(np.arange(n), n_neighbors)
    col_idx = indices.flatten()
    data = weights.flatten()

    # filter out zero entries (if any)
    nz_mask = data > 0
    adj = sparse.csr_matrix((data[nz_mask], (row_idx[nz_mask], col_idx[nz_mask])), shape=(n, n))

    # save raw distances
    adata.obsp[f"{save_key_prefix}_distances"] = sparse.csr_matrix(
        (distances.flatten()[nz_mask], (row_idx[nz_mask], col_idx[nz_mask])),
        shape=(n, n)
    )

    # symmetrize the adjacency matrix
    if symmetric == "max":
        adj_sym = adj.maximum(adj.T)
    elif symmetric == "mean":
        adj_sym = (adj + adj.T) * 0.5
    elif symmetric == "sum":
        adj_sym = adj + adj.T
    else:
        raise ValueError("symmetric should be 'max' | 'mean' | 'sum'")

    # row normalize
    if row_normalize:
        row_sum = np.asarray(adj_sym.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        D_inv = sparse.diags(1.0 / row_sum)
        adj_out = D_inv @ adj_sym
    else:
        adj_out = adj_sym

    adata.obsp[f"{save_key_prefix}_connectivities"] = adj_out
    print(
        f"Spatial graph（symmetric={symmetric}, row_normalize={row_normalize}） has been computed and saved in "
        f"obsp['{save_key_prefix}_connectivities']"
    )
    return adata


def laplacian_smooth_expression(adata, obsp_key='connectivities', layer_key='lap_smooth', alpha=0.5):
    ad = adata.copy()

    if obsp_key not in ad.obsp:
        raise KeyError(f"'{obsp_key}' not found in adata.obsp")

    print(f"laplacian smoothing of expression using graph in obsp['{obsp_key}'], alpha={alpha}")
    A = ad.obsp[obsp_key]
    if not sparse.issparse(A):
        raise TypeError(f"adata.obsp['{obsp_key}'] must be a sparse matrix")
    A = A.tocsr()

    X = ad.X
    if sparse.issparse(X):
        Xs = X.tocsr()
    else:
        Xs = sparse.csr_matrix(np.asarray(X))

    # D^{-1} A, D_ii = sum_j A_ij
    deg = np.asarray(A.sum(axis=1)).ravel()

    inv_deg = np.empty_like(deg)
    mask = deg != 0
    inv_deg[mask] = 1.0 / deg[mask]
    inv_deg[~mask] = 1.0

    # D^{-1}A
    DinvA = A.multiply(inv_deg[:, None])  # 仍为 CSR

    AX = DinvA @ Xs
    X_smooth = (1.0 - alpha) * Xs + alpha * AX

    ad.layers[layer_key] = X_smooth.tocsr()

    return ad
