import pandas as pd
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

def compute_SVC_adata(SVC_obs, sc_ref, adata_st, use_total_counts=True):
    # Normalize reference expression per cell type
    sc_ref_norm = sc_ref.div(sc_ref.sum(axis=1), axis=0)

    all_genes = sc_ref.columns
    spot_names = adata_st.obs_names
    svc_data = []

    for spot in spot_names:
        # 所有属于该 spot 的 cell
        cells_in_spot = SVC_obs[SVC_obs['spot_name'] == spot]
        if cells_in_spot.empty:
            continue

        # 初始化这个 spot 的重建表达矩阵（cell 数 × 基因数）
        expr_cells = []

        # 每个 cell 的初步表达量
        for _, row in cells_in_spot.iterrows():
            cell_id = row['cell_id']
            cell_type = row['true_cell_type']
            total_count = row['total_counts'] if use_total_counts else 1.0
            if cell_type not in sc_ref_norm.index:
                continue
            expr = sc_ref_norm.loc[cell_type] * total_count
            expr_cells.append((cell_id, expr))

        if not expr_cells:
            continue

        # spot-level 基因表达重建（sum over all cells）
        df_cells = pd.DataFrame({cid: expr for cid, expr in expr_cells}).T  # cell_id × gene
        spot_expr_reconstructed = df_cells.sum(axis=0)  # series, gene-level sum

        # spot measured expression（from adata_st）
        spot_index = list(adata_st.obs_names).index(spot)
        spot_expr_measured = adata_st.X[spot_index].toarray().flatten()
        spot_expr_measured = pd.Series(spot_expr_measured, index=all_genes)

        # 计算 fold change
        with np.errstate(divide='ignore', invalid='ignore'):
            fold_change = spot_expr_measured / spot_expr_reconstructed
            fold_change.replace([np.inf, -np.inf], 0, inplace=True)
            fold_change.fillna(0, inplace=True)

        # 用 fold_change 调整每个 cell 的初步表达量
        for cid, expr in expr_cells:
            expr_adjusted = expr * fold_change
            svc_data.append((cid, expr_adjusted))

    # 创建 SVC_adata
    if not svc_data:
        raise ValueError("No valid cells found for reconstruction.")
    
    cell_ids, expr_list = zip(*svc_data)
    expr_matrix = pd.DataFrame(expr_list, index=cell_ids, columns=all_genes)
    SVC_adata = AnnData(X=sp.csr_matrix(expr_matrix.values), obs=pd.DataFrame(index=cell_ids), var=pd.DataFrame(index=all_genes))
    return SVC_adata

use_total_counts=False
use_total_counts=True
SVC_adata = compute_SVC_adata(SVC_obs, sc_ref, adata_st, use_total_counts=use_total_counts)
SVC_adata



from scipy.sparse import issparse, csr_matrix
from scipy.spatial import KDTree

def sp_SVC_recon_sp(adata_st, SVC_obs, n_neighbors=20, cell_type_col='Level1'):
    """
    sp-like Spatial Virtual Cell (SVC) reconstruction with spot-specific sc_ref
    Args:
        adata_st: AnnData object of spatial transcriptomics data, obs['is_pure'] indicates pure spots
        SVC_obs: DataFrame containing 'spot_name', 'cell_id', and 'cell_type' columns
        n_neighbors: Number of nearest pure spots to consider for each cell type
        cell_type_col: Column name in adata_st.obs containing cell type information
    Returns:
        SVC_adata: AnnData object of reconstructed single-cell data
    """
    # Get unique spots and cell types from SVC_obs
    spots = SVC_obs['spot_name'].unique()
    type_list = sorted(SVC_obs['cell_type'].unique())
    total_counts_df = pd.read_csv("/home/wys/Sim2Real-ST/REVISE/WSI_Xenium/cut/total_counts.csv", index_col=0)
    total_counts_df # cell_id	total_counts

    # Create spot to index mapping
    spot_to_idx = {spot: idx for idx, spot in enumerate(spots)}
    
    # Get spot expression data
    adata_spot = adata_st.copy()
    X = adata_spot.X if isinstance(adata_spot.X, np.ndarray) else adata_spot.X.toarray()
    
    # Identify pure spots
    pure_spots = adata_st[adata_st.obs['is_pure']].copy()
    pure_coords = pure_spots.obsm['spatial']
    pure_cell_types = pure_spots.obs[cell_type_col].values
    
    # Build spot-specific sc_ref for each cell type
    spot_specific_refs = {spot: {} for spot in spots}
    
    # Determine required cell types per spot
    spot_cell_types = {spot: set(SVC_obs[SVC_obs['spot_name'] == spot]['true_cell_type']) for spot in spots}
    
    # Build KDTree for each cell type's pure spots and compute mean expressions
    for cell_type in tqdm(type_list, desc="Building spot-specific references"):
        # Get pure spots for this cell type
        type_mask = pure_cell_types == cell_type
        if not np.any(type_mask):
            continue  # Skip if no pure spots for this cell type
        type_coords = pure_coords[type_mask]
        type_X = pure_spots.X[type_mask] if isinstance(pure_spots.X, np.ndarray) else pure_spots.X[type_mask].toarray()
        
        # Build KDTree for this cell type's pure spots
        kdtree = KDTree(type_coords)
        
        # Query n_neighbors nearest pure spots for all relevant spots
        relevant_spots = [spot for spot in spots if cell_type in spot_cell_types[spot]]
        if not relevant_spots:
            continue
        relevant_coords = np.array([adata_st.obs.loc[spot, ["x","y"]] for spot in relevant_spots])
        
        # Find n_neighbors nearest neighbors
        _, neighbor_indices = kdtree.query(relevant_coords, k=n_neighbors)
        
        # Compute mean expression for each spot
        for spot, indices in zip(relevant_spots, neighbor_indices):
            if len(indices) > 0:
                # Ensure indices are valid
                valid_indices = [i for i in indices if i < len(type_X)]
                if valid_indices:
                    mean_expr = np.mean(type_X[valid_indices], axis=0)
                    mean_expr = mean_expr / (np.sum(mean_expr) + 1e-10)  # Normalize
                    spot_specific_refs[spot][cell_type] = mean_expr * total_count
    
    # Initialize Y array for spot-specific references
    Y = np.zeros((len(spots), len(adata_st.var_names), len(type_list)))
    type_idx_map = {t: i for i, t in enumerate(type_list)}
    
    # Fill Y with spot-specific reference profiles
    for spot in spots:
        spot_idx = spot_to_idx[spot]
        for cell_type in spot_cell_types[spot]:
            if cell_type in spot_specific_refs[spot]:
                type_idx = type_idx_map[cell_type]
                Y[spot_idx, :, type_idx] = spot_specific_refs[spot][cell_type]
    
    # Step 2: 求粗估Y的总和（把不同cell type贡献相加）
    Y_total = np.sum(Y, axis=2)  # n_spot × n_gene

    # Step 3: 计算每个 spot、每个 gene 上的倍数关系
    scale = X / (Y_total + 1e-10)  # 避免除零，n_spot × n_gene

    # Step 4: 把这个比例乘回原始Y上（broadcast到cell type）
    Y = Y * scale[:, :, np.newaxis]  # n_spot × n_gene × n_type
    
    # Initialize output array for single-cell expressions
    SVC_X = np.zeros((len(SVC_obs), len(adata_st.var_names)))
    
    # Assign expressions based on SVC_obs
    spot_indices = np.array([spot_to_idx[spot] for spot in SVC_obs['spot_name']])
    type_indices = np.array([type_idx_map[t] for t in SVC_obs['cell_type']])
    
    # Assign normalized expressions to each cell
    for i in range(len(SVC_obs)):
        SVC_X[i, :] = Y[spot_indices[i], :, type_indices[i]]
    
    # Normalize final gene expression
    # print(np.sum(SVC_X, axis=1))
    # SVC_X = np.round(SVC_X)
    SVC_X = SVC_X / (np.sum(SVC_X, axis=1, keepdims=True) + 1e-10) * 1e4
    
    print(f"Number of cells processed: {len(SVC_obs)}")
    print(f"Number of unique spots: {len(spots)}")
    print(f"Shape of SVC_X: {SVC_X.shape}")
    
    # Create new AnnData object
    SVC_adata = sc.AnnData(SVC_X)
    SVC_adata.var_names = adata_st.var_names
    SVC_adata.obs = SVC_obs.copy()
    
    return SVC_adata