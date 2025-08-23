import numpy as np
import scipy.sparse as sp
import scanpy as sc

from scipy.spatial import KDTree
from tqdm import tqdm
from sklearn.decomposition import PCA


def replace_effect_spots_only_confidence(adata, sc_ref, celltype_col="Level1", no_effect_col="no_effect", n_search_neighbors=10,
                         confidence_threshold=0.9, high_confidence_adjustment="rank_matching"):
    """
    Replace gene expression of cells where no_effect_col is False with the mean expression
    of n_search_neighbors nearest cells of the same cell type.

    Parameters:
    - adata: AnnData object containing spatial data and gene expression
    - sc_ref: single cell reference expression data
    - celltype_col: Column name in adata.obs for cell type annotations (default: "Level1")
    - no_effect_col: Column name in adata.obs indicating cells to replace (default: "no_effect")
    - n_search_neighbors: Number of nearest neighbors to consider (default: 10)

    Returns:
    - adata: Modified AnnData object with updated .X for specified cells
    """
    # Ensure adata.X is a dense array for modification
    gene_intersect = list(set(sc_ref.columns.tolist()) & set(adata.var.index.tolist()))
    adata = adata[:, gene_intersect]

    if not isinstance(adata.X, np.ndarray):
        expr = adata[:, gene_intersect].X.toarray()
        expr_ref = adata[:, gene_intersect].X.toarray()
    else:
        expr = adata[:, gene_intersect].X
        expr_ref = adata[:, gene_intersect].X
    expr_to_adjust = expr.copy()

    # Extract spatial coordinates and relevant columns
    spot_coords = adata.obsm['spatial']
    cell_types = adata.obs[celltype_col].values
    no_effect = adata.obs[no_effect_col].values

    # Build KDTree for spatial coordinates
    kdtree = KDTree(spot_coords)

    # filter high confidence cells
    confidence_values = adata.obs['Confidence'].values
    cell_idx_high_confidence = np.where(confidence_values >= confidence_threshold)[0]

    cell_neighbor_different_cell_type_count = 0
    for cell_idx in tqdm(cell_idx_high_confidence, desc="High confidence adjustment"):
        cell_type = cell_types[cell_idx]
        cell_coord = spot_coords[cell_idx].reshape(1, -1)
        distances, neighbor_indices = kdtree.query(cell_coord, k=n_search_neighbors + 1)
        neighbor_cell_type = np.array(cell_types[neighbor_indices[0]].tolist())
        different_cell_type_in_neighbors = np.where(neighbor_cell_type != cell_type)[0]

        # expression of the center cell and its neighbors
        center_sp_exp = expr_ref[cell_idx]
        neighbor_sp_exp = expr_ref[neighbor_indices[0][different_cell_type_in_neighbors]]

        # sc_ref expression of the center cell and its neighbors
        center_cell_type_sc_ref = sc_ref.loc[cell_type]
        neighbor_different_cell_type_sc_ref = sc_ref.loc[neighbor_cell_type[different_cell_type_in_neighbors]]

        if len(different_cell_type_in_neighbors) > 0:
            # TODO: 通过对比center/neighbor sp expression的差异与sc_ref center/neighbor expression的差异，来判断哪些gene需要矫正，并矫正center_sp_exp
            # Compute expression difference between sp center and neighbors
            # sp_diff = np.mean(neighbor_sp_exp, axis=0) - center_sp_exp
            eps = 1e-6
            sp_diff = np.log2((np.mean(neighbor_sp_exp, axis=0) + eps) / (center_sp_exp + eps))

            # Compute expression difference between reference neighbor and center
            # ref_diff = np.mean(neighbor_different_cell_type_sc_ref.values, axis=0) - center_cell_type_sc_ref.values
            ref_diff = np.log2((np.mean(neighbor_different_cell_type_sc_ref.values, axis=0) + eps) / (
                    center_cell_type_sc_ref.values + eps))

            # Use relative ranking to evaluate direction consistency while avoiding the effect of zeros
            ref_order = np.argsort(ref_diff)
            sp_order = np.argsort(sp_diff)
            rank_corr = np.zeros_like(ref_diff, dtype=float)
            rank_corr[ref_order] = np.arange(len(ref_diff))
            rank_corr[sp_order] -= np.arange(len(sp_diff))
            direction_agree = np.abs(rank_corr) < (0.5 * len(ref_diff))  # allow some tolerance in ranking

            # Genes to correct: based on direction disagreement only
            gene_mask = ~direction_agree

            # Perform relative rank-based correction: shift selected genes toward the local spatial neighbor mean
            corrected_values = center_sp_exp.copy()

            if high_confidence_adjustment == "mean":
                # Replace with mean of neighbors for genes that disagree in direction
                corrected_values[gene_mask] = np.mean(neighbor_sp_exp[:, gene_mask], axis=0)
            elif high_confidence_adjustment == "distance_weighted":
                # 获取不同 cell type 的空间邻居及其距离
                neighbor_distances = distances[0][different_cell_type_in_neighbors]
                weights = 1 / (neighbor_distances + 1e-6)
                weights /= weights.sum()

                # 加权平均表达
                weighted_expr = np.average(neighbor_sp_exp, axis=0, weights=weights)
                corrected_values[gene_mask] = weighted_expr[gene_mask]
            elif high_confidence_adjustment == "rank_matching":
                # Adjust expression based on log fold-change ratio alignment to reference
                adjusted_expr = center_sp_exp.copy()
                # import pdb; pdb.set_trace()
                adjusted_expr[np.newaxis, gene_mask] = (np.sum(
                    np.concatenate([center_sp_exp[np.newaxis, gene_mask], neighbor_sp_exp[:, gene_mask]], axis=0),
                    axis=0)[0] * center_cell_type_sc_ref.values[np.newaxis, gene_mask]) / np.sum(np.concatenate(
                    [center_cell_type_sc_ref.values[np.newaxis, gene_mask],
                     neighbor_different_cell_type_sc_ref.values[:, gene_mask]], axis=0), axis=0)
                adjusted_expr = np.nan_to_num(adjusted_expr, nan=0.0, posinf=0.0, neginf=0.0)
                corrected_values = adjusted_expr
            elif high_confidence_adjustment == "similarity_projection":
                # 构造一个参考表达差向量方向
                ref_vector = ref_diff / (np.linalg.norm(ref_diff) + 1e-6)

                # 将当前表达投影到该方向
                projection = np.dot(sp_diff, ref_vector) * ref_vector

                # 调整表达
                corrected_values[gene_mask] = center_sp_exp[gene_mask] + projection[gene_mask]
            else:
                raise NotImplementedError(
                    f"High confidence adjustment {high_confidence_adjustment} is not implemented.")

            # Apply correction
            expr_to_adjust[cell_idx] = corrected_values
            cell_neighbor_different_cell_type_count += 1

    adata.X = sp.csr_matrix(expr_to_adjust)
    print(
        f"Number of high-confidence cells with at least one neighbor of a different cell type: {cell_neighbor_different_cell_type_count}")
    return adata


def replace_effect_spots_with_adjust(adata, sc_ref, celltype_col="Level1", no_effect_col="no_effect", n_search_neighbors=10, confidence_threshold=0.9, high_confidence_adjustment="rank_matching"):
    """
    Replace gene expression of cells where no_effect_col is False with the mean expression
    of n_search_neighbors nearest cells of the same cell type.
    
    Parameters:
    - adata: AnnData object containing spatial data and gene expression
    - sc_ref: single cell reference expression data
    - celltype_col: Column name in adata.obs for cell type annotations (default: "Level1")
    - no_effect_col: Column name in adata.obs indicating cells to replace (default: "no_effect")
    - n_search_neighbors: Number of nearest neighbors to consider (default: 10)
    
    Returns:
    - adata: Modified AnnData object with updated .X for specified cells
    """
    # Ensure adata.X is a dense array for modification
    gene_intersect = list(set(sc_ref.columns.tolist()) & set(adata.var.index.tolist()))
    adata = adata[:, gene_intersect]

    if not isinstance(adata.X, np.ndarray):
        expr = adata[:, gene_intersect].X.toarray()
        expr_ref = adata[:, gene_intersect].X.toarray()
    else:
        expr = adata[:, gene_intersect].X
        expr_ref = adata[:, gene_intersect].X
    expr_to_adjust = expr.copy()

    # Extract spatial coordinates and relevant columns
    spot_coords = adata.obsm['spatial']
    cell_types = adata.obs[celltype_col].values
    no_effect = adata.obs[no_effect_col].values
    
    # Build KDTree for spatial coordinates
    kdtree = KDTree(spot_coords)
    
    # Get unique cell types
    unique_cell_types = np.unique(cell_types)
    
    # Identify cells to replace (where no_effect is False)
    cells_to_replace = np.where(~no_effect)[0]

    # filter high confidence cells
    confidence_values = adata.obs['Confidence'].values
    cell_idx_high_confidence = np.where(confidence_values >= confidence_threshold)[0]

    cell_neighbor_different_cell_type_count = 0
    for cell_idx in tqdm(cell_idx_high_confidence, desc="High confidence adjustment"):
        cell_type = cell_types[cell_idx]
        cell_coord = spot_coords[cell_idx].reshape(1, -1)
        distances, neighbor_indices = kdtree.query(cell_coord, k=n_search_neighbors + 1)
        neighbor_cell_type = np.array(cell_types[neighbor_indices[0]].tolist())
        different_cell_type_in_neighbors = np.where(neighbor_cell_type != cell_type)[0]

        # expression of the center cell and its neighbors
        center_sp_exp = expr_ref[cell_idx]
        neighbor_sp_exp = expr_ref[neighbor_indices[0][different_cell_type_in_neighbors]]

        # sc_ref expression of the center cell and its neighbors
        center_cell_type_sc_ref = sc_ref.loc[cell_type]
        neighbor_different_cell_type_sc_ref = sc_ref.loc[neighbor_cell_type[different_cell_type_in_neighbors]]

        if len(different_cell_type_in_neighbors) > 0:
            # TODO: 通过对比center/neighbor sp expression的差异与sc_ref center/neighbor expression的差异，来判断哪些gene需要矫正，并矫正center_sp_exp
            # Compute expression difference between sp center and neighbors
            # sp_diff = np.mean(neighbor_sp_exp, axis=0) - center_sp_exp
            eps = 1e-6
            sp_diff = np.log2((np.mean(neighbor_sp_exp, axis=0) + eps) / (center_sp_exp + eps))

            # Compute expression difference between reference neighbor and center
            # ref_diff = np.mean(neighbor_different_cell_type_sc_ref.values, axis=0) - center_cell_type_sc_ref.values
            ref_diff = np.log2((np.mean(neighbor_different_cell_type_sc_ref.values, axis=0) + eps) / (
                        center_cell_type_sc_ref.values + eps))

            # Use relative ranking to evaluate direction consistency while avoiding the effect of zeros
            ref_order = np.argsort(ref_diff)
            sp_order = np.argsort(sp_diff)
            rank_corr = np.zeros_like(ref_diff, dtype=float)
            rank_corr[ref_order] = np.arange(len(ref_diff))
            rank_corr[sp_order] -= np.arange(len(sp_diff))
            direction_agree = np.abs(rank_corr) < (0.5 * len(ref_diff))  # allow some tolerance in ranking

            # Genes to correct: based on direction disagreement only
            gene_mask = ~direction_agree

            # Perform relative rank-based correction: shift selected genes toward the local spatial neighbor mean
            corrected_values = center_sp_exp.copy()

            if high_confidence_adjustment == "mean":
                # Replace with mean of neighbors for genes that disagree in direction
                corrected_values[gene_mask] = np.mean(neighbor_sp_exp[:, gene_mask], axis=0)
            elif high_confidence_adjustment == "distance_weighted":
                # 获取不同 cell type 的空间邻居及其距离
                neighbor_distances = distances[0][different_cell_type_in_neighbors]
                weights = 1 / (neighbor_distances + 1e-6)
                weights /= weights.sum()

                # 加权平均表达
                weighted_expr = np.average(neighbor_sp_exp, axis=0, weights=weights)
                corrected_values[gene_mask] = weighted_expr[gene_mask]
            elif high_confidence_adjustment == "rank_matching":
                # Adjust expression based on log fold-change ratio alignment to reference
                adjusted_expr = center_sp_exp.copy()
                # import pdb; pdb.set_trace()
                adjusted_expr[np.newaxis, gene_mask] = (np.sum(np.concatenate([center_sp_exp[np.newaxis, gene_mask], neighbor_sp_exp[:, gene_mask]], axis=0), axis=0)[0] * center_cell_type_sc_ref.values[np.newaxis, gene_mask]) / np.sum(np.concatenate([center_cell_type_sc_ref.values[np.newaxis, gene_mask], neighbor_different_cell_type_sc_ref.values[:, gene_mask]], axis=0), axis=0)
                adjusted_expr = np.nan_to_num(adjusted_expr, nan=0.0)
                corrected_values = adjusted_expr
            elif high_confidence_adjustment == "similarity_projection":
                # 构造一个参考表达差向量方向
                ref_vector = ref_diff / (np.linalg.norm(ref_diff) + 1e-6)

                # 将当前表达投影到该方向
                projection = np.dot(sp_diff, ref_vector) * ref_vector

                # 调整表达
                corrected_values[gene_mask] = center_sp_exp[gene_mask] + projection[gene_mask]
            else:
                raise NotImplementedError(f"High confidence adjustment {high_confidence_adjustment} is not implemented.")

            # Apply correction
            expr_to_adjust[cell_idx] = corrected_values
            cell_neighbor_different_cell_type_count += 1

    print(f"Number of high-confidence cells with at least one neighbor of a different cell type: {cell_neighbor_different_cell_type_count}")

    for cell_idx in tqdm(cells_to_replace, desc="Replacing cell expressions"):
        cell_type = cell_types[cell_idx]
        
        # Get indices of cells with the same cell type
        same_type_mask = cell_types == cell_type
        same_type_indices = np.where(same_type_mask)[0]
        
        if len(same_type_indices) <= 1:  # Only the cell itself, no other neighbors
            continue
        
        # Get coordinates of the current cell
        cell_coord = spot_coords[cell_idx].reshape(1, -1)
        
        # Query nearest neighbors
        _, neighbor_indices = kdtree.query(cell_coord, k=n_search_neighbors + 1, workers=4)  # +1 to include self
        
        # Filter neighbors to those of the same cell type, excluding self
        # valid_indices = [idx for idx in neighbor_indices[0] if idx in same_type_indices and idx != cell_idx]
        valid_indices = [idx for idx in neighbor_indices[0] if idx in same_type_indices]
        
        if len(valid_indices) == 0:  # No valid neighbors found
            continue
        
        # Limit to requested number of neighbors
        valid_indices = valid_indices[:n_search_neighbors]
        
        # Get expression data for selected neighbors
        neighbor_X = expr_to_adjust[valid_indices]
        
        # Compute mean expression
        mean_expr = np.mean(neighbor_X, axis=0)
        
        # Replace the cell's expression
        adata.X[cell_idx] = np.round(mean_expr)
    # Update AnnData.X with the modified expression matrix
    adata.X = sp.csr_matrix(adata.X)
    
    return adata


def replace_effect_spots(adata, celltype_col="Level1", no_effect_col="no_effect", n_search_neighbors=10):
    """
    Replace gene expression of cells where no_effect_col is False with the mean expression
    of n_search_neighbors nearest cells of the same cell type.

    Parameters:
    - adata: AnnData object containing spatial data and gene expression
    - sc_ref: single cell reference expression data
    - celltype_col: Column name in adata.obs for cell type annotations (default: "Level1")
    - no_effect_col: Column name in adata.obs indicating cells to replace (default: "no_effect")
    - n_search_neighbors: Number of nearest neighbors to consider (default: 10)

    Returns:
    - adata: Modified AnnData object with updated .X for specified cells
    """
    # Ensure adata.X is a dense array for modification

    if not isinstance(adata.X, np.ndarray):
        expr = adata.X.toarray()
    else:
        expr = adata.X

    # Extract spatial coordinates and relevant columns
    spot_coords = adata.obsm['spatial']
    cell_types = adata.obs[celltype_col].values
    no_effect = adata.obs[no_effect_col].values

    # Build KDTree for spatial coordinates
    kdtree = KDTree(spot_coords)

    # Identify cells to replace (where no_effect is False)
    cells_to_replace = np.where(~no_effect)[0]

    expr_list = []
    cell_id_list = []
    for cell_idx in tqdm(cells_to_replace, desc="Replacing cell expressions"):
        cell_type = cell_types[cell_idx]

        # step1_time = time.time()
        # Get indices of cells with the same cell type
        same_type_mask = cell_types == cell_type
        same_type_indices = np.where(same_type_mask)[0]

        if len(same_type_indices) <= 1:  # Only the cell itself, no other neighbors
            continue

        # Get coordinates of the current cell
        cell_coord = spot_coords[cell_idx].reshape(1, -1)
        # step2_time = time.time()
        # print(f"Cell {cell_idx} - Same-type filtering time: {step2_time - step1_time:.4f} s")
        # Query nearest neighbors
        _, neighbor_indices = kdtree.query(cell_coord, k=n_search_neighbors + 1)  # +1 to include self
        # step3_time = time.time()
        # print(f"Cell {cell_idx} - KDTree query time: {step3_time - step2_time:.4f} s")
        # Filter neighbors to those of the same cell type, excluding self
        valid_indices = [idx for idx in neighbor_indices[0] if idx in same_type_indices and idx != cell_idx]
        # valid_indices = [idx for idx in neighbor_indices[0] if idx in same_type_indices]
        # step4_time = time.time()
        # print(f"Cell {cell_idx} - Neighbor filtering time: {step4_time - step3_time:.4f} s")
        if len(valid_indices) == 0:  # No valid neighbors found
            continue

        # Limit to requested number of neighbors
        valid_indices = valid_indices[:n_search_neighbors]

        # Get expression data for selected neighbors
        neighbor_X = expr[valid_indices]

        # Compute mean expression
        mean_expr = np.mean(neighbor_X, axis=0)
        # step5_time = time.time()
        # print(f"Cell {cell_idx} - Mean expression computation time: {step5_time - step4_time:.4f} s")

        # Replace the cell's expression
        # adata.X[cell_idx] = np.round(mean_expr)
        # adata.X[cell_idx] = mean_expr
        expr_list.append(mean_expr[np.newaxis, :])
        cell_id_list.append(cell_idx)
        # cell_end_time = time.time()
        # print(f"Cell {cell_idx} - Round time: {cell_end_time - step5_time:.4f} s")
    # Update AnnData.X with the modified expression matrix

    expr_arr = np.concatenate(expr_list, axis=0)
    adata.X[cell_id_list] = sp.csr_matrix(np.round(expr_arr))
    # adata.X = sp.csr_matrix(adata.X)

    return adata


def impute_from_query_neighbors(adata, be_queried_idx, query_idx, celltype_col="Level1", n_search_neighbors=10, use_exp_similarity=False, expr_similarity_top_n=5):
    """

    Args:
        adata: base adata to be built using KDTree
        be_queried_idx: filter idx to be queried from
        query_idx: cell idx to query for neighbors
        celltype_col:
        n_search_neighbors:
        use_exp_similarity: whether to use expression similarity for neighbor selection
        expr_similarity_top_n: cell number to combine based on expression similarity

    Returns:

    """

    # 1. 在函数开头定义统计容器
    from collections import defaultdict
    valid_neighbor_counts = defaultdict(list)

    if not isinstance(adata.X, np.ndarray):
        expr = adata.X.toarray()
    else:
        expr = adata.X
    cell_types = adata.obs[celltype_col].values

    # build KDTree based on spatial coordinates
    spot_coords = adata.obsm['spatial']
    kdtree = KDTree(spot_coords)

    expr_list = []
    cell_id_list = []

    cell_type_to_indices = {ct: np.where(cell_types == ct)[0] for ct in np.unique(cell_types)}

    for cell_idx in tqdm(query_idx, desc="Replacing cell expressions"):
        cell_type = cell_types[cell_idx]

        # same_type_mask = cell_types == cell_type
        # same_type_indices = np.where(same_type_mask)[0]
        same_type_indices = cell_type_to_indices[cell_type]

        if len(same_type_indices) <= 1:  # Only the cell itself, no other neighbors
            continue

        cell_coord = spot_coords[cell_idx].reshape(1, -1)
        distances, neighbor_indices = kdtree.query(cell_coord, k=n_search_neighbors)
        if be_queried_idx is not None:
            valid_indices = [idx for idx in neighbor_indices[0] if idx in same_type_indices and idx in be_queried_idx]
        else:
            valid_indices = [idx for idx in neighbor_indices[0] if idx in same_type_indices]

        # 2. 在判断前添加统计记录
        valid_neighbor_counts[cell_type].append(len(valid_indices))

        if len(valid_indices) == 0:  # No valid neighbors found
            continue
        neighbor_expr = expr[valid_indices]
        if use_exp_similarity:

            # TODO: select the most similar neighbors based on expressions
            # Compute cosine similarity between the query cell and its neighbors
            from sklearn.metrics.pairwise import cosine_similarity
            query_expr = expr[cell_idx].reshape(1, -1)
            neighbor_expr_matrix = expr[valid_indices]
            similarities = cosine_similarity(query_expr, neighbor_expr_matrix)[0]

            # Get indices of top-k most similar neighbors
            top_k = min(expr_similarity_top_n, len(similarities))
            top_indices = np.argsort(similarities)[-top_k:]

            # Compute weighted average using similarity as weights
            weights = similarities[top_indices] + 1e-8
            weights /= weights.sum()
            mean_expr = np.average(neighbor_expr_matrix[top_indices], axis=0, weights=weights)
        else:
            mean_expr = np.mean(neighbor_expr, axis=0)

        expr_list.append(mean_expr[np.newaxis, :])
        cell_id_list.append(cell_idx)

    expr_arr = np.concatenate(expr_list, axis=0)
    expr[cell_id_list] = np.round(expr_arr)

    adata = adata.copy()
    adata.X = sp.csr_matrix(expr)

    print("Average number of valid neighbors per cell type:")
    for ct, counts in valid_neighbor_counts.items():
        avg_count = np.mean(counts)
        print(f"{ct}: {avg_count:.2f}")

    return adata


def impute_from_query_neighbors_by_cell_type(adata, be_queried_idx, query_idx, n_search_neighbors=10, use_exp_similarity=False, expr_similarity_top_n=10):
    """

    Args:
        adata: base adata to be built using KDTree
        be_queried_idx: filter idx to be queried from
        query_idx: cell idx to query for neighbors
        n_search_neighbors:
        use_exp_similarity: whether to use expression similarity for neighbor selection
        expr_similarity_top_n: cell number to combine based on expression similarity

    Returns:

    """

    if not isinstance(adata.X, np.ndarray):
        expr = adata.X.toarray()
    else:
        expr = adata.X

    # build KDTree based on spatial coordinates
    spot_coords = adata.obsm['spatial']
    kdtree = KDTree(spot_coords)

    cell_coord = spot_coords[query_idx]
    query_distances, query_neighbor_indices = kdtree.query(cell_coord, k=n_search_neighbors)

    expr_list = []
    cell_id_list = []

    for idx, cell_idx in tqdm(enumerate(query_idx), desc="Replacing cell expressions"):
        neighbor_indices = query_neighbor_indices[idx]
        if be_queried_idx is not None:
            valid_indices = [idx for idx in neighbor_indices if idx in be_queried_idx]
        else:
            valid_indices = neighbor_indices

        if len(valid_indices) == 0:  # No valid neighbors found
            continue
        neighbor_expr = expr[valid_indices]
        if use_exp_similarity:
            # Compute cosine similarity between the query cell and its neighbors
            from sklearn.metrics.pairwise import cosine_similarity
            query_expr = expr[cell_idx].reshape(1, -1)
            neighbor_expr_matrix = expr[valid_indices]
            similarities = cosine_similarity(query_expr, neighbor_expr_matrix)[0]

            # Get indices of top-k most similar neighbors
            top_k = min(expr_similarity_top_n, len(similarities))
            top_indices = np.argsort(similarities)[-top_k:]

            # Compute weighted average using similarity as weights
            weights = similarities[top_indices] + 1e-8
            weights /= weights.sum()
            mean_expr = np.average(neighbor_expr_matrix[top_indices], axis=0, weights=weights)
        else:
            mean_expr = np.mean(neighbor_expr, axis=0)
            if np.isnan(mean_expr).any():
                print(f"Warning: NaN values found in mean expression for cell {cell_idx}. Using mean of neighbors instead.")

        expr_list.append(mean_expr[np.newaxis, :])
        cell_id_list.append(cell_idx)

    expr_arr = np.concatenate(expr_list, axis=0)
    expr[cell_id_list] = np.round(expr_arr)
    adata = adata.copy()
    adata.X = sp.csr_matrix(expr)

    return adata


def impute_from_query_neighbors_by_cell_type_confidence(adata, query_idx, n_search_neighbors=10, use_exp_similarity=False, expr_similarity_top_n=10, high_confidence_top_ratio=0.3):
    """

    Args:
        high_confidence_top_ratio:
        adata: base adata to be built using KDTree
        query_idx: cell idx to query for neighbors
        n_search_neighbors:
        use_exp_similarity: whether to use expression similarity for neighbor selection
        expr_similarity_top_n: cell number to combine based on expression similarity

    Returns:

    """

    if not isinstance(adata.X, np.ndarray):
        expr = adata.X.toarray()
    else:
        expr = adata.X

    # build KDTree based on spatial coordinates
    spot_coords = adata.obsm['spatial']
    kdtree = KDTree(spot_coords)

    cell_coord = spot_coords[query_idx]
    query_distances, query_neighbor_indices = kdtree.query(cell_coord, k=n_search_neighbors)

    expr_list = []
    cell_id_list = []
    counter = [0] * adata.shape[0]

    for idx, cell_idx in tqdm(enumerate(query_idx), desc="Replacing cell expressions"):
        cell_confidence = adata.obs['Confidence'].values[cell_idx]
        neighbor_indices = query_neighbor_indices[idx]
        valid_indices = neighbor_indices.copy()
        # valid_indices = [idx for idx in neighbor_indices if adata.obs['Confidence'].values[idx] >= cell_confidence]

        if len(valid_indices) == 0:  # No valid neighbors found
            top_k = int(len(neighbor_indices) * high_confidence_top_ratio)
            valid_indices = neighbor_indices[np.argsort([adata.obs['Confidence'].values[idx] for idx in neighbor_indices])[-top_k:]]
        neighbor_expr = expr[valid_indices]

        for valid_idx in valid_indices:
            counter[valid_idx] += 1

        if use_exp_similarity:
            # Compute cosine similarity between the query cell and its neighbors
            from sklearn.metrics.pairwise import cosine_similarity
            query_expr = expr[cell_idx].reshape(1, -1)
            # similarities = cosine_similarity(query_expr, neighbor_expr)[0]
            similarities = np.linalg.norm(neighbor_expr - query_expr, axis=1)

            # Get indices of top-k most similar neighbors
            top_k = min(expr_similarity_top_n, len(similarities))
            top_indices = np.argsort(similarities)[-top_k:]
            mean_expr = np.mean(neighbor_expr[top_indices], axis=0)

            # Compute weighted average using similarity as weights
            # weights = similarities[top_indices] + 1e-8
            # weights /= weights.sum()
            # mean_expr = np.average(neighbor_expr_matrix[top_indices], axis=0, weights=weights)
        else:
            mean_expr = np.mean(neighbor_expr, axis=0)

        expr_list.append(mean_expr[np.newaxis, :])
        cell_id_list.append(cell_idx)

    expr_arr = np.concatenate(expr_list, axis=0)
    expr[cell_id_list] = np.round(expr_arr)
    adata = adata.copy()
    adata.X = sp.csr_matrix(expr)
    adata.obs['valid_neighbor_count'] = counter

    return adata



def impute_from_query_neighbors_by_cell_type_deprecated(adata, be_queried_idx, query_idx, celltype_col="Level1", n_search_neighbors=10, use_exp_similarity=False, expr_similarity_top_n=5):
    """

    Args:
        adata: base adata to be built using KDTree
        be_queried_idx: filter idx to be queried from
        query_idx: cell idx to query for neighbors
        celltype_col:
        n_search_neighbors:
        use_exp_similarity: whether to use expression similarity for neighbor selection
        expr_similarity_top_n: cell number to combine based on expression similarity

    Returns:

    """
    if not isinstance(adata.X, np.ndarray):
        expr = adata.X.toarray()
    else:
        expr = adata.X

    # build KDTree based on spatial coordinates
    spot_coords = adata.obsm['spatial']
    kdtree_spatial = KDTree(spot_coords)
    spatial_query_coords = spot_coords[query_idx]
    spatial_distances, spatial_all_neighbors = kdtree_spatial.query(spatial_query_coords, k=n_search_neighbors)

    if use_exp_similarity:
        pca = PCA(n_components=50)
        X_dense = adata.X.todense()
        X_pca = pca.fit_transform(X_dense)

        if be_queried_idx:
            kdtree_neighbors = KDTree(X_pca[be_queried_idx])
        else:
            kdtree_neighbors = KDTree(X_pca)

        query_coords = X_pca[query_idx]
        expr_distances, expr_all_neighbors = kdtree_neighbors.query(query_coords, k=expr_similarity_top_n)

        # 设置平滑参数，避免除以0
        epsilon = 1e-8

        # 计算加权权重：越近权重越高
        weights = 1.0 / (expr_distances + epsilon)
        weights = weights / weights.sum(axis=1, keepdims=True)  # 行归一化

        if be_queried_idx:
            global_neighbor_indices = np.take(be_queried_idx, expr_all_neighbors)
        else:
            # 没有子集约束，则邻居索引已是全局索引
            global_neighbor_indices = expr_all_neighbors

        neighbor_expr_stack = expr[global_neighbor_indices]  # 如果 X 是 dense，可以直接索引

        # 加权平均
        weighted_expr = np.einsum('ij,ijk->ik', weights, neighbor_expr_stack)
    else:
        neighbor_expr_stack = expr[spatial_all_neighbors]
        weighted_expr = np.mean(neighbor_expr_stack, axis=1)

    expr[query_idx] = weighted_expr
    adata.X = sp.csr_matrix(expr)
    return adata


def impute_from_query_neighbors_by_cell_type_acc(adata, be_queried_idx, query_idx, n_search_neighbors=10, use_exp_similarity=False):
    """
    Impute gene expression for query_idx cells using the mean of their spatial neighbors (from be_queried_idx),
    fully vectorized using scanpy.pp.neighbors and the connectivities graph.
    Assumes all cells are of the same cell type.
    Args:
        adata:
        be_queried_idx:
        query_idx:
        n_search_neighbors:

    Returns:

    """
    if not isinstance(adata.X, np.ndarray):
        expr = adata.X.toarray()
    else:
        expr = adata.X

    n_cells, n_genes = expr.shape
    # Ensure neighbor graph is computed
    if 'connectivities' not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=n_search_neighbors, use_rep='spatial')

    conn = adata.obsp['connectivities'].tocsr(copy=True)
    # Limit neighbors to be_queried_idx only
    if be_queried_idx is not None:
        mask = np.zeros(n_cells, dtype=bool)
        mask[be_queried_idx] = True
        conn[:, ~mask] = 0
        conn.eliminate_zeros()
    # Matrix multiply to get neighbor-averaged expression
    neighbor_expr_sum = conn @ expr
    neighbor_weights = np.array(conn.sum(axis=1)).flatten()[:, None] + 1e-8
    neighbor_expr_mean = neighbor_expr_sum / neighbor_weights
    # Apply imputation only to query_idx
    expr[query_idx] = np.round(neighbor_expr_mean[query_idx])
    adata.X = sp.csr_matrix(expr)

    return adata


def create_meta_cells(adata, sc_pixel, n_neighbors=5, cell_type_col='Level1',
                     bandwidth=None, min_neighbors=3, agg_method='weighted_mean', dist_mode='spatial'):
    """
    通过聚合同类型邻近细胞的表达数据来创建meta cells

    参数:
    - adata: AnnData对象，包含归一化表达矩阵和空间坐标
    - sc_pixel: 空间分辨率，用于定义邻居搜索半径
    - n_neighbors: 最大邻居数量
    - cell_type_col: 细胞类型列名
    - bandwidth: 高斯核带宽，用于空间权重计算（如果为None则自动估计）
    - min_neighbors: 构建meta cell所需的最小邻居数量
    - agg_method: 聚合方法，可选 'weighted_mean' 或 'concat'

    返回:
    - meta_adata: 包含meta cells的新AnnData对象
    """

    sc_pixel = sc_pixel * n_neighbors
    # 复制原始AnnData对象
    meta_adata = adata.copy()

    # 基本参数检查
    if 'spatial' not in adata.obsm:
        raise ValueError("Missing spatial coordinates")
    if cell_type_col not in adata.obs:
        raise ValueError(f"Missing cell type column: {cell_type_col}")

    spatial_coords = adata.obsm['spatial']
    cell_types = adata.obs[cell_type_col].values

    # 获取表达矩阵
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X

    # 自动估计带宽（如果未提供）
    if bandwidth is None:
        # 使用sc_pixel作为参考
        bandwidth = sc_pixel / 2
        print(f"Set bandwidth to sc_pixel/2: {bandwidth:.4f}")

    # 初始化结果数组，保持原始顺序
    if agg_method == 'weighted_mean':
        meta_expressions = X.copy()  # 默认保持原始表达
    else:  # concat模式
        # 初始化为最大可能大小
        meta_expressions = np.zeros((len(adata), n_neighbors + 1, X.shape[1]))
        meta_expressions[:, 0, :] = X  # 第一个位置放原始细胞

    meta_cell_sizes = np.ones(len(adata), dtype=int)  # 默认大小为1
    meta_cell_indices = [[i] for i in range(len(adata))]  # 每个细胞初始只包含自己

    # 按细胞类型处理
    unique_cell_types = np.unique(cell_types)
    for cell_type in tqdm(unique_cell_types, desc="Processing cell types"):
        # 获取当前类型的细胞
        type_mask = cell_types == cell_type
        type_indices = np.where(type_mask)[0]
        type_coords = spatial_coords[type_mask]
        type_expr = X[type_mask]

        # 构建KDTree
        kdtree = KDTree(type_coords)

        # 处理每个细胞
        for local_idx, global_idx in enumerate(type_indices):
            # 在sc_pixel范围内搜索邻居
            neighbors = kdtree.query_ball_point(type_coords[local_idx], sc_pixel)
            # 排除自身
            neighbors = [n for n in neighbors if n != local_idx]

            # 如果邻居数量超过n_neighbors，随机选择n_neighbors个
            if len(neighbors) > n_neighbors:
                neighbors = np.random.choice(neighbors, n_neighbors, replace=False)

            if len(neighbors) < min_neighbors:
                continue  # 保持原始表达值

            # 计算到中心细胞的距离
            if dist_mode == "spatial":
                distances = np.array([np.linalg.norm(type_coords[local_idx] - type_coords[n]) for n in neighbors])
                # 计算空间权重

                weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            elif dist_mode == "expression":
                distances = np.linalg.norm(type_expr[local_idx] - type_expr[neighbors], axis=1)
                weights = distances
            else:
                raise NotImplementedError

            weights = weights / (np.sum(weights) + 1e-6)

            # 获取邻居表达数据
            neighbor_expr = type_expr[neighbors]

            if agg_method == 'weighted_mean':
                # 计算加权平均表达
                meta_expr = np.sum(neighbor_expr * weights[:, None], axis=0)
                meta_expressions[global_idx] = meta_expr

            elif agg_method == 'concat':
                # 填充邻居表达数据
                n_neighbors_actual = len(neighbors)
                meta_expressions[global_idx, 1:n_neighbors_actual+1] = neighbor_expr

            # 更新meta cell信息
            meta_cell_sizes[global_idx] = len(neighbors) + 1
            meta_cell_indices[global_idx].extend([type_indices[n] for n in neighbors])

    # 更新meta_adata
    if agg_method == 'weighted_mean':
        if sp.issparse(adata.X):
            meta_expressions = sp.csr_matrix(meta_expressions)
    else:  # concat模式
        if sp.issparse(adata.X):
            meta_expressions = sp.csr_matrix(meta_expressions.reshape(len(adata), -1))
        else:
            meta_expressions = meta_expressions.reshape(len(adata), -1)

    # 更新表达矩阵
    meta_adata.X = meta_expressions

    # 添加meta cell信息
    meta_adata.obs['meta_cell_size'] = meta_cell_sizes
    meta_adata.uns['meta_cell_indices'] = meta_cell_indices

    # 打印统计信息
    enhanced_cells = np.sum(meta_cell_sizes > 1)
    print(f"Enhanced {enhanced_cells} cells ({enhanced_cells/len(adata)*100:.1f}%) with neighbors")
    print(f"Average meta cell size: {np.mean(meta_cell_sizes):.2f} cells")

    return meta_adata