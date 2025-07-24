import scanpy as sc
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cosine


import pandas as pd
import numpy as np

def cut_off_low_gene(sc_ref, gene_cutoff_ratio, mode="all"):
    """
    Apply a high-pass filter to set low-expression genes to zero in sc_ref DataFrame
    based on percentile thresholds.
    
    Parameters:
    - sc_ref (pd.DataFrame): DataFrame with cell types as rows and genes as columns.
    - gene_cutoff_ratio (float): Percentile (0-1) to determine the cutoff threshold.
    - mode (str): "all" to use a global percentile threshold across all cell types, 
                  "self" to use per-cell-type percentile thresholds.
    
    Returns:
    - pd.DataFrame: Modified DataFrame with low-expression genes set to zero.
    """
    if not 0 <= gene_cutoff_ratio <= 1:
        raise ValueError("gene_cutoff_ratio must be between 0 and 1")
    
    type_list = sc_ref.index
    sc_ref_modified = sc_ref.copy()  # Create a copy to avoid modifying the original
    
    if mode == "all":
        # Calculate global threshold based on percentile across all values
        threshold = np.percentile(sc_ref.values.flatten(), gene_cutoff_ratio * 100)
        
        # Set values below the threshold to zero
        sc_ref_modified = sc_ref_modified.where(sc_ref_modified >= threshold, 0)
    
    elif mode == "self":
        # Apply threshold individually for each cell type
        for cell_type in type_list:
            # Calculate percentile threshold for this cell type
            threshold = np.percentile(sc_ref_modified.loc[cell_type].values, gene_cutoff_ratio * 100)
            
            # Set values below the threshold to zero for this cell type
            sc_ref_modified.loc[cell_type] = sc_ref_modified.loc[cell_type].where(
                sc_ref_modified.loc[cell_type] >= threshold, 0)
    
    else:
        raise ValueError("Mode must be 'all' or 'self'")
    
    return sc_ref_modified
    

def construct_pure_reference(X, cell_proportion, sc_ref, type_list, threshold=0.95, top_n=3, max_n=10):
    """
    构建纯净的参考矩阵
    Args:
        X: ST数据矩阵 (n_spots, n_genes)
        cell_proportion: 第一次解卷积得到的细胞比例 (n_spots, n_types)
        sc_ref: 原始参考矩阵 (n_types, n_genes)
        type_list: 细胞类型列表
        threshold: 判定为纯净spot的阈值
        top_n: 当某类型没有纯净spot时，选择比例最高的前n个spot
        max_n: 每种细胞类型最多使用的spot数量
    Returns:
        sc_ref_pure: 纯净的参考矩阵 (n_types, n_genes)
        pure_spots_info: 每种细胞类型的spot信息
    """
    n_spots, n_types = cell_proportion.shape
    n_genes = X.shape[1]
    
    # 统计每种细胞类型的纯净spot
    pure_spots_by_type = {t: [] for t in type_list}
    for s in range(n_spots):
        max_val, max_idx = cell_proportion[s].max(), cell_proportion[s].argmax()
        if max_val > threshold:
            pure_spots_by_type[type_list[max_idx]].append(s)
    
    # 构建纯净的参考矩阵和收集信息
    sc_ref_pure = np.zeros((n_types, n_genes))  # 保持与sc_ref相同的维度顺序
    pure_spots_info = []
    
    for i, t in enumerate(type_list):
        spots = pure_spots_by_type[t]
        if len(spots) == 0:
            # 如果没有纯净spot，使用比例最高的top_n个
            proportions = cell_proportion[:, i]
            spots = np.argsort(proportions)[-top_n:]
            source = "top"
        else:
            source = "pure"
            # 如果超过max_n，只取最高的max_n个
            if len(spots) > max_n:
                proportions = cell_proportion[spots, i]
                top_indices = np.argsort(proportions)[-max_n:]
                spots = [spots[idx] for idx in top_indices]
        
        # 计算平均表达谱得到纯净参考
        spot_profiles = X[spots]  # (n_selected_spots, n_genes)
        sc_ref_pure[i] = np.mean(spot_profiles, axis=0)  # (n_genes,)
        
        # 计算两种余弦相似度：
        # 1. 纯净参考与原始参考的相似度
        ref_sim = 1 - cosine(sc_ref_pure[i], sc_ref[i])
        
        # 2. 每个选出的spot与纯净参考的相似度
        spot_sims = []
        for spot_profile in spot_profiles:  # spot_profile shape: (n_genes,)
            sim = 1 - cosine(spot_profile, sc_ref_pure[i])
            spot_sims.append(sim)
        
        # 获取选中spots的细胞比例
        spot_proportions = cell_proportion[spots, i]
        
        # 收集信息
        pure_spots_info.append({
            'cell_type': t,
            'n_spots': len(spots),
            'source': source,
            'ref_similarity': ref_sim,  # 与原始参考的相似度
            'spot_similarities': spot_sims,  # 与纯净参考的相似度数组
            'mean_spot_sim': np.mean(spot_sims),
            'min_spot_sim': np.min(spot_sims),
            'max_spot_sim': np.max(spot_sims),
            'median_spot_sim': np.median(spot_sims),
            'proportions': spot_proportions,  # 选中spots的细胞比例
            'mean_prop': np.mean(spot_proportions),
            'min_prop': np.min(spot_proportions),
            'max_prop': np.max(spot_proportions),
            'median_prop': np.median(spot_proportions)
        })
    
    return sc_ref_pure, pure_spots_info

def preprocess(adata_sc, adata_st):
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.normalize_total(adata_st, target_sum=1e4)

    common_genes = adata_sc.var_names.intersection(adata_st.var_names)
    adata_sc = adata_sc[:, common_genes]
    adata_st = adata_st[:, common_genes]
    
    return adata_sc, adata_st


def marker_selection(adata_sc, key_type, threshold_cover=0.6, threshold_p=0.1,
                     threshold_fold=1.5, n_select=40, verbose=0, return_dict=False, q=0):
    """Find marker genes based on pairwise ratio test.

    Args:
        adata_sc: scRNA data (Anndata).
        key_type: The key that is used to extract cell type information from adata_sc.obs.
        threshold_cover: Minimum proportion of non-zero reads of a marker gene in assigned cell type.
        threshold_p: Maximum p-value for a gene to be marker gene.
        threshold_fold: Minimum fold change for a gene to be marker gene.
        n_select: Number of marker genes selected for each cell type.
        verbose: 0: silent. 1: print the number of marker genes of each cell type.
        return_dict: If true, return a dictionary of marker genes, where the keys are the name of the cell types.
        q: Quantile of the fold-change that we considered.

    Returns:
       List of the marker genes or a dictionary of marker genes, where the keys are the name of the cell types.
    """
    X = adata_sc.X if type(adata_sc.X) is np.ndarray else adata_sc.X.toarray()

    # Derive mean and std matrix
    type_list = sorted(list(adata_sc.obs[key_type].unique()))  # list of the cell type.
    n_gene, n_type = adata_sc.shape[1], len(type_list)
    expression_mu = np.zeros((n_type, n_gene))  # Mean expression of each gene in each cell type.
    expression_sd = np.zeros((n_type, n_gene))  # Standard deviation of expression of each gene in each cell type.
    n_cell_by_type = np.zeros(n_type)
    data_type = []  # The expression data categorized by cell types.
    for i in range(n_type):
        data_type.append(X[adata_sc.obs[key_type] == type_list[i]])
        expression_mu[i] = np.mean(data_type[i], axis=0)
        expression_sd[i] = np.std(data_type[i], axis=0)
        n_cell_by_type[i] = len(data_type[i])
    del X
    expression_sd = expression_sd + 1e-10
    expression_mu = expression_mu + 1e-10

    # t test
    fold_change = np.zeros((n_type, n_gene))
    p_value = np.zeros((n_type, n_gene))
    type_index_max = np.argmax(expression_mu, axis=0)  # Cell type index with the maximum mean expression of each gene
    for i in range(n_gene):
        mu0 = expression_mu[type_index_max[i], i]
        sd0 = expression_sd[type_index_max[i], i]
        n0 = n_cell_by_type[type_index_max[i]]
        A = sd0 ** 2 / n0 + expression_sd[:, i] ** 2 / n_cell_by_type
        B = (sd0 ** 2 / n0)**2/(n0-1) + (expression_sd[:, i] ** 2 / n_cell_by_type)**2/(n_cell_by_type-1)
        t_stat = (mu0 - expression_mu[:, i]) / np.sqrt(A)
        fold_change[:, i] = mu0/expression_mu[:, i]
        df = A**2/B
        p_value[:, i] = stats.t.sf(abs(t_stat), df)

    # determine the marker genes
    p_value_sort = np.sort(p_value, axis=0)
    fold_change_sort = np.sort(fold_change, axis=0)
    gene_name = np.array(adata_sc.var_names)
    marker_gene = dict() if return_dict else []
    for i in range(n_type):
        # fraction of non-zero reads in current datatype
        cover_fraction = np.sum(data_type[i][:, type_index_max == i] > 0, axis=0) / n_cell_by_type[i]
        p_value_temp = p_value_sort[-2, type_index_max == i]  # second-largest p-value
        # fold_change_temp = fold_change_sort[1, type_index_max == i]  # second-smallest fold change
        fold_change_temp = fold_change_sort[max(1, int(np.round(q*(n_type-1)))), type_index_max == i]
        selected = np.logical_and(cover_fraction >= threshold_cover, p_value_temp < threshold_p)
        selected = np.logical_and(fold_change_temp >= threshold_fold, selected)
        gene_name_temp = gene_name[type_index_max == i][selected]

        fold_change_temp = fold_change_temp[selected]
        selected_gene_idx = np.argsort(fold_change_temp)[::-1][:n_select]
        selected_gene = gene_name_temp[selected_gene_idx]

        if return_dict:
            marker_gene[type_list[i]] = list(selected_gene)
        else:
            marker_gene.extend(list(selected_gene))
        if verbose == 1:
            print(f'{type_list[i]}: {len(list(selected_gene)):d}')
        elif len(list(selected_gene)) < 5:
            print(f'Warning: Only {len(list(selected_gene)):d} genes are selected for {type_list[i]}.')
    return marker_gene


def construct_sc_ref(adata_sc, key_type, type_list=None):
    """
    Construct the scRNA reference from scRNA data.

    Args:
        adata_sc: scRNA data.
        key_type: The key that is used to extract cell type information from adata_sc.obs.

    Returns:
        scRNA reference. Numpy assay with dimension n_type*n_gene
    """
    # type_list = sorted(list(adata_sc.obs[key_type].unique()))
    n_gene, n_type = adata_sc.shape[1], len(type_list)
    sc_ref = np.zeros((n_type, n_gene))
    # X = np.array(adata_sc.X)
    X = adata_sc.X if isinstance(adata_sc.X, np.ndarray) else adata_sc.X.toarray()
    for i, cell_type in tqdm(enumerate(type_list)):
        # sc_X_temp = np.sum(X[adata_sc.obs[key_type] == cell_type], axis=0)
        # sc_ref[i] = sc_X_temp/np.sum(sc_X_temp)
        sc_ref[i] = np.mean(X[adata_sc.obs[key_type] == cell_type], axis=0)

    sc_ref = pd.DataFrame(sc_ref, index=type_list, columns=adata_sc.var_names)
    
    return sc_ref



import pandas as pd
from tqdm import tqdm
def check_purity_of_spots(all_cells_in_spot, cell_info, celltype_col):
    """
    检查每个spot中的细胞是否属于单一的Level1类型

    参数:
    all_cells_in_spot (dict): 每个spot中的细胞ID列表
    cell_info (DataFrame): 包含细胞ID和对应的Level1类型

    返回:
    DataFrame: 每个spot是否纯净（单一Level1类型）
    """
    # 初始化结果列表
    results = []
    if celltype_col not in cell_info.columns:
       cell_info[celltype_col] = cell_info['clusters']
    # 遍历每个spot
    for spot, cell_ids in tqdm(all_cells_in_spot.items(), desc = "spots"):
        spot_cell_info = cell_info[cell_info['cell_id'].isin(cell_ids)]
        spot_levels = list(spot_cell_info[celltype_col].unique())

        # 判断是否纯净
        if len(spot_levels) == 1:
            is_pure = True
        else:
            is_pure = False

        # 记录结果
        results.append({
            'spot': spot,
            'is_pure': is_pure,
            f'{celltype_col}_types': spot_levels
        })

    # 转换为DataFrame
    result_df = pd.DataFrame(results)
    print(result_df['is_pure'].value_counts())
    return result_df
