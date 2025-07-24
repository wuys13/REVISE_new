import numpy as np
from scipy.stats import norm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def calculate_morphology_probability(morphology_features, feature_list, mu, sigma, scaled=True):
    """
    E 步：计算形态学概率 PM_on_cell
    Args:
        morphology_features: DataFrame, 形态学特征，(n_cells, n_features)
        feature_list: 形态学特征名称列表
        mu: 均值，(n_types, n_features)
        sigma: 标准差，(n_types, n_features)
        scaled: 是否归一化
    Returns:
        PM_on_cell: 形态学概率，(n_cells, n_types)
    """
    n_cells = morphology_features.shape[0]
    n_types = mu.shape[0]
    PM = np.zeros((n_cells, n_types))

    for t in range(n_types):
        # 计算每个细胞的形态学概率
        log_prob = np.zeros(n_cells)
        for f in feature_list:
            x = morphology_features[f].values
            mu_tf = mu[t, feature_list.index(f)]
            sigma_tf = sigma[t, feature_list.index(f)]
            log_prob += norm.logpdf(x, loc=mu_tf, scale=sigma_tf)
        PM[:, t] = np.exp(log_prob)

    # 避免数值下溢
    PM[PM < 1e-100] = 1e-100

    # 归一化
    if scaled:
        PM = PM / PM.sum(axis=1, keepdims=True)

    return PM

def compute_PME_on_cell(PE_on_spot, PM_on_cell, cells_on_spot):
    """
    E 步：计算联合概率 PME_on_cell
    Args:
        PE_on_spot: 基因表达概率，(n_spots, n_types)
        PM_on_cell: 形态学概率，(n_cells, n_types)
        cells_on_spot: DataFrame，包含 cell_id 和 spot
    Returns:
        PME_on_cell: 联合概率，(n_cells, n_types)
    """
    spot_id = cells_on_spot['spot_name'].astype(str).values
    cell_id = cells_on_spot['cell_id'].astype(str).values

    # 映射 PE_on_spot 到细胞级别
    spot_indices = {s: i for i, s in enumerate(np.unique(spot_id))}
    spot_idx = np.array([spot_indices[s] for s in spot_id])
    PE_on_cell = PE_on_spot[spot_idx, :]

    # 计算 PME_on_cell
    PME_on_cell = PM_on_cell * PE_on_cell
    PME_on_cell = PME_on_cell / PME_on_cell.sum(axis=1, keepdims=True)

    return PME_on_cell, cell_id

# M步：更新形态学参数
def update_morphology_parameter(PE_on_spot, PM_on_cell, cells_on_spot, morphology_features, features):
    """
    M 步：更新形态学参数 mu 和 sigma
    Args:
        PE_on_spot: 基因表达概率，(n_spots, n_types)
        PM_on_cell: 形态学概率，(n_cells, n_types)
        cells_on_spot: DataFrame，包含 cell_id 和形态学特征
        features: 形态学特征名称列表
    Returns:
        mu: 更新后的均值，(n_types, n_features)
        sigma: 更新后的标准差，(n_types, n_features)
    """
    # 计算 PME_on_cell
    PME_on_cell, cell_id = compute_PME_on_cell(PE_on_spot, PM_on_cell, cells_on_spot)

    # 处理重复细胞：取最大概率
    cell_ids_unique = np.unique(cell_id)
    PME_uni_cell = np.zeros((len(cell_ids_unique), PME_on_cell.shape[1]))
    for i, cid in enumerate(cell_ids_unique):
        mask = cell_id == cid
        PME_uni_cell[i, :] = PME_on_cell[mask, :].max(axis=0)

    # 更新 mu
    X = morphology_features[features].values  # (n_cells, n_features)
    a = PME_uni_cell.T @ X  # (n_types, n_features)
    s = PME_uni_cell.sum(axis=0, keepdims=True)  # (1, n_types)
    mu = a / s.T  # (n_types, n_features)

    # 更新 sigma
    n_types = PME_on_cell.shape[1]
    sigma = np.zeros_like(mu)
    for t in range(n_types):
        p = PME_uni_cell[:, t:t+1]  # (n_cells, 1)
        d = X - mu[t, :]  # (n_cells, n_features)
        b = (p * d**2).sum(axis=0)  # (n_features,)
        sigma[t, :] = np.sqrt(b / s[0, t])

    return mu, sigma


# 计算每个细胞属于各个类型的概率
def cal_prob_kmeans(cell_contributions, cells_on_spot, type_list,
                   morphology_features, feature_list, scale=True, temp=0.5):
    """
    基于K-means聚类计算每个细胞属于各个类型的概率
    Args:
        cell_contributions: array of shape (n_unique_spots, n_types)，解卷积得到的每个spot中各个类型的比例
        cells_on_spot: dictionary，包含spot_name和all_cells_in_spot，spot可能重复出现
        type_list: list，细胞类型列表
        morphology_features: DataFrame，形态学特征
        feature_list: list，要使用的特征列表
        scale: bool，是否对特征进行标准化
        temp: float，用于计算概率的温度参数，越大概率分布越平滑
    Returns:
        PM_on_cell: array of shape (n_cells, n_types)，每个细胞属于各个类型的概率
    """
    
    # 准备形态学特征
    X_morph = morphology_features[feature_list].values
    if scale:
        scaler = StandardScaler()
        X_morph = scaler.fit_transform(X_morph)
    
    # K-means聚类
    n_types = len(type_list)
    kmeans = KMeans(n_clusters=n_types, random_state=42)
    cluster_labels = kmeans.fit_predict(X_morph)
    print("K-means clustering labels:", cluster_labels)
    
    # 获取唯一的spots和它们在cell_contributions中的索引
    unique_contribution_spots = cells_on_spot['spot_name']  # cell_contributions中的spots
    spot_to_contrib_idx = {spot: idx for idx, spot in enumerate(unique_contribution_spots)}
    
    # 获取所有细胞的spot信息
    cell_spot_ids = cells_on_spot['spot_name'].astype(str).values
    unique_cell_spots = np.unique(cell_spot_ids)
    
    # 构建spot到cluster的映射
    spot_to_clusters = {}
    for spot in unique_cell_spots:
        mask = cell_spot_ids == spot
        spot_to_clusters[spot] = cluster_labels[mask]
    
    # 计算每个cluster对应的cell type
    # 构建cost matrix：每个cluster分配给每个type的代价
    cost_matrix = np.zeros((n_types, n_types))
    for i in range(n_types):  # cluster id
        for j in range(n_types):  # cell type id
            weighted_cost = 0
            total_weight = 0
            
            # 只处理在cell_contributions中有对应索引的spots
            for spot in unique_cell_spots:
                if spot not in spot_to_contrib_idx:
                    continue
                    
                contrib_idx = spot_to_contrib_idx[spot]
                if contrib_idx >= len(cell_contributions):
                    continue
                    
                cluster_counts = spot_to_clusters[spot]
                total_cells = len(cluster_counts)
                if total_cells > 0:
                    # 计算cluster比例和cell type比例
                    cluster_prop = np.sum(cluster_counts == i) / total_cells
                    type_prop = cell_contributions[contrib_idx, j]
                    
                    # 使用cell type比例作为权重
                    weight = type_prop
                    weighted_cost += weight * abs(cluster_prop - type_prop)
                    total_weight += weight
            
            # 归一化cost
            cost_matrix[i, j] = weighted_cost / (total_weight + 1e-10)
    
    print("Cost matrix:", cost_matrix)
    
    # 使用匈牙利算法找到最优匹配
    cluster_indices, type_indices = linear_sum_assignment(cost_matrix)
    print("Cluster to type mapping:", dict(zip(cluster_indices, type_indices)))
    
    # 构建cluster到type的映射
    cluster_to_type = {cluster_idx: type_idx for cluster_idx, type_idx in zip(cluster_indices, type_indices)}
    
    # 计算每个细胞到各个类中心的距离
    distances = cdist(X_morph, kmeans.cluster_centers_, metric='euclidean')
    
    # 将距离转换为概率
    # 使用softmax函数：P(type_j|cell_i) = exp(-d_ij/temp) / sum_k(exp(-d_ik/temp))
    probabilities = np.exp(-distances/temp)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # 根据cluster到type的映射重排概率矩阵的列
    PM_on_cell = np.zeros_like(probabilities)
    for cluster_idx, type_idx in cluster_to_type.items():
        PM_on_cell[:, type_idx] = probabilities[:, cluster_idx]
    
    return PM_on_cell



# cell_contributions 提供了每个 spot 的细胞类型分布（先验），
# 而 cells_on_spot 提供了细胞与 spot 的对应关系
# 通过结合特征信息（likelihood）和先验信息，
# 可以直接计算每个细胞的后验概率 P(type|cell)，无需显式聚类和硬匹配
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

def cal_prob_bayes(cell_contributions, cells_on_spot,
                           morphology_features, feature_list,
                           scale=True, temp=0.5):
    """
    基于贝叶斯框架计算每个细胞属于各个类型的概率，先对齐细胞类型标签
    """
    type_list = cell_contributions.columns.tolist()

    # Step 1: 准备形态学特征
    X_morph = morphology_features[feature_list].values
    if scale:
        scaler = StandardScaler()
        X_morph = scaler.fit_transform(X_morph)
    
    n_cells = len(morphology_features)
    n_types = len(type_list)
    
    # Step 2: 对形态学特征进行聚类
    gmm = GaussianMixture(n_components=n_types, random_state=42)
    gmm.fit(X_morph)
    morph_cluster_labels = gmm.predict(X_morph)
    morphology_features['morph_type'] = morph_cluster_labels

    # Step 3: 统计每个 spot 上形态学分类的类型分布
    cells_on_spot = cells_on_spot.set_index('cell_id').loc[morphology_features.index]
    morphology_features['spot_name'] = cells_on_spot['spot_name'].values
    cell_contributions_morph = morphology_features.groupby(['spot_name', 'morph_type']).size().unstack(fill_value=0)
    cell_contributions_morph = cell_contributions_morph.div(cell_contributions_morph.sum(axis=1), axis=0)

    # Step 4: 对齐 cell_contributions 和 cell_contributions_morph 的分类
    # 构建代价矩阵，使用所有 spot 的平均分布差异
    common_spots = cell_contributions.index.intersection(cell_contributions_morph.index)
    cost_matrix = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(n_types):
            diff = np.abs(cell_contributions.iloc[:, i].loc[common_spots] - 
                          cell_contributions_morph.iloc[:, j].loc[common_spots])
            cost_matrix[i, j] = diff.mean()
    
    # 匈牙利算法进行最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    type_alignment = dict(zip(col_ind, cell_contributions.columns[row_ind]))

    # Step 5: 重新映射 GMM cluster label 到 cell type 标签
    mapped_morph_labels = [type_alignment[label] for label in morph_cluster_labels]

    # Step 6: 准备贝叶斯计算中需要的先验分布
    prior_probs = np.zeros((n_cells, n_types))
    for idx, cell_id in enumerate(morphology_features.index):
        cell_spot = cells_on_spot.loc[cell_id, 'spot_name']
        prior_probs[idx] = cell_contributions.loc[cell_spot].values

    # Step 7: 根据映射后的 GMM 分布构建 likelihood
    likelihoods = np.zeros((n_cells, n_types))
    for j, type_name in enumerate(type_list):
        gmm_type = [k for k, v in type_alignment.items() if v == type_name][0]
        mean = gmm.means_[gmm_type]
        cov = gmm.covariances_[gmm_type]
        likelihoods[:, j] = multivariate_normal.pdf(X_morph, mean=mean, cov=cov)
    
    # Step 8: 正则化和计算后验
    likelihoods = likelihoods / (likelihoods.max(axis=1, keepdims=True) + 1e-10)
    posterior_probs = likelihoods * prior_probs
    posterior_probs = posterior_probs / (posterior_probs.sum(axis=1, keepdims=True) + 1e-10)

    # Step 9: 应用温度参数平滑概率
    if temp != 1.0:
        posterior_probs = np.power(posterior_probs, 1.0 / temp)
        posterior_probs = posterior_probs / (posterior_probs.sum(axis=1, keepdims=True) + 1e-10)

    PM_on_cell = pd.DataFrame(posterior_probs, columns=type_list, index=morphology_features.index)
    return PM_on_cell





import pandas as pd
import numpy as np
from scipy import sparse
import scanpy as sc

def get_similarity_df(adata, celltype_col='Level1'):
    # Assume adata is your AnnData object, celltype_col = "Level1"

    # Step 1: Calculate mean gene expression for each cell type
    cell_types = adata.obs[celltype_col].unique()
    mean_expr = {}
    for ct in cell_types:
        mask = adata.obs[celltype_col] == ct
        # Compute mean expression, handling sparse or dense matrix
        mean_expr[ct] = adata[mask].X.mean(axis=0).A1 if sparse.issparse(adata.X) else adata[mask].X.mean(axis=0)

    # Create DataFrame of mean expressions (genes x cell types)
    mean_expr_df = pd.DataFrame(mean_expr, index=adata.var_names)

    # Step 2: Compute pairwise PCC between cell type mean expressions
    pcc_matrix = mean_expr_df.corr(method='pearson')

    # Step 3: Normalize PCCs to sum to 1 for each cell type
    pcc_matrix = pcc_matrix.div(pcc_matrix.sum(axis=1), axis=0)
    pcc_matrix = pcc_matrix.fillna(0)  # Handle NaN values

    # Step 4: Assign normalized PCC vector to each cell based on its cell type
    similarity_df = pd.DataFrame(0.0, index=adata.obs['cell_id'], columns=cell_types)
    for ct in cell_types:
        # Select cells of this cell type
        mask = adata.obs[celltype_col] == ct
        cell_ids = adata.obs['cell_id'][mask]
        # Assign the corresponding normalized PCC vector
        similarity_df.loc[cell_ids, :] = pcc_matrix.loc[ct].values

    similarity_df.columns = similarity_df.columns.astype(str)

    return similarity_df