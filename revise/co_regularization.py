from tqdm import tqdm
import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
import scipy

import numpy as np
import scipy.sparse

def get_merge_score(adata, label="Level1", spatial_coeff=0.9):

    # 1. 计算空间邻居图
    sq.gr.spatial_neighbors(adata)
    nn_graph_space = adata.obsp["spatial_connectivities"]

    labels = adata.obs[label].to_numpy()
    classes = adata.obsm[label].columns.to_numpy()

    # 2. 创建 label 的整数编码
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    class_order = le.classes_

    if not np.all(class_order == classes):
        print(f"⚠️ 警告: classes顺序与编码顺序不一致，自动对齐为 {class_order}")
        classes = class_order

    n_cells = labels.shape[0]
    n_classes = len(classes)

    # 3. 用稀疏矩阵一次性统计每个类别的邻居数
    rows, cols = nn_graph_space.nonzero()
    data = np.ones_like(rows)

    neighbor_labels = labels_encoded[cols]

    # 生成 (cell_id, class_id) pair
    from scipy.sparse import coo_matrix

    counts = coo_matrix(
        (data, (rows, neighbor_labels)),
        shape=(n_cells, n_classes)
    ).tocsr()

    # 4. 转 numpy，归一化
    counts = counts.toarray()
    total_neighbors = np.array(nn_graph_space.sum(axis=1)).flatten()
    total_neighbors[total_neighbors == 0] = 1  # 防止除以0

    spatial_scores = counts / total_neighbors[:, None]  # 每行归一化到 [0,1]

    # 5. 加权平均 gene_scores 和 spatial_scores
    gene_scores = adata.obsm[label].to_numpy()
    merge_scores = (gene_scores + spatial_coeff * spatial_scores) / (1 + spatial_coeff)

    adata.obsm['spatial_scores'] = spatial_scores
    adata.obsm['merge_scores'] = merge_scores

    adata.obs['merge_scores'] = merge_scores.max(axis=1)
    adata.obs['merge_label'] = classes[merge_scores.argmax(axis=1)]
    adata.obs['change'] = adata.obs['merge_label'] != adata.obs[label]
    print(adata_sample.obs.groupby('Level1')[['merge_label','change']].value_counts())

    return adata



def get_spatial_score(adata, res):
    # 为每个细胞计算空间邻居是一个leiden标签的个数
    nn_graph_space = adata.obsp["spatial_connectivities"]

    labels = adata.obs[f"leiden_{res}"].to_numpy()

    # 统计一下邻居中和当前标签一致的个数

    same_label_matrix = (labels[:, None] == labels[None, :]).astype(int)
    same_label_matrix = scipy.sparse.csr_matrix(same_label_matrix)
    spatial_count = (nn_graph_space.multiply(same_label_matrix)).sum(axis=1).A1
    # spatial_count = (nn_graph_space.dot(labels) == labels).sum(axis=1)
    
    adata.obs[f"spatial_score_{res}"] = spatial_count

    return adata


def get_align_score(adata, res, label = "Level2"):
    leiden_class = adata.obs[f'leiden_{res}']
    unique_class = np.unique(leiden_class)
    # 计算每个class数目最多label的数量
    align_score = 0
    for class_name in unique_class:
        class_label = adata.obs[label][leiden_class == class_name]
        max_label_num = class_label.value_counts().max()
        align_score += max_label_num
        
    align_score = align_score / len(leiden_class)
    return align_score

def assign_max_label(adata, res, label = "Level2"):
    leiden_class = adata.obs[f'leiden_{res}']
    unique_class = np.unique(leiden_class)
    relabel_name = f"relabel_{label}"
    # 计算每个class数目最多label的数量
    for class_name in unique_class:
        class_label = adata.obs[label][leiden_class == class_name]
        max_label = class_label.value_counts().index[0]
        adata.obs[relabel_name][leiden_class == class_name] = max_label
    
    adata.obs['match'] = adata.obs[relabel_name] == adata.obs[label]
    return adata

def get_best_cluster(adata, neighbors_method = "pca", alpha = 0.2, resolutions = [0.1, 0.3, 0.5, 0.7, 0.9], label = "Level2"):
    adata = adata.copy()
    adata_raw = adata.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=100)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.pca(adata, n_comps=30)

    sc.pp.neighbors(adata, n_pcs=30)
    nn_graph_genes = adata.obsp["connectivities"]
    # spatial proximity graph
    sq.gr.spatial_neighbors(adata)
    nn_graph_space = adata.obsp["spatial_connectivities"]
    joint_graph = (1 - alpha) * nn_graph_genes + alpha * nn_graph_space

    if neighbors_method == "pca":
        adjacency_graph = nn_graph_genes
    elif neighbors_method == "spatial":
        adjacency_graph = nn_graph_space
    elif neighbors_method == "joint":
        adjacency_graph = joint_graph
    else:
        raise ValueError("neighbors_method must be pca or spatial")
    
    merge_df = pd.DataFrame()
    for res in tqdm(resolutions, desc = "leiden"):
        sc.tl.leiden(adata, adjacency=adjacency_graph, resolution=res, key_added=f"leiden_{res}" )
    
        adata = get_spatial_score(adata, res = res)
        align_score = get_align_score(adata, res = res, label = label)
        mean_score = np.mean(adata.obs[f"spatial_score_{res}"])
        cluster_num = len(np.unique(adata.obs[f"leiden_{res}"]))
        # weighted_score = mean_score * np.log(cluster_num)
        df = pd.DataFrame({
            "resolution": res,
            "cluster_num": cluster_num,
            "mean_score": mean_score,
            "align_score": align_score,
        }, index = [0])
        merge_df = pd.concat([merge_df, df], axis = 0)
        best_res = merge_df[merge_df["align_score"] == merge_df["align_score"].max()]["resolution"].values[0]

        adata = assign_max_label(adata, res = best_res, label = label)

        print(f"Resolution {res}: {cluster_num} clusters mean spatial score: {mean_score:.4f} {align_score}...")
        sc.pl.scatter(adata, x = "x", y = "y", color = [f"leiden_{res}", f"spatial_score_{res}"] )
    
    merge_df.reset_index(inplace=True)
    adata_raw.obs = adata.obs.copy()
    
    return adata_raw, merge_df, best_res