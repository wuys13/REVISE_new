import numpy as np
import pandas as pd
from scipy.sparse import issparse

from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score

import scanpy as sc


def normalize_data(data):
    min_vals = np.min(data, axis=0, keepdims=True)
    max_vals = np.max(data, axis=0, keepdims=True)
    
    # 避免除以零的情况
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 如果范围为零，将其设置为1，避免NaN

    normalized_data = (data - min_vals) / range_vals
    return normalized_data

def cw_ssim(img1, img2, **kwargs):
    """
    Compute the Complex Wavelet Structural Similarity (CW-SSIM) index between two images.
    """
import numpy as np

def mean_without_outliers_iqr(data, k=1.5, nan_policy='omit'):
    """
    用 IQR 法去除离群值后求均值。

    参数
    ----
    data : array-like
        输入的一维数组，可含 NaN。
    k : float, optional
        控制须线长度的系数，默认 1.5（Tukey 标准）。
    nan_policy : {'omit', 'raise'}, optional
        'omit'（默认）会先剔除 NaN；'raise' 则遇到 NaN 会抛出 ValueError。

    返回
    ----
    float
        去掉离群值后的均值；若过滤后无数据，返回 NaN。
    """
    a = np.asarray(data)

    if nan_policy == 'omit':
        a = a[~np.isnan(a)]
    elif nan_policy == 'raise' and np.isnan(a).any():
        raise ValueError("Input contains NaN.")

    if a.size == 0:
        return np.nan

    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr

    filtered = a[(a >= lower) & (a <= upper)]
    return float(np.mean(filtered)) if filtered.size else np.nan 

def compute_metric(adata_original, adata_noisy, adata_process=False, sampe_ratio = None, gene_list=None, normalize=False,fenlei = False):
    """
    计算加噪前后每个基因的 PCC、SSIM 和 MSE。

    参数:
    - adata_original: 原始 AnnData 对象。
    - adata_noisy: 加噪后的 AnnData 对象。
    - data_process: 如果为 True，则对数据进行归一化和对数转换。

    返回:
    - 三个 DataFrame，分别包含每个基因的 PCC、SSIM 和 MSE。
    """
    # 复制 AnnData 对象以避免修改原始数据
    adata_orig = adata_original.copy()
    adata_noisy_copy = adata_noisy.copy()

    if sampe_ratio is not None:
        print(f"Sampling {sampe_ratio*100}% of cells.")
        indices = np.random.choice(adata_orig.n_obs, int(adata_orig.n_obs * sampe_ratio), replace=False)
        adata_orig = adata_orig[indices, :]
        adata_noisy_copy = adata_noisy_copy[indices, :]

    if gene_list is not None:
        print(f"Using {len(gene_list)} genes.")
        adata_orig = adata_orig[:, gene_list]
        adata_noisy_copy = adata_noisy_copy[:, gene_list]

    # 如果 data_process 为 True，进行归一化和对数转换
    if adata_process:
        print("Normalizing and log-transforming data.")
        sc.pp.normalize_total(adata_orig, target_sum=1e4)
        sc.pp.normalize_total(adata_noisy_copy, target_sum=1e4)
        sc.pp.log1p(adata_orig)
        sc.pp.log1p(adata_noisy_copy)
        
    # 提取表达矩阵，处理稀疏矩阵
    X_orig = adata_orig.X.toarray() if issparse(adata_orig.X) else adata_orig.X
    X_noisy = adata_noisy_copy.X.toarray() if issparse(adata_noisy_copy.X) else adata_noisy_copy.X

    if normalize:
        X_orig = normalize_data(X_orig)
        X_noisy = normalize_data(X_noisy)

    # for gene
    pcc_types =[]
    pcc_values = []
    ssim_values = []
    cw_ssim_values = []
    mse_values = []
    nrmse_values = []

    genes = adata_orig.var_names

    # 对每个基因计算指标
    for i, gene in enumerate(genes):
        orig_expr = X_orig[:, i]
        noisy_expr = X_noisy[:, i]

        # 计算 PCC
        if fenlei:
            noisy_expr_num = len(np.unique(noisy_expr))
            if noisy_expr_num > 1:
                pcc, _ = pearsonr(orig_expr, noisy_expr)
                pcc_type = "PCC"
            # pcc, _ = spearmanr(orig_expr, noisy_expr)
            # pcc is NA: 将orig_expr转为binary，并计算Accuracy
            else:
                orig_expr = np.where(orig_expr > 0, 1, 0)
                noisy_expr = np.where(noisy_expr > 0, 1, 0)
                pcc = accuracy_score(orig_expr, noisy_expr)
                pcc_type = "Accuracy"
        else:
            pcc, _ = pearsonr(orig_expr, noisy_expr)
            pcc_type = "PCC"
        
        pcc_values.append(pcc)
        pcc_types.append(pcc_type)

        # 计算 SSIM
        if normalize:
            data_range = 1
        else:
            data_range = orig_expr.max() - orig_expr.min()
        ssim_value = ssim(orig_expr, noisy_expr, data_range=data_range)
        ssim_values.append(ssim_value)

        # cw_ssim_value = cw_ssim(orig_expr, noisy_expr)
        # cw_ssim_values.append(cw_ssim_value)

        # 计算 MSE
        mse = np.mean((orig_expr - noisy_expr) ** 2) 
        mse_values.append(mse)

        # 计算 NRMSE
        nrmse = np.sqrt(mse) / np.mean(orig_expr)
        nrmse_values.append(nrmse)



    print(f"PCC: {np.nanmean(pcc_values):.4f}, {np.nanmax(pcc_values):.4f}, {np.nanmin(pcc_values):.4f}")
    print(f"SSIM: {np.nanmean(ssim_values):.4f}, {np.nanmax(ssim_values):.4f}, {np.nanmin(ssim_values):.4f}")
    
    print(f"MSE: {np.nanmean(mse_values):.4f}, {np.nanmax(mse_values):.4f}, {np.nanmin(mse_values):.4f}")
    print(f"NRMSE: {np.nanmean(nrmse_values):.4f}, {np.nanmax(nrmse_values):.4f}, {np.nanmin(nrmse_values):.4f}")
    # 创建 DataFrame
    metrics_df = pd.DataFrame({'Gene': genes, 
                               'PCC_type': pcc_types,
                    'PCC': pcc_values,
                    'SSIM': ssim_values,
                    "MSE": mse_values,
                    "NRMSE": nrmse_values})
    
    # if gene_list is not None:
    #     metrics_df['use_aligment'] = metrics_df['Gene'].apply(lambda x: x in gene_list)
    
    #     a_df = metrics_df[metrics_df['use_aligment'] == True]
    #     b_df = metrics_df[metrics_df['use_aligment'] == False]

    #     print(f"Alignment genes PCC: {a_df['PCC'].mean():.4f}, {a_df['PCC'].max():.4f}, {a_df['PCC'].min():.4f}")
    #     print(f"Alignment genes SSIM: {a_df['SSIM'].mean():.4f}, {a_df['SSIM'].max():.4f}, {a_df['SSIM'].min():.4f}")
    #     print(f"Alignment genes MSE: {a_df['MSE'].mean():.4f}, {a_df['MSE'].max():.4f}, {a_df['MSE'].min():.4f}")

    #     print(f"Non-alignment genes PCC: {b_df['PCC'].mean():.4f}, {b_df['PCC'].max():.4f}, {b_df['PCC'].min():.4f}")
    #     print(f"Non-alignment genes SSIM: {b_df['SSIM'].mean():.4f}, {b_df['SSIM'].max():.4f}, {b_df['SSIM'].min():.4f}")
    #     print(f"Non-alignment genes MSE: {b_df['MSE'].mean():.4f}, {b_df['MSE'].max():.4f}, {b_df['MSE'].min():.4f}")


    # # for cell
    # mse_values = []
    # nrmse_values = []
    # for i in range(X_orig.shape[0]):
    #     orig_expr = X_orig[i, :]
    #     noisy_expr = X_noisy[i, :]
    #     # cell_id = adata_orig.obs_names[i]
    #     # cell_type = adata_orig.obs.loc[cell_id, "Level1"]

    #     mse = np.mean((orig_expr - noisy_expr) ** 2)
    #     mse_values.append(mse)

    #     nrmse = np.sqrt(mse) / np.mean(orig_expr)
    #     nrmse_values.append(nrmse)

    # num = 30
    # if "Level1" in adata_orig.obs.columns:
    #     celltype_col = "Level1"
    # elif "clusters" in adata_orig.obs.columns:
    #     celltype_col = "clusters"
    # else:
    #     raise ValueError("No cell type column found in adata.obs")

    # cell_metrics_df = pd.DataFrame({'Cell': adata_orig.obs_names,
    #                                 'Cell_type': adata_orig.obs[celltype_col].values,
    #                 'MSE': mse_values,
    #                 'NRMSE': nrmse_values})
    # top_mse = cell_metrics_df.nlargest(num, 'MSE')
    # top_nrmse = cell_metrics_df.nlargest(num, 'NRMSE')
    # print("Spot top_mse: ", top_mse["Cell_type"].value_counts())
    # print("Spot top_nrmse: ", top_nrmse["Cell_type"].value_counts())


    return metrics_df

