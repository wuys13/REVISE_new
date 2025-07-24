import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap(adata_raw, scale=True, color = ["Level1"], save_path=None):
    adata = adata_raw.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    # adata = adata[:, adata.var['highly_variable']]
    if scale:
        sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=color)

    adata_raw.obsm["X_umap"] = adata.obsm["X_umap"]

    

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return adata_raw

    adata_raw.obsm["X_umap"] = adata.obsm["X_umap"]
    return adata_raw

def plot_shared_embedding(adata_raw, color, use_rep='X_shared', save_path=None):
    adata = adata_raw.copy()
    if adata.shape[0] > 20000:
        print(f"Subsampling {adata.shape[0]} observations to 10000")
        sc.pp.subsample(adata, n_obs=10000)
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
        
    sc.pl.umap(adata, color=color, show=False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    if adata_raw.shape[0] < 20000:
        adata_raw.obsm["X_umap"] = adata.obsm["X_umap"]
    return adata_raw

def plot_usage_histograms(stat_csv='cell_stat_all.csv', output_folder='plot/'):
    """
    读取cell_stat_all.csv，为每个select_ct绘制频率分布直方图。
    """
    df = pd.read_csv(stat_csv, index_col=0)
    
    for select_ct, group_df in df.groupby('select_ct'):
        
        # Plot for used_count
        plt.figure(figsize=(10, 6))
        
        # Group by cell_type and normalize each group's 'used_count'
        for ct, ct_df in group_df.groupby('cell_type'):
            sns.histplot(ct_df['used_count'], label=ct, stat='density', common_norm=False, kde=True)

        plt.title(f'Used Count Distribution for {select_ct}')
        plt.xlabel('Used Count')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}{select_ct}_used_count_dist.png")
        plt.close()

        # Plot for used_weight_mean
        plt.figure(figsize=(10, 6))
        
        for ct, ct_df in group_df.groupby('cell_type'):
            sns.histplot(ct_df['used_weight_mean'], label=ct, stat='density', common_norm=False, kde=True)

        plt.title(f'Mean Weight Distribution for {select_ct}')
        plt.xlabel('Mean Weight')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_folder}{select_ct}_used_weight_mean_dist.png")
        plt.close()
