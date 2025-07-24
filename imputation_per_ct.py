import scanpy as sc
from imputation.impute import get_common_embedding, get_imputed_adata, get_harmony_embedding
from imputation.plot import plot_shared_embedding, plot_usage_histograms
import pandas as pd
import os

# test_gene_idx=340
test_gene_idx = None # all train 

part_num = 1
# part_num = 1000

# Create plot directory
if not os.path.exists('plot'):
    os.makedirs('plot')

task = "imputation"
patient_id = "P2CRC"

source_path = f"data/{task}/{patient_id}/cut_part{part_num}"
adata_sp = sc.read(f"{source_path}/selected_xenium.h5ad")
adata_sc = sc.read(f"{source_path}/real_sc_ref.h5ad")
adata_sp.obs['Level1'].replace("Mono/Macro", "Mono_Macro", inplace=True)
adata_sc.obs['Level1'].replace("Mono/Macro", "Mono_Macro", inplace=True)

overlap_genes = list(adata_sp.var_names.intersection(adata_sc.var_names))
print(len(overlap_genes))

if test_gene_idx is None:
    train_gene = overlap_genes
    test_gene = overlap_genes
else:
    train_gene = overlap_genes[:test_gene_idx]
    test_gene = overlap_genes[test_gene_idx:]
adata_sp_raw = adata_sp.copy()

# 
method = "cca"
method = "svd"
# method = "harmony"

emb = "pca"
# emb = "nmf"
# emb = "raw"
impute_method = "MNN"
impute_method = "NearestNeighbors"


all_cts = adata_sp.obs['Level1'].unique()
print(adata_sp.obs['Level1'].value_counts())
print(all_cts)
adata_sp_impute = None 
from tqdm import tqdm
for select_ct in tqdm(all_cts, "Cell_type"):
    ct_adata_sp = adata_sp[adata_sp.obs['Level1'] == select_ct]
    ct_adata_sc = adata_sc[adata_sc.obs['Level1'] == select_ct]
    if method == "harmony":
        ct_adata_sp, ct_adata_sc = get_harmony_embedding(
            ct_adata_sp, ct_adata_sc, 
            n_pcs=100,
        )
    else:
        ct_adata_sp, ct_adata_sc = get_common_embedding(
            ct_adata_sp[:,train_gene], ct_adata_sc, 
            method=method, emb=emb,
            n_components=100, project_on="sp", 
            cos_threshold = 0.3,
            normalize_flag = False,
        )

    # Plot UMAP for each select_ct
    combined_adata = ct_adata_sp.concatenate(ct_adata_sc, batch_key='batch')
    plot_shared_embedding(combined_adata, color=['batch'], use_rep='X_shared', 
                          save_path=f'plot/{select_ct}_batch_umap.png',
                          )

    imputed_result = get_imputed_adata(
        ct_adata_sp, ct_adata_sc[:, test_gene], method=impute_method, 
        n_neighbors=50)
    
    if isinstance(imputed_result, tuple):
        adata, df_stat = imputed_result
    else:
        adata, df_stat = imputed_result, None

    # # 保存统计df
    # if df_stat is not None:
    #     df_stat['select_ct'] = select_ct
    #     if 'all_stat_df' not in locals():
    #         all_stat_df = df_stat.copy()
    #     else:
    #         all_stat_df = pd.concat([all_stat_df, df_stat])
    
    if adata_sp_impute is None:
        adata_sp_impute = adata.copy()
    else:
        adata_sp_impute = adata_sp_impute.concatenate(adata, join='outer', index_unique=None)
    
print("Finish imputation")

overlap_genes = adata_sp_raw.var_names.intersection(adata_sp_impute.var_names)
print(len(overlap_genes))
adata_1 = adata_sp_raw[:, overlap_genes]
adata_2 = adata_sp_impute[adata_sp_raw.obs_names, overlap_genes]

from revise.metrics import compute_metric

metrics_df = compute_metric(adata_1, adata_2,
                            adata_process=False, gene_list=None,
                            normalize=False)
metrics_df = compute_metric(adata_1, adata_2,
                            adata_process=False, gene_list=None,
                            normalize=True)

# if 'all_stat_df' in locals():
#     all_stat_df.to_csv('cell_stat_all.csv')
#     plot_usage_histograms(stat_csv='cell_stat_all.csv', output_folder='plot/')