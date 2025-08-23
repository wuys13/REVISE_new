import scanpy as sc
from imputation.impute import get_common_embedding, get_imputed_adata, get_harmony_embedding

# test_gene_idx=340
test_gene_idx = None # all train 

part_num = 3
# part_num = 1000


task = "imputation"
patient_id = "P2CRC"


source_path = f"data/{task}/{patient_id}/cut_part{part_num}"
adata_sp = sc.read(f"{source_path}/selected_xenium.h5ad")

task = "spot"
batch_num = 3
source_path = f"data/{task}/{patient_id}/cut_part{part_num}"
if batch_num == 1:
    file_name = "selected_xenium.h5ad"
elif batch_num == 2:
    file_name = "real_sc_ref_all.h5ad"
elif batch_num == 3:
    file_name = "real_sc_ref_part.h5ad"
elif batch_num == 4:
    file_name = "real_sc_ref_others.h5ad"

sc_file = f"{source_path}/{file_name}"

# sc_file = "/home/wys/Sim2Real-ST/REVISE_data_process/raw_data/adata_sc_all_reanno.h5ad"
# adata_sc = sc.read(sc_file)
# all_cts = adata_sc.obs["Level1"].unique()
# adata_sc = adata_sc[adata_sc.obs["Level1"].isin(all_cts)]


adata_sc = sc.read(sc_file)


overlap_genes = list(adata_sp.var_names.intersection(adata_sc.var_names))
print(len(overlap_genes))


if test_gene_idx is None:
    train_gene = overlap_genes
    test_gene = overlap_genes
else:
    train_gene = overlap_genes[:test_gene_idx]
    test_gene = overlap_genes[test_gene_idx:]

# 
method = "cca"
method = "svd"

emb = "pca"
# emb = "nmf"
# emb = "raw"

adata_sp_raw = adata_sp.copy()
adata_sp, adata_sc = get_common_embedding(
    adata_sp[:,train_gene], adata_sc, 
    method=method, emb=emb,
    n_components=100, project_on="sp", 
    cos_threshold = 0.3,
    normalize_flag = False,
)

# adata_sp, adata_sc = get_harmony_embedding(
#     adata_sp[:,train_gene], adata_sc, 
#     n_pcs=100,
# )

print("Finish emb")

impute_method = "MNN"
impute_method = "NearestNeighbors"

adata, _ = get_imputed_adata(
    adata_sp, adata_sc[:, test_gene], method=impute_method, 
    n_neighbors=50)
print("Finish imputation")


overlap_genes = adata_sp_raw.var_names.intersection(adata.var_names)
print(len(overlap_genes))
adata_1 = adata_sp_raw[:, overlap_genes]
adata_2 = adata[:, overlap_genes]

from revise.metrics import compute_metric

metrics_df = compute_metric(adata_1, adata_2,
                            adata_process=False, gene_list=None,
                            normalize=False)
metrics_df = compute_metric(adata_1, adata_2,
                            adata_process=False, gene_list=None,
                            normalize=True)
