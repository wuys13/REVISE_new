import scanpy as sc
from imputation.impute import get_common_embedding, get_imputed_adata, get_harmony_embedding
import os

# test_gene_idx=340
test_gene_idx = None # all train 

part_num = 1
# part_num = 1000


task = "imputation"
patient_id = "P2CRC"


# source_path = f"data/{task}/{patient_id}/cut_part{part_num}"
# adata_sp = sc.read(f"{source_path}/selected_xenium.h5ad")
# adata_sc = sc.read(f"{source_path}/real_sc_ref.h5ad")
root_path = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/wuyushuai/REVISE_new/data/spot"
source_path = os.path.join(root_path, patient_id, f"cut_part{part_num}")
adata_sp = sc.read(f"{source_path}/selected_xenium.h5ad")
adata_sc = sc.read(f"{source_path}/real_sc_ref.h5ad")

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
