import os
import scanpy as sc
from imputation.impute import get_imputed_adata
import numpy as np
import tacco as tc
from imputation.impute import get_common_embedding
import anndata as ad

# test_gene_idx=340
test_gene_idx = None # all train

part_num = 1
# part_num = 1000

task = "imputation"
patient_id = "P2CRC"
method = "svd"
emb = "pca"
# impute_method = "NearestNeighbors"
impute_method = "OT"


def tacco_get(adata_st, adata_sc, **kwargs):
    adata_st = adata_st.copy()
    adata_st.X = np.around(adata_st.X)
    if "annotation_key" in kwargs:
        a, adata_st_pv, adata_sc_pv = tc.tl.annotate(adata_st, adata_sc, **kwargs)
    else:
        a, adata_st_pv, adata_sc_pv = tc.tl.annotate(adata_st, adata_sc, annotation_key="Level1", **kwargs)
    return a, adata_st_pv, adata_sc_pv


if __name__ == "__main__":
    root_path = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/wuyushuai/REVISE_new/data/spot"
    source_path = os.path.join(root_path, patient_id, f"cut_part{part_num}")
    adata_sp = sc.read(f"{source_path}/selected_xenium.h5ad")
    adata_sc = sc.read(f"{source_path}/real_sc_ref.h5ad")

    adata_sp_raw = adata_sp.copy()
    # overlap_genes = list(adata_sp.var_names.intersection(adata_sc.var_names))
    # adata_sp, adata_sc = get_common_embedding(
    #     adata_sp[:,overlap_genes], adata_sc,
    #     method=method, emb=emb,
    #     n_components=100, project_on="sp",
    #     cos_threshold = 0.3,
    #     normalize_flag = False,
    # )
    # print("Finish emb")

    # adata_sp_pv = ad.AnnData(X=adata_sp.obsm['X_shared'].copy(), obs=adata_sp.obs.copy())
    # adata_sp_pv.obsm['spatial'] = adata_sp.obsm['spatial'].copy()
    # adata_sc_pv = ad.AnnData(X=adata_sc.obsm['X_shared'].copy(), obs=adata_sc.obs.copy())

    params = {'method': 'OTREVISE', 'annotation_key': 'index', 'metric': 'cosine'}
    adata_sc.obs = adata_sc.obs.reset_index().copy()
    sc_sp_contribution, adata_sp_pv, adata_sc_pv = tacco_get(adata_sp, adata_sc, **params)

    overlap_genes = list(adata_sp_pv.var_names.intersection(adata_sc_pv.var_names))
    print(len(overlap_genes))
    if test_gene_idx is None:
        train_gene = overlap_genes
        test_gene = overlap_genes
    else:
        train_gene = overlap_genes[:test_gene_idx]
        test_gene = overlap_genes[test_gene_idx:]

    import pdb; pdb.set_trace()
    for n_neighbors in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        print('-'*10, n_neighbors, '-'*10)
        adata, _ = get_imputed_adata(
            adata_sp_pv, adata_sc_pv[:, test_gene], method=impute_method,
            n_neighbors=n_neighbors, ot_mapping=sc_sp_contribution)
        print("Finish imputation")

        adata_1 = adata_sp_raw[:, overlap_genes]
        adata_2 = adata[:, overlap_genes]

        from revise.metrics import compute_metric

        metrics_df = compute_metric(adata_1, adata_2,
                                    adata_process=False, gene_list=None,
                                    normalize=False)
        metrics_df = compute_metric(adata_1, adata_2,
                                    adata_process=False, gene_list=None,
                                    normalize=True)
