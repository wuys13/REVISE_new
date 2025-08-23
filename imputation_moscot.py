import os
import scanpy as sc
from moscot.problems.space import MappingProblem
from tqdm import tqdm
from scipy import sparse as sp
import numpy as np
from anndata import AnnData

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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


def impute(solutions, var_names, device, adata_sc, adata_sp):
    if var_names is None:
        var_names = adata_sc.var_names

    gexp_sc = adata_sc[:, var_names].X
    if sp.issparse(gexp_sc):
        gexp_sc = gexp_sc.toarray()

    predictions = np.nan_to_num(
        np.vstack(
            [
                np.hstack(
                    [
                        val.to(device=device).pull(x, scale_by_marginals=True)
                        for x in np.array_split(gexp_sc, 1, axis=1)
                    ]
                )
                for val in solutions.values()
            ]
        ),
        nan=0.0,
        copy=False,
    )

    adata_pred = AnnData(X=predictions, obsm=adata_sp.obsm.copy())
    adata_pred.obs_names = adata_sp.obs_names
    adata_pred.var_names = var_names
    return adata_pred


if __name__ == "__main__":
    root_path = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/wuyushuai/REVISE_new/data/spot"
    source_path = os.path.join(root_path, patient_id, f"cut_part{part_num}")
    adata_sp = sc.read(f"{source_path}/selected_xenium.h5ad")
    adata_sc = sc.read(f"{source_path}/real_sc_ref.h5ad")
    overlap_genes = list(adata_sp.var_names.intersection(adata_sc.var_names))
    adata_sc = adata_sc[:, overlap_genes]
    adata_sp = adata_sp[:, overlap_genes]

    adata_sp_raw = adata_sp.copy()
    adata_sc_raw = adata_sc.copy()

    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    sc.tl.pca(adata_sc, n_comps=50)

    sc.pp.normalize_total(adata_sp, target_sum=1e4)
    sc.pp.log1p(adata_sp)
    sc.tl.pca(adata_sp, n_comps=50)

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

    # params = {'method': 'OTREVISE', 'annotation_key': 'index', 'metric': 'cosine'}
    # adata_sc.obs = adata_sc.obs.reset_index().copy()
    # sc_sp_contribution, adata_sp_pv, adata_sc_pv = tacco_get(adata_sp, adata_sc, **params)

    mp = MappingProblem(adata_sc=adata_sc, adata_sp=adata_sp)
    mp = mp.prepare(sc_attr={"attr": "obsm", "key": "X_pca"}, xy_callback="local-pca")
    mp = mp.solve()
    impute_result = []
    tmp_genes = []
    for genes in tqdm(overlap_genes):
        if len(tmp_genes) == 9 or genes == overlap_genes[-1]:
            tmp_genes.append(genes)
            # adata_imputed = mp.impute(var_names=genes)
            adata_imputed = impute(mp.solutions, tmp_genes, "cuda", adata_sc_raw, adata_sp_raw)
            impute_result.append(adata_imputed)
            tmp_genes = []
        else:
            tmp_genes.append(genes)
    adata_imputed = sc.concat(impute_result, axis=1)

    print(adata_imputed)
    if test_gene_idx is None:
        train_gene = overlap_genes
        test_gene = overlap_genes
    else:
        train_gene = overlap_genes[:test_gene_idx]
        test_gene = overlap_genes[test_gene_idx:]

    adata_1 = adata_sp_raw[:, overlap_genes]
    adata_2 = adata_imputed[:, overlap_genes]

    from revise.metrics import compute_metric

    metrics_df = compute_metric(adata_1, adata_2,
                                adata_process=False, gene_list=None,
                                normalize=False)
    metrics_df = compute_metric(adata_1, adata_2,
                                adata_process=False, gene_list=None,
                                normalize=True)
