import os
import torch

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

from revise.svc import get_cell_contributions, SVC_recon, sc_SVC_recon, optimize_cell_contributions, assign_cell_types, assign_cell_types_easy, get_spot_cell_distribution
from revise.sc_ref import construct_sc_ref, marker_selection, preprocess, cut_off_low_gene
from revise.sc_meta import get_sc_obs, get_true_cell_type, get_sc_id
from revise.metrics import compute_metric
from revise.morphology import get_similarity_df


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("--task", type=str, default="spot", help="task")
    parser.add_argument("--spot_size", type=int, default=200, help="spot size")
    parser.add_argument("--patient_id", type=str, default="P2CRC", help="patient_id")

    parser.add_argument("--use_raw_flag", type=int, default=1, help="use raw xenium")
    parser.add_argument("--real_sc_ref", type=int, default=1, help="use real sc data")
    parser.add_argument("--cell_completeness", type=int, default=1, help="cell completeness")

    parser.add_argument("--lambda_morph", type=float, default=0.0, help="lambda for morphology")
    
    parser.add_argument("--n_epoch", type=int, default=4000, help="number of epoch")
    parser.add_argument("--celltype_col", type=str, default="Level1", help="cell type column")
    parser.add_argument("--part", type=str, default="part1", help="part")
    parser.add_argument("--result_dir", type=str, default="results", help="result directory")
    args = parser.parse_args()
    return args



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    task = args.task
    spot_size = args.spot_size
    patient_id = args.patient_id
    
    use_raw_flag = args.use_raw_flag
    real_sc_ref = args.real_sc_ref
    n_epoch = args.n_epoch
    lambda_morph = args.lambda_morph

    celltype_col = args.celltype_col
    part = args.part
    result_dir = args.result_dir

    if args.cell_completeness == 1:
        cell_completeness = True
    else:
        cell_completeness = False

    
    input_dir = f"data/{task}/{patient_id}/cut_{part}"

    spot_path = os.path.join(input_dir, f"spot_{spot_size}")
    if use_raw_flag == 1:
        st_path = f"{spot_path}/xenium_spot.h5ad"
    else:
        st_path = f'{spot_path}/simulated_xenium_spot.h5ad'

    sc_ref_path = None
    if real_sc_ref == 1:
        print("Using real sc data")
        sc_ref_path = f"{input_dir}/real_sc_ref.h5ad"
        use_raw_flag = 1 # 必须用真实的Xenium数据

    if use_raw_flag == 1:
        sc_path = f"{input_dir}/selected_xenium.h5ad"
    else:
        sc_path = f"{input_dir}/simulated_xenium.h5ad"
    
    morphology_path = f"{input_dir}/select_cell_features.csv"
    morphology_path = f"{input_dir}/morphology_features.csv"
    morphology_path = None

    result_dir=f"{result_dir}/{task}/{patient_id}/{part}/{spot_size}_{use_raw_flag}_{real_sc_ref}"
    os.makedirs(result_dir, exist_ok=True)


    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)
    if sc_ref_path is not None:
        adata_sc_ref = sc.read_h5ad(sc_ref_path)
    else:
        adata_sc_ref = adata_sc.copy() # 用它本身

    if morphology_path is not None:
        morphology_features = pd.read_csv(morphology_path, index_col=0)
    else:
        morphology_features = None
    

    SVC_obs = get_sc_obs(adata_st.obs.index, adata_st.uns['all_cells_in_spot'])
    SVC_obs = get_true_cell_type(SVC_obs, adata_sc)
    print(SVC_obs)

    if cell_completeness:
        complete_mask = None
    else:
        print("Cell is not complete, using cell_completeness mask")
        # todo: which cell is not complete?
        # complete_mask = np.zeros((adata_st.shape[0], adata_sc.shape[0]), dtype=np.float32)

    key_type = "clusters"
    if key_type not in adata_sc_ref.obs.columns:
        adata_sc_ref.obs[key_type] = adata_sc_ref.obs[celltype_col].astype(str)
    type_list = sorted(list(adata_sc_ref.obs[key_type].unique().astype(str)))
    print(f'There are {len(type_list)} cell types: {type_list}')

    adata_st.var_names_make_unique()
    adata_st.obsm['spatial'] = adata_st.obsm['spatial'].astype(np.int32)

    adata_sc_ref, adata_st = preprocess(adata_sc_ref, adata_st)
    marker_gene_dict = marker_selection(adata_sc_ref, key_type=key_type, return_dict=True, 
                                                          n_select=50, threshold_p=0.1, threshold_fold=1.2,
                                                          q=0.3)
    marker_gene = []
    marker_gene_label = []
    for type_ in type_list:
        marker_gene.extend(marker_gene_dict[type_])
        marker_gene_label.extend([type_]*len(marker_gene_dict[type_]))
    marker_gene_df = pd.DataFrame({'gene':marker_gene, 'label':marker_gene_label})
    marker_gene_df.to_csv(f"{result_dir}/marker_gene.csv")

    # Filter scRNA and spatial matrices with marker genes
    adata_sc_marker = adata_sc_ref[:, marker_gene]
    adata_st_marker = adata_st[:, marker_gene]

    print(marker_gene_df['label'].value_counts())

    sc_ref = construct_sc_ref(adata_sc_marker, key_type=key_type, type_list = type_list)
    sc_ref_all = construct_sc_ref(adata_sc_ref, key_type=key_type, type_list = type_list)
    print(sc_ref.shape, sc_ref_all.shape)

    # X = np.array(adata_st_marker.X)

    # 得到 Spot-level 的 cell_contributions，以及 cell-level 的 PM_on_cell based on morphology
    print("Calculating cell contributions and PM_on_cell...")
    cell_contributions_file = f"{result_dir}/cell_contributions_{celltype_col}.csv"
    not_run_again = False
    if os.path.exists(cell_contributions_file) and not_run_again:
        cell_contributions = pd.read_csv(cell_contributions_file, index_col=0)
    else:
        from revise.svc import tacco_get
        cell_contributions = tacco_get(adata_st, adata_sc)

    print(cell_contributions.values.max(axis=1))
    

    PM_on_cell_file = f"{input_dir}/PM_on_cell.csv"
    if os.path.exists(PM_on_cell_file):
        PM_on_cell = pd.read_csv(PM_on_cell_file, index_col=0)
    else:
        PM_on_cell = get_similarity_df(adata_sc)
        PM_on_cell.to_csv(PM_on_cell_file)

    SVC_obs = assign_cell_types_easy(SVC_obs, cell_contributions, mode="max")
    SVC_obs['match'] = SVC_obs['cell_type'] == SVC_obs['true_cell_type']
    max_match = sum(SVC_obs['match'] == True) / len(SVC_obs)
    print(SVC_obs['match'].value_counts(), max_match)
    SVC_obs.to_csv(f"{result_dir}/SVC_obs.csv")


    # # 根据 in-spot cell count 来优化 cell_contributions
    # print("Optimizing cell contributions...")
    # cell_contributions = optimize_cell_contributions(cell_contributions = cell_contributions, 
    #                                                  SVC_obs = SVC_obs, cell_completeness = cell_completeness)
    # print(cell_contributions.values.max(axis=1))

    # SVC_obs = assign_cell_types_easy(SVC_obs, cell_contributions, mode="max")
    # SVC_obs['match'] = SVC_obs['cell_type'] == SVC_obs['true_cell_type']
    # print(SVC_obs['match'].value_counts())

    spot_cell_distribution = get_spot_cell_distribution(cell_contributions = cell_contributions, 
                                                      SVC_obs = SVC_obs, cell_completeness = cell_completeness)


    # 结合 morphology 得到的 PM_on_cell 和 cell_contributions 来得到 cell_type_dict
    if spot_size > 30:
        print("Assigning cell types...")
        SVC_obs = assign_cell_types(SVC_obs = SVC_obs, PM_on_cell = PM_on_cell, 
                                        spot_cell_distribution = spot_cell_distribution
                                        )
        print(SVC_obs)
        SVC_obs['match'] = SVC_obs['cell_type'] == SVC_obs['true_cell_type']
        SVC_obs.to_csv(f"{result_dir}/SVC_obs_assign.csv")

        enhance_match = sum(SVC_obs['match'] == True) / len(SVC_obs)
        print(SVC_obs['match'].value_counts(), enhance_match)

        match_matrix = pd.DataFrame({
            "max_match": [max_match],
            "enhance_match": [enhance_match],
        })
        match_matrix.to_csv(f"{result_dir}/match_matrix.csv", index=False)
    

    adata_sc_orig = sc.read_h5ad(sc_path)
    adata_st_orig = sc.read_h5ad(st_path)
    adata_sc_orig = adata_sc_orig[:, sc_ref_all.columns]
    adata_st_orig = adata_st_orig[:, sc_ref_all.columns]

    adata_st_orig.var_names_make_unique()
    sc.pp.normalize_total(adata_st_orig, target_sum=1e4)
    sc.pp.normalize_total(adata_sc_orig, target_sum=1e4)
    print("Constructing SVC adata...")
    
    # gene_cutoff_ratio = None
    # # gene_cutoff_ratio = 0.1
    # if gene_cutoff_ratio is not None:
    #     sc_ref_all = cut_off_low_gene(sc_ref_all, gene_cutoff_ratio, mode="self")
    
    SVC_adata = SVC_recon(adata_st = adata_st_orig, sc_ref = sc_ref_all,
                           cell_contributions = cell_contributions, SVC_obs = SVC_obs,
                           complete_mask = complete_mask)
    print(SVC_adata)
    SVC_adata.write(f"{result_dir}/SVC_adata.h5ad")
    # print(SVC_adata.X[:5, :5])

    gene_names = adata_sc_orig.var_names.intersection(SVC_adata.var_names)
    print(len(adata_sc_orig.var_names), len(SVC_adata.var_names), len(gene_names))

    # sc_list = get_sc_id(SVC_adata.obs['spot_name'].values, adata_st_orig.uns['all_cells_in_spot'])
    sc_list = SVC_obs['cell_id'].values
    adata_sc_orig.obs.index = adata_sc_orig.obs['cell_id'].values
    adata_sc_orig = adata_sc_orig[sc_list, gene_names]
    # sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    # sc.pp.log1p(adata)
    SVC_adata = SVC_adata[:, gene_names]

    # log
    metrics_df = compute_metric(adata_sc_orig, SVC_adata, 
                            adata_process=True, gene_list=None,
                            normalize=False)
    metrics_df = compute_metric(adata_sc_orig, SVC_adata, 
                            adata_process=True, gene_list=None,
                            normalize=True)
    
    print("\n")
    metrics_df = compute_metric(adata_sc_orig, SVC_adata, 
                                adata_process=False, gene_list=None,
                                normalize=False)
    metrics_df.to_csv(f"{result_dir}/metrics.csv", index=False)
    metrics_df = compute_metric(adata_sc_orig, SVC_adata, 
                                adata_process=False, gene_list=None,
                                normalize=True)
    metrics_df.to_csv(f"{result_dir}/metrics_normalized.csv", index=False)

    
    print("Finish")
    
if __name__ == "__main__":
    args = get_args()
    main(args)