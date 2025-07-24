import os
import torch

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

from revise.svc import get_cell_contributions, sp_SVC_recon, sc_SVC_recon, optimize_cell_contributions, assign_cell_types, assign_cell_types_easy, get_spot_cell_distribution
from revise.sc_ref import construct_sc_ref, marker_selection, preprocess
from revise.sc_meta import get_sc_obs, get_true_cell_type, get_sc_id
from revise.metrics import compute_metric


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("--sn", type=str, default="xenium", help="sn")
    parser.add_argument("--spot_size", type=int, default=100, help="spot size")

    parser.add_argument("--dropout_prob", type=int, default=0, help="dropout probability")
    parser.add_argument("--dropout_scale", type=int, default=0, help="dropout scale")
    parser.add_argument("--swapping_prob", type=int, default=0, help="swapping probability")
    parser.add_argument("--swapping_scale", type=int, default=0, help="swapping scale")

    parser.add_argument("--use_raw_flag", type=int, default=1, help="use raw xenium")
    parser.add_argument("--real_sc_ref", type=int, default=0, help="use real sc data")
    parser.add_argument("--cell_completeness", type=int, default=1, help="cell completeness")

    parser.add_argument("--lambda_morph", type=float, default=0.0, help="lambda for morphology")
    
    parser.add_argument("--n_epoch", type=int, default=8000, help="number of epoch")
    # parser.add_argument("--image_path", type=str, default="/home/wys/0_test/Tangram-master/Xenium_data/a.tif", help="image path")
    args = parser.parse_args()
    return args



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sn = args.sn
    spot_size = args.spot_size

    dropout_prob = args.dropout_prob
    dropout_scale = args.dropout_scale
    swapping_prob = args.swapping_prob
    swapping_scale = args.swapping_scale
    
    use_raw_flag = args.use_raw_flag
    real_sc_ref = args.real_sc_ref
    n_epoch = args.n_epoch
    lambda_morph = args.lambda_morph

    if args.cell_completeness == 1:
        cell_completeness = True
    else:
        cell_completeness = False

    print(f"sn: {sn}, spot_size: {spot_size}, dropout_prob: {dropout_prob}, dropout_scale: {dropout_scale}, swapping_prob: {swapping_prob}, swapping_scale: {swapping_scale}, use_raw_flag: {use_raw_flag}, real_sc_ref: {real_sc_ref}, cell_completeness: {cell_completeness}, n_epoch: {n_epoch}, lambda_morph: {lambda_morph}")
    
    # lambda_morph=0
    # SVC_obs = None
    # morphology_features = None

    outdir = f"WSI_{sn}/cut"

    if dropout_prob == 0 and swapping_prob == 0:
        print("Use the original data, no noise...")
        spot_path = os.path.join(outdir, f"spot_{spot_size}")
        if use_raw_flag == 1:
            st_path = f"{spot_path}/xenium_spot.h5ad"
        else:
            st_path = f'{spot_path}/simulated_xenium_spot.h5ad'
    else:
        print("Use the noise data...")
        if use_raw_flag == 1:
            spot_path = os.path.join(outdir, f"spot_{spot_size}/noise_raw")
        else:
            spot_path = os.path.join(outdir, f"spot_{spot_size}/noise_simulated")
        st_path = f'{spot_path}/{dropout_prob}_{dropout_scale}_{swapping_prob}_{swapping_scale}.h5ad'


    if real_sc_ref == 1:
        print("Using real sc data")
        sc_ref_path = f"{outdir}/real_sc_ref.h5ad"
        use_raw_flag = 1 # 必须用真实的Xenium数据

    if use_raw_flag == 1:
        # st_path = f"{spot_path}/xenium_spot.h5ad"
        sc_path = f"{outdir}/selected_xenium.h5ad"
    else:
        # st_path = f'{spot_path}/simulated_xenium_spot.h5ad'
        sc_path = f"{outdir}/simulated_xenium.h5ad"
    
    sc_ref_path = None
    morphology_path = f"{outdir}/select_cell_features.csv"
    morphology_path = f"{outdir}/morphology_features.csv"

    # 最后两个是 [是否用真实Xenium数据] 和 [是否用真实 sc_ref 数据]
    result_dir=f"0_revise_results/{spot_size}_{use_raw_flag}_{real_sc_ref}/{dropout_prob}_{dropout_scale}_{swapping_prob}_{swapping_scale}"
    os.makedirs(result_dir, exist_ok=True)


    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)
    if sc_ref_path is not None:
        adata_sc_ref = sc.read_h5ad(sc_ref_path)
    else:
        adata_sc_ref = adata_sc.copy() # 用它本身

    morphology_features = pd.read_csv(morphology_path, index_col=0)
    
    SVC_obs = get_sc_obs(adata_st.obs.index, adata_st.uns['all_cells_in_spot'])
    SVC_obs = get_true_cell_type(SVC_obs, adata_sc)
    print(SVC_obs)

    if cell_completeness:
        complete_mask = None
    else:
        print("Cell is not complete, using cell_completeness mask")
        # todo: which cell is not complete?
        # complete_mask = np.zeros((adata_st.shape[0], adata_sc.shape[0]), dtype=np.float32)

    key_type = 'clusters'
    if key_type not in adata_sc_ref.obs.keys():
        print(f"Key {key_type} not found in adata_sc.obs, using 'Level1' instead.")
        adata_sc_ref.obs[key_type] = adata_sc_ref.obs['Level1'].astype(str)
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
    cell_contributions_file = f"{result_dir}/cell_contributions.csv"
    if os.path.exists(cell_contributions_file):
        cell_contributions = pd.read_csv(f"{result_dir}/cell_contributions.csv", index_col=0)
    else:
        cell_contributions = get_cell_contributions(adata_st_marker, adata_sc_marker, sc_ref, key_type, device=device,
                                                cells_on_spot = SVC_obs, morphology_features = morphology_features, feature_list=None, lambda_morph=lambda_morph,
                                                n_epoch=n_epoch, adam_params=None, batch_prior=2,
                                                plot_file_name = "a.png")
        cell_contributions.to_csv(f"{result_dir}/cell_contributions.csv")

    print(cell_contributions.values.max(axis=1))
    

    # PM_on_cell = cal_prob_bayes(cell_contributions, cells_on_spot = SVC_obs,
    #                              morphology_features = morphology_features, feature_list = None, scale = True)
    # PM_on_cell.to_csv(f"{result_dir}/PM_on_cell.csv")
    PM_on_cell = pd.read_csv(f"./WSI_xenium/cut/PM_on_cell.csv", index_col=0)


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
    adata_st_orig.var_names_make_unique()
    sc.pp.normalize_total(adata_st_orig, target_sum=1e4)
    sc.pp.normalize_total(adata_sc_orig, target_sum=1e4)
    print("Constructing SVC adata...")
    SVC_adata = sp_SVC_recon(adata_st = adata_st_orig, sc_ref = sc_ref_all,
                           cell_contributions = cell_contributions, SVC_obs = SVC_obs,
                           complete_mask = complete_mask)
    print(SVC_adata)
    # SVC_adata.write(f"{result_dir}/SVC_adata.h5ad")
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