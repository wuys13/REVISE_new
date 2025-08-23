import os
import torch

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

from revise.svc import get_cell_contributions, SVC_recon, sp_SVC_recon, sc_SVC_recon, optimize_cell_contributions, assign_cell_types, assign_cell_types_easy, get_spot_cell_distribution
from revise.sc_ref import construct_sc_ref, marker_selection, preprocess, cut_off_low_gene
from revise.sc_meta import get_sc_obs, get_true_cell_type, get_sc_id
from revise.metrics import compute_metric
from revise.morphology import get_similarity_df
from revise.svc import tacco_get
torch.set_float32_matmul_precision("high")

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("--task", type=str, default="seg", help="task")
    parser.add_argument("--spot_size", type=int, default=1, help="seg")
    parser.add_argument("--patient_id", type=str, default="P2CRC", help="patient_id")
    parser.add_argument("--iteration", type=int, default=0, help="iteration")

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
    iteration = args.iteration
    
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

    print(f"task: {task}, spot_size: {spot_size}, use_raw_flag: {use_raw_flag}, real_sc_ref: {real_sc_ref}, cell_completeness: {cell_completeness}, n_epoch: {n_epoch}, lambda_morph: {lambda_morph}")
    

    input_dir = f"data/{task}/{patient_id}/cut_{part}/{iteration}"

    spot_path = os.path.join(input_dir, f"spot_{spot_size}")
    if use_raw_flag == 1:
        st_path = f"{spot_path}/xenium_spot.h5ad"
    else:
        st_path = f'{spot_path}/simulated_xenium_spot.h5ad'


    if real_sc_ref == 1:
        print("Using real sc data")
        sc_ref_path = f"{input_dir}/real_sc_ref.h5ad"
        use_raw_flag = 1 # 必须用真实的Xenium数据

    if use_raw_flag == 1:
        sc_path = f"{input_dir}/selected_xenium.h5ad"
    else:
        sc_path = f"{input_dir}/simulated_xenium.h5ad"
    
    sc_ref_path = None
    morphology_path = f"{input_dir}/select_cell_features.csv"
    morphology_path = f"{input_dir}/morphology_features.csv"
    morphology_path = None

    result_dir=f"{result_dir}/{task}/{patient_id}/{part}/{iteration}/{spot_size}_{use_raw_flag}_{real_sc_ref}"
    os.makedirs(result_dir, exist_ok=True)

    st_path = f"/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/wuyushuai/store_data/CRC_processed/{args.patient_id}_HD.h5ad"
    sc_path = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/wuyushuai/store_data/CRC_processed/adata_sc_all_reanno.h5ad"

    adata_sc = sc.read_h5ad(sc_path)
    adata_sc = adata_sc[adata_sc.obs['Patient'] == args.patient_id]
    adata_st = sc.read_h5ad(st_path)
    if sc_ref_path is not None:
        adata_sc_ref = sc.read_h5ad(sc_ref_path)
    else:
        adata_sc_ref = adata_sc.copy() # 用它本身

    if morphology_path is not None:
        morphology_features = pd.read_csv(morphology_path, index_col=0)
    else:
        morphology_features = None



    if cell_completeness:
        complete_mask = None
    else:
        print("Cell is not complete, using cell_completeness mask")
        

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



    # 得到 Spot-level 的 cell_contributions，以及 cell-level 的 PM_on_cell based on morphology
    print("Calculating cell contributions...")
    not_run_again = False
    not_run_again = True
    dec_mode = "tacco" # vi/tacco/destvi/tacco_gft
    cell_contributions_file = f"{result_dir}/cell_contributions_{celltype_col}_{dec_mode}.csv"

    if os.path.exists(cell_contributions_file) and not_run_again:
        cell_contributions = pd.read_csv(cell_contributions_file, index_col=0)
        print(f"read cell contributions from {cell_contributions_file}")
    else:
        if dec_mode == "vi":
            SVC_obs = get_sc_obs(adata_st.obs.index, adata_st.uns['all_cells_in_spot'])
            SVC_obs = get_true_cell_type(SVC_obs, adata_sc)
            print(SVC_obs)
            cell_contributions = get_cell_contributions(adata_st_marker, adata_sc_marker, sc_ref, key_type, device=device,
                                                    cells_on_spot = SVC_obs, morphology_features = morphology_features, feature_list=None, lambda_morph=lambda_morph,
                                                    n_epoch=n_epoch, adam_params=None, batch_prior=2,
                                                    plot_file_name = f"{result_dir}/1.png")
        elif dec_mode == "tacco":

            # params = {'method': 'OTGFT', 'multi_center': 10, "S_spagft": 24, "lamada_mtb": 0.8}
            params = {'method': 'OT'}
            print(f"input st data shape: {adata_st_marker.shape}, input sc data shape: {adata_sc_marker.shape}")
            cell_contributions = tacco_get(adata_st, adata_sc, **params)
        elif dec_mode == "tacco_gft":
            params = {'method': 'OTGFT', 'multi_center': 10, "S_spagft": 24, "lamada_mtb": 0.8}
            print(f"input st data shape: {adata_st_marker.shape}, input sc data shape: {adata_sc_marker.shape}")
            cell_contributions = tacco_get(adata_st, adata_sc, **params)
        elif dec_mode == "tacco_revise":
            params = {'method': 'OTREVISE'}
            cell_contributions = tacco_get(adata_st, adata_sc, **params)
        elif dec_mode == "destvi":
            from scvi.model import CondSCVI, DestVI
            adata_sc = adata_sc.copy()

            CondSCVI.setup_anndata(adata_sc, labels_key="Level1")
            sc_model = CondSCVI(adata_sc, weight_obs=False)
            sc_model.train(max_epochs=100)
            sc.pp.filter_cells(adata_st, min_counts=1)
            DestVI.setup_anndata(adata_st)
            st_model = DestVI.from_rna_model(adata_st, sc_model)
            st_model.train(max_epochs=100)
            cell_contributions = st_model.get_proportions()
        else:
            raise NotImplementedError(f"wrong dec_mode: {dec_mode}")

        cell_contributions.to_csv(cell_contributions_file)
        print(f"save cell contributions to {cell_contributions_file}")

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


    spot_cell_distribution = get_spot_cell_distribution(cell_contributions = cell_contributions, 
                                                      SVC_obs = SVC_obs, cell_completeness = cell_completeness)


    # 结合 morphology 得到的 PM_on_cell 和 cell_contributions 来得到 cell_type_dict
    print("Assigning cell types...")
    SVC_obs_raw = SVC_obs.copy()
    SVC_obs = assign_cell_types(SVC_obs = SVC_obs, PM_on_cell = PM_on_cell, 
                                    spot_cell_distribution = spot_cell_distribution
                                    )
    print(SVC_obs)
    SVC_obs['match'] = SVC_obs['cell_type'] == SVC_obs['true_cell_type']
    SVC_obs.to_csv(f"{result_dir}/SVC_obs_assign.csv")

    enhance_match = sum(SVC_obs['match'] == True) / len(SVC_obs)
    print(SVC_obs['match'].value_counts(), enhance_match)
    if enhance_match < max_match:
        SVC_obs = SVC_obs_raw.copy()

    match_matrix = pd.DataFrame({
        "max_match": [max_match],
        "enhance_match": [enhance_match],
    })
    match_matrix.to_csv(f"{result_dir}/match_matrix.csv", index=False)
    




    ## check achor acc
    max_vals = cell_contributions.max(axis=1)
    max_column_names = cell_contributions.idxmax(axis=1)

    cell_contributions_df = pd.DataFrame({
        "Confidence": max_vals,
        "Max_type": max_column_names
    })
    # match_score_list = []
    # for s in cell_contributions.index:
    #     match_score = (SVC_obs[SVC_obs['spot_name'] == s]["true_cell_type"] == cell_contributions_df.loc[s, "Max_type"]).mean()
    #     match_score_list.append(match_score)

    # cell_contributions_df['Match_score'] = match_score_list
    # cell_contributions_df.to_csv(f"{result_dir}/cell_contributions_df.csv")

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.scatter(cell_contributions_df['Confidence'], cell_contributions_df['Match_score'], color='blue', marker='o')
    # plt.xlabel('Confidence')
    # plt.ylabel('Match Score')
    # plt.savefig(f"{result_dir}/distribution_relation.png")
    # ##


    


    adata_sc_orig = sc.read_h5ad(sc_path)
    adata_st_orig = sc.read_h5ad(st_path)
    adata_st_orig.var_names_make_unique()
    adata_sc_orig.obs['total_counts'] = adata_sc_orig.X.sum(axis=1).A1


    ## 重置那些 Diminishing 和 Expanding 的 cell
    dropout_total_counts = 60
    swapping_total_counts = 300
    lower_ts = 0.2
    upper_ts = 0.8
    
    adata_st_orig.obs['Max_type'] = cell_contributions_df['Max_type']
    adata_st_orig.obs['Confidence'] = cell_contributions_df['Confidence']
    adata_st_orig.obs[['x', 'y']] = adata_st_orig.obsm['spatial']
    adata_st_orig.obs['total_counts'] = adata_st_orig.X.sum(axis=1).A1
    b = adata_st_orig.obs.copy()
    c = b[b["total_counts"] > swapping_total_counts]
    d = b[b["total_counts"] < swapping_total_counts]
    # 计算 Diminishing Ratio
    diminishing_condition_1 = (b['total_counts'] < dropout_total_counts) & (b['seg_error'] == "Diminishing")
    diminishing_condition_2 = (b['total_counts'] > dropout_total_counts) & (b['Confidence'] < lower_ts)  & (b['seg_error'] == "Diminishing")
    diminishing_ratio = (diminishing_condition_1.sum() + diminishing_condition_2.sum()) / (b['seg_error'] == "Diminishing").sum()

    # 计算 Expanding Ratio
    expanding_condition = (b['total_counts'] > swapping_total_counts) & (b['Confidence'] < upper_ts) & (b['seg_error'] == "Expanding")
    expanding_ratio = expanding_condition.sum() / (b['seg_error'] == "Expanding").sum()

    # 计算 Unchanged Ratio
    unchanged_condition_1 = (b['total_counts'] > swapping_total_counts) & (b['Confidence'] > upper_ts) & (b['seg_error'] == "Unchanged")
    unchanged_condition_2 = (b['total_counts'] < swapping_total_counts) & (b['Confidence'] > lower_ts) & (b['seg_error'] == "Unchanged")
    unchanged_ratio = (unchanged_condition_1.sum() + unchanged_condition_2.sum()) / (b['seg_error'] == "Unchanged").sum()

    # 打印结果
    print("Final recovery rates:")
    print(f"Diminishing Ratio: {diminishing_ratio}")
    print(f"Expanding Ratio: {expanding_ratio}")
    print(f"Unchanged Ratio: {unchanged_ratio}")

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    palettes = {
    'Diminishing': '#8ECFC9',
    'Expanding': '#FFBE7A',
    'Unchanged': '#82B0D2'
    }
    bins=20
    sns.histplot(data=b, x='Confidence', hue='seg_error', bins=bins, palette=palettes, kde=True, ax=ax1)
    sns.histplot(data=c, x='Confidence', hue='seg_error', bins=bins, palette=palettes, kde=True, ax=ax2)
    sns.histplot(data=d, x='Confidence', hue='seg_error', bins=bins, palette=palettes, kde=True, ax=ax3)

    ax1.set_title('All Cells')
    ax2.set_title(f'Cells with Total Counts > {swapping_total_counts}')
    ax3.set_title(f'Cells with Total Counts < {swapping_total_counts}')

    plt.tight_layout()
    plt.savefig(f"{result_dir}/segmentation.pdf")


    no_effect_indices = ( (b['total_counts'] > swapping_total_counts) & (b['Confidence'] > upper_ts) ) | ( (b['total_counts'] < swapping_total_counts) & (b['Confidence'] > lower_ts) )
    print(f"no_effect_indices ratio : {no_effect_indices.sum()} / {len(b)}")
    adata_st_orig.obs['no_effect'] = no_effect_indices
    
    minor_CF = True
    if minor_CF:
        from revise.minor_CF import replace_effect_spots
        adata_st_orig = replace_effect_spots(adata_st_orig, celltype_col="Level1", no_effect_col="no_effect", )


        
    sc.pp.normalize_total(adata_st_orig, target_sum=1e4)
    sc.pp.normalize_total(adata_sc_orig, target_sum=1e4)
    metrics_df = compute_metric(adata_sc_orig, adata_st_orig, 
                                adata_process=False, gene_list=None,
                                normalize=True)
    metrics_df.to_csv(f"{result_dir}/sp_metrics_normalized.csv")

    print("Finish")
    
if __name__ == "__main__":
    args = get_args()
    main(args)