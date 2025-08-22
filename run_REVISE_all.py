import scanpy as sc


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="all")
    parser.add_argument("--input_dir", type=str, default="raw_data", help="input directory")
    parser.add_argument("--task", type=str, default="application", help="task")
    parser.add_argument("--patient_id", type=str, default="P2CRC", help="patient_id")
    parser.add_argument("--data_type", type=str, default="Xenium", help="ST technology")
    
    parser.add_argument("--result_dir", type=str, default="results", help="result directory")
    args = parser.parse_args()
    return args

def main(args):
    input_dir = args.input_dir
    patient_id = args.patient_id
    data_type = args.data_type
    result_dir = args.result_dir

    ## load data ----------
    sc_ref_path = f"{input_dir}/adata_sc_all_reanno.h5ad"
    st_path = f"{input_dir}/{patient_id}_{data_type}.h5ad"

    adata_sp = sc.read_h5ad(st_path)
    adata_sc = sc.read_h5ad(sc_ref_path)

    adata_sc = adata_sc[adata_sc.obs['Patient'] == "P2CRC", :]
    adata_sc.obs = adata_sc.obs[['Level1','Level2']]
    sc.pp.filter_genes(adata_sc, min_cells=30)
    adata_sc.obs['Level1'].replace({"Mono/Macro": "Mono_Macro"}, inplace=True)

    # overlap_genes = adata_sp.var_names.intersection(adata_sc.var_names)
    # adata_sp = adata_sp[:, overlap_genes]
    # adata_sc = adata_sc[:, overlap_genes]


    ## REVISE: 2 steps: 
    # (1) Level1 annotation: 
    # (2) REVISE reconstruction: 3 modes
    #       1. VisiumHD: neighbor enhance
    #       2. Visium: cell type align & distribute into single cells
    #       3. Xenium: single cell align & impute


    ## Step 1 --------
    adata_sp = annotate(adata_sp, adata_sc, "Level1")


    ## Step 2 --------
    if data_type == "VisiumHD":
        select_mode = "mode_1"
    elif data_type == "Visium":
        select_mode = "mode_2"
    elif data_type == "Xenium":
        select_mode = "mode_3"
    
    adata_sp = REVISE_reconstruct(adata_sp, adata_sc, "Level1", select_mode = select_mode)
    adata_sp.write(f"{result_dir}/{patient_id}_{data_type}_REVISE.h5ad")




if __name__ == '__main__':
    args = get_args()
    main(args)