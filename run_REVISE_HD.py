import argparse
from conf.visium_hd_conf import VisiumHDConf
import scanpy as sc

from revise.sp_svc import SpSVC


def preprocess(adata):
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=30)
    replace_columns = {k: k.replace("/", "_") for k in adata.obs['Level1'].unique().tolist() if '/' in k}
    adata.obs['Level1'].replace(replace_columns, inplace=True)
    replace_columns = {k: k.replace("/", "_") for k in adata.obs['Level2'].unique().tolist() if '/' in k}
    adata.obs['Level2'].replace(replace_columns, inplace=True)
    return adata

def parse_args():
    parser = argparse.ArgumentParser(description="VisiumHD REVISE")
    parser.add_argument("--patient_id", type=str, help="patient_id")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = VisiumHDConf(args.patient_id)
    adata_st = sc.read_h5ad(config.st_file)
    adata_sc = sc.read_h5ad(config.sc_file)
    adata_sc = adata_sc[adata_sc.obs['Patient'] == config.patient_id, :]
    print(f"ST data shape: {adata_st.shape}, SC data shape: {adata_sc.shape}")
    adata_sc = preprocess(adata_st)
    sp_svc = SpSVC(adata_st, adata_sc, config)
    sp_svc.annotate()
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()