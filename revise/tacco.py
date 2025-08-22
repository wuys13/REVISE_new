import tacco as tc
import pandas as pd
import numpy as np
def assign_max_level(adata, ct_name, print_flag = False):
    # Extract DataFrame from obsm for the specified level
    df = adata.obsm[ct_name]

    # Find the column name and value with the maximum value for each row
    max_columns = df.idxmax(axis=1)
    max_values = df.max(axis=1)
    if print_flag:
        print(np.percentile(max_values.values, [0, 25, 50, 75, 100]))


    result_df = pd.DataFrame({
        'cell_type': max_columns.index,
        ct_name: max_columns.values,
        'max_score': max_values.values,
    })
    result_df.set_index('cell_type', inplace=True)


    # Assign result to adata.obs
    adata.obs[ct_name] = result_df[ct_name]
    adata.obs[ct_name] = adata.obs[ct_name].astype(object)
    adata.obs[ct_name].replace({np.nan: 'Unknown'}, inplace=True)
    adata.obs['max_score'] = result_df['max_score']

    return adata


def tacco_anno(adata_sp, adata_sc, ct_name, multi_center=1, lamb=1e-3, print_flag = False):
    adata_sp_raw = adata_sp.copy()
    # adata_sp.X = adata_sp.X.asty
    tc.tl.annotate(adata_sp, adata_sc, 
                   ct_name, result_key=ct_name,
                   multi_center=multi_center, lamb=lamb)
    adata_sp = assign_max_level(adata_sp, ct_name, print_flag = print_flag)
    adata_sp_raw.obs = adata_sp.obs.copy()
    adata_sp_raw.obsm = adata_sp.obsm.copy()

    return adata_sp_raw