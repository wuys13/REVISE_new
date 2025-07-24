import numpy as np
import pandas as pd

def get_sc_obs(spot_names, all_cells_in_spot):
    sc_obs = pd.DataFrame()
    for i in spot_names:
        sc_ids = all_cells_in_spot[i]
        spot_sc_obs = pd.DataFrame({'spot_name': [i]*len(sc_ids), 'cell_id': sc_ids})
        sc_obs = pd.concat([sc_obs, spot_sc_obs], axis=0)
    sc_obs.reset_index(drop=True, inplace=True)

    return sc_obs

def get_true_cell_type(SVC_obs, adata_sc):
    
    true_cell_type_df = pd.DataFrame(adata_sc.obs)
    true_cell_type_df.set_index('cell_id', inplace=True)
    if 'clusters' not in true_cell_type_df.columns:
        true_cell_type_df['clusters'] = true_cell_type_df['Level1']
    
    SVC_obs['true_cell_type'] = true_cell_type_df.loc[SVC_obs['cell_id'], 'clusters'].values
    SVC_obs[["x", "y"]] = true_cell_type_df.loc[SVC_obs["cell_id"], ["x", "y"]].values
    SVC_obs['cell_id'] = SVC_obs['cell_id'].astype(str)

    return SVC_obs


def get_sc_id(spot_names, all_cells_in_spot):
    """
    Get the cell ids in each spot
    """
    sc_list = []
    for i in spot_names:
        sc_ids = all_cells_in_spot[i]
        sc_list.append(sc_ids[0])

    print(len(sc_list))

    return np.array(sc_list)

