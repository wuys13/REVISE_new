import os
import time
import numpy as np
import anndata

def simulation(adata_st: anndata.AnnData, adata_sc: anndata.AnnData, key_type: str, cell_proportion: np.ndarray,
               n_cell=10, batch_effect_sigma=0.1, zero_proportion=0.3, additive_noise_sigma=0.05, save=True,
               out_dir='', filename="ST_Simulated.h5ad", verbose=0):
    """
    Simulation of the spatial transcriptomics data based on a real spatial sample and deconvolution results 

    Args:
        adata_st: Original spatial transcriptomics data.
        adata_sc: Original single-cell data.
        key_type: The key that is used to extract cell type information from adata_sc.obs.
        cell_proportion: Proportion of each cell type obtained by the deconvolution.
        n_cell: Number of cells in each spot, either a key of adata_st.obs or a positive integer.
        batch_effect_sigma: Sigma of the log-normal distribution when generate batch effect.
        zero_proportion: Proportion of gene expression set to 0. Note that since some gene expression in the original
            X is already 0, the final proportion of 0 gene read is larger than zero_proportion.
        additive_noise_sigma: Sigma of the log-normal distribution when generate additive noise.
        save: If True, save the generated adata_st as a file.
        out_dir: Output directory.
        filename: Name of the saved file.
        verbose: Whether print the time spend.
    Returns:
        Simulated ST Anndata.
    """
    time_start = time.time()
    # Construct ground truth
    type_list = sorted(list(adata_sc.obs[key_type].unique()))  # list of the cell type.
    assert len(type_list) == cell_proportion.shape[1]
    assert len(cell_proportion) == len(adata_st)
    n_spot = len(adata_st)
    n_type = len(type_list)
    if isinstance(n_cell, (int, float)):
        assert n_cell >= 1
        n_cell = np.array([int(n_cell)] * n_spot)
    else:
        n_cell = adata_st.obs[n_cell].values.astype(int)
        n_cell[n_cell <= 0] = 1
    cell_count = np.zeros(cell_proportion.shape)
    for i in range(n_spot):
        cell_count[i] = proportion_to_count(cell_proportion[i], n_cell[i])
    cell_count = cell_count.astype(int)
    adata_st.obsm['ground_truth'] = cell_count
    adata_st.obs['cell_count'] = n_cell
    adata_st.uns['type_list'] = type_list
    if verbose:
        print('Prepared the ground truth. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

    # Construct expression matrix
    common_genes = list(set(adata_st.var_names).intersection(set(adata_sc.var_names)))
    adata_sc = adata_sc[:, common_genes]
    adata_st = adata_st[:, common_genes]
    n_gene = len(common_genes)
    X = np.zeros(adata_st.shape)
    Y = adata_sc.X if type(adata_sc.X) is np.ndarray else adata_sc.X.toarray()
    Y = Y * 1e6 / np.sum(Y, axis=1, keepdims=True)
    type_index = []
    for i in range(n_type):
        type_index.append(np.where(adata_sc.obs[key_type] == type_list[i])[0])
    for i in range(n_spot):
        for j in range(n_type):
            if cell_count[i, j] > 0:
                X_temp = np.array(np.sum(Y[np.random.choice(type_index[j], cell_count[i, j])], axis=0))
                if X_temp.ndim > 1:
                    X[i] += X_temp[0]
                else:
                    X[i] += X_temp
    if verbose:
        print('Constructed the ground truth. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

    # Add noise
    # Batch effect
    batch_effect = np.random.lognormal(0, batch_effect_sigma, size=n_gene)
    X = X * batch_effect
    # Zero reads
    zero_read = np.random.binomial(1, 1 - zero_proportion, size=X.shape)
    X = X * zero_read
    # Additive noise
    additive_noise = np.random.lognormal(0, additive_noise_sigma, size=X.shape)
    X = X * additive_noise
    adata_st.X = X
    adata_st.uns['batch_effect'] = batch_effect
    if verbose:
        print('Added batch effect, zero reads, and additive noise. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

    if save:
        if out_dir and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = out_dir + '/' if out_dir else ''
        adata_st.raw = None
        adata_st.write(out_dir + filename)
        if verbose:
            print('Saved the simulated data to file. Time use {:.2f}'.format(time.time() - time_start))

    return adata_st


def proportion_to_count(p, n, multiple_spots=False):
    """
    Convert the cell proportion to the absolute cell number.

    Args:
        p: Cell proportion.
        n: Number of cells.
        multiple_spots: If the data is related to multiple spots

    Returns:
        Cell count of each cell type.
    """
    if multiple_spots:
        assert len(p) == len(n)
        count = np.zeros(p.shape)
        for i in range(len(p)):
            count[i] = proportion_to_count(p[i], n[i])
    else:
        assert (np.abs(np.sum(p) - 1) < 1e-5)
        c0 = p * n
        count = np.floor(c0)
        r = c0 - count
        if np.sum(count) == n:
            return count
        idx = np.argsort(r)[-int(np.round(n - np.sum(count))):]
        count[idx] += 1
    return count
