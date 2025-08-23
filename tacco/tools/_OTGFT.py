import numpy as np
import pandas as pd
import anndata as ad
import gc
import scanpy as sc
from .. import get
from .. import preprocessing
from .. import utils
from ..utils import _math
from ..utils._utils import _run_OT
from . import _helper as helper
from ._annotate import annotate
from scipy.sparse import issparse, csr_matrix
from numba import njit, prange
import SpaGFT as spg

@njit(parallel=True,cache=True)
def _get_minimal_transitions(vis, vos):
    transitions = np.zeros((vis.shape[0],vis.shape[1],vos.shape[1]))
    for i in prange(vis.shape[0]):
        vi = vis[i]
        vo = vos[i]
        delta = vo - vi
        delta_p = delta * (delta > 0)
        delta_m = delta * (delta < 0)
        sum_p = delta_p.sum()
        sum_m = delta_m.sum()
        if sum_m != 0 and sum_p != 0:
            delta_m *= sum_p/sum_m
            transitions[i] += np.outer(delta_m, delta_p * (1 / sum_p))
        transitions[i] += np.diag(vi-delta_m)
    return transitions

def get_minimal_transitions(
    aa,
    bb,
):
    assert(aa.shape==bb.shape)
    if issparse(aa):
        aa = aa.A
    if issparse(bb):
        bb = bb.A
    
    res = _get_minimal_transitions(aa,bb)
    
    return res

def _annotate_OTGFT(
    adata,
    reference,
    annotation_key=None,
    annotation_prior=None,
    epsilon=5e-3,
    lamb=0.2,
    decomposition=False,
    deconvolution=False,
    **kw_args,
    ):

    """\
    Implements the functionality of :func:`~annotate_OT` without data
    integrity checks.
    """

    cell_prior = helper.prep_cell_priors(adata, reads=True)

    type_cell_dist = helper.prep_distance(adata, reference, annotation_key, decomposition=decomposition, deconvolution=deconvolution, **kw_args)

    if decomposition:
        type_cell_dist, mixtures = type_cell_dist
    types = type_cell_dist.index
    
    if decomposition: # include the annotation profiles themselves to obtain a measurement of confusion
        test_weight = 1e-6*cell_prior.sum()/(len(type_cell_dist.columns)-len(cell_prior.index)) # keep a low weight to not influence the actual typing (much)
        cell_prior = cell_prior.reindex(index=type_cell_dist.columns,fill_value=test_weight)

    
    cell_type = _run_OT(type_cell_dist, annotation_prior, cell_prior=cell_prior, epsilon=epsilon, lamb=lamb)
    cell_type_read_csv = cell_type.copy(deep=True)

    if decomposition:
        cell_type /= cell_type.sum(axis=1).to_numpy()[:,None]
        # need to run measurement on mixtures instead of pure profiles to get the confusion as pure profiles should never be confused at all.
        # miXture Measurements
        xm = cell_type.loc[~cell_type.index.isin(adata.obs.index)].to_numpy()
        
        # Observation Measurements
        om = cell_type.loc[adata.obs.index].to_numpy()
        # miXture Annotation joint probability distribution
        xa = mixtures.copy()
        # get probability of miXture given Annotation
        utils.col_scale(xa,1/xa.sum(axis=0).A.flatten())
        # get measurement given annotation

        # assume independent m and a
        # ma = utils.gemmT(xm.T, xa.T)

        # get probability of Annotation given mixture
        ax = mixtures.T
        utils.col_scale(ax,1/ax.sum(axis=0).A.flatten())
        # assume minimal error between m and a
        xam = get_minimal_transitions(ax.T, xm)
        ax.eliminate_zeros()
        ax.data = 1 / ax.data
        ma = np.einsum('xam,xa,ax->ma', xam, xa.A, ax.A)
        
        cell_type = utils.parallel_nnls(ma, om)
        
        cell_type = pd.DataFrame(cell_type, columns=types, index=adata.obs.index)
    
    cell_type = helper.normalize_result_format(cell_type)
    # cell_type.to_csv('/bmbl_data/jiangyi/SpaGFT/TACCO_Update/test_data/cell_type.csv')

    # cell_type_read_csv = pd.read_csv("/bmbl_data/jiangyi/SpaGFT/TACCO_Update/test_data/cell_type.csv",index_col=0)
    cell_type_read = sc.AnnData(cell_type_read_csv)
    cell_type_read.obs = adata.obs
    adata_copy = adata.copy()
    adata_copy.var_names_make_unique()
    adata_copy.raw = adata_copy
    adata_copy.obs = adata.obs
    # QC
    sc.pp.filter_genes(adata_copy, min_cells=10)
    # Normalization
    sc.pp.normalize_total(adata_copy, inplace=True)
    sc.pp.log1p(adata_copy)
    (ratio_low_adata, ratio_high_adata) = spg.gft.determine_frequency_ratio(adata_copy, ratio_neighbors=1, spatial_info=adata_copy.obs_keys())
    fm_domain_adata = spg.calculate_frequency_domain(adata_copy,ratio_neighbors=1, spatial_info=adata_copy.obs_keys(),ratio_low_freq=ratio_low_adata,ratio_high_freq=ratio_high_adata)
    fm_domain_cell_type_read = spg.calculate_frequency_domain(cell_type_read,ratio_neighbors=1,spatial_info=adata_copy.obs_keys(),ratio_low_freq=ratio_low_adata,ratio_high_freq=ratio_high_adata)
    gene_df_adata = spg.detect_svg(adata_copy,
                            spatial_info=adata_copy.obs_keys(),
                            ratio_low_freq=ratio_low_adata,
                            ratio_high_freq=ratio_high_adata,
                            ratio_neighbors=1,
                            filter_peaks=True,
                            S=kw_args['S_spagft'])

    svp_list_adata = gene_df_adata[gene_df_adata.cutoff_gft_score][gene_df_adata.fdr < 0.05].index.tolist()

    t_m = adata_copy[:,svp_list_adata].X.dot(fm_domain_adata[svp_list_adata].T)

    from sklearn.metrics.pairwise import cosine_distances
    distance = np.ones((len(fm_domain_cell_type_read.T.values), len(t_m)))
    for v_ct in range(len(fm_domain_cell_type_read.T.values)):
        for v_m in range(len(t_m)):
            vector1 = fm_domain_cell_type_read.T.values[v_ct].reshape(1, -1)
            vector2 = t_m[v_m].reshape(1, -1)
            distance[v_ct][v_m] = abs(cosine_distances(vector1, vector2))
    distance_FM = pd.DataFrame(distance,index=fm_domain_cell_type_read.columns,columns=adata.obs.index)
    
    cell_type = _annotate_OTGFT_supplement(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
        annotation_prior=annotation_prior,
        epsilon=epsilon,
        lamb=lamb,
        decomposition=decomposition,
        deconvolution=deconvolution,
        distance_FM = distance_FM,
        alph_cell_type = kw_args['lamada_mtb'],
        # alph_cell_type = 0.5,
        **kw_args,)
    return cell_type



def _annotate_OTGFT_supplement(
    adata,
    reference,
    annotation_key=None,
    annotation_prior=None,
    epsilon=5e-3,
    lamb=0.1,
    decomposition=False,
    deconvolution=False,
    distance_FM = 0,
    alph_cell_type = 0.5,
    **kw_args,
    ):

    """\
    Implements the functionality of :func:`~annotate_OT` without data
    integrity checks.
    """

    cell_prior = helper.prep_cell_priors(adata, reads=True)

    type_cell_dist = helper.prep_distance(adata, reference, annotation_key, decomposition=decomposition, deconvolution=deconvolution, **kw_args)

    if decomposition:
        type_cell_dist, mixtures = type_cell_dist
    types = type_cell_dist.index
    
    if decomposition: # include the annotation profiles themselves to obtain a measurement of confusion
        test_weight = 1e-6*cell_prior.sum()/(len(type_cell_dist.columns)-len(cell_prior.index)) # keep a low weight to not influence the actual typing (much)
        cell_prior = cell_prior.reindex(index=type_cell_dist.columns,fill_value=test_weight)
    type_cell_dist = alph_cell_type*type_cell_dist+ (1-alph_cell_type)*distance_FM
    cell_type = _run_OT(type_cell_dist, annotation_prior, cell_prior=cell_prior, epsilon=epsilon, lamb=lamb)
    
    if decomposition:
        cell_type /= cell_type.sum(axis=1).to_numpy()[:,None]
        # need to run measurement on mixtures instead of pure profiles to get the confusion as pure profiles should never be confused at all.
        # miXture Measurements
        xm = cell_type.loc[~cell_type.index.isin(adata.obs.index)].to_numpy()
        
        # Observation Measurements
        om = cell_type.loc[adata.obs.index].to_numpy()
        # miXture Annotation joint probability distribution
        xa = mixtures.copy()
        # get probability of miXture given Annotation
        utils.col_scale(xa,1/xa.sum(axis=0).A.flatten())
        # get measurement given annotation

        # assume independent m and a
        # ma = utils.gemmT(xm.T, xa.T)

        # get probability of Annotation given mixture
        ax = mixtures.T
        utils.col_scale(ax,1/ax.sum(axis=0).A.flatten())
        # assume minimal error between m and a
        xam = get_minimal_transitions(ax.T, xm)
        ax.eliminate_zeros()
        ax.data = 1 / ax.data
        ma = np.einsum('xam,xa,ax->ma', xam, xa.A, ax.A)
        
        cell_type = utils.parallel_nnls(ma, om)
        
        cell_type = pd.DataFrame(cell_type, columns=types, index=adata.obs.index)
    
    cell_type = helper.normalize_result_format(cell_type)
    
    return cell_type

def annotate_OTGFT(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    annotation_prior=None,
    epsilon=5e-3,
    lamb=0.1,
    decomposition=False,
    deconvolution=False,
    **kw_args,
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by
    semi-balanced optimal transport.

    This is the direct interface to this annotation method. In practice using
    the general wrapper :func:`~tacco.tools.annotate` is recommended due to its
    higher flexibility.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X`.
    reference
        Reference data to get the annotation definition from.
    annotation_key
        The `.obs` and/or `.varm` key where the annotation and/or profiles are
        stored in the `reference`. If `None`, it is inferred from `reference`,
        if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    annotation_prior
        A :class:`~pandas.Series` containing the annotation prior distribution.
        This argument is required and will throw a :class:`~ValueError` if it
        is unset.
    epsilon
        The prefactor for entropy regularization in Optimal Transport. Small
        values like `5e-3` tend to give only a few or a single annotation
        category per observation, while large values like `5e-1` give many
        annotation categories per observation.
    lamb
        The prefactor for prior constraint relaxation by KL divergence in
        unbalanced Optimal Transport. Smaller values like `1e-2` relax the
        constraint more than larger ones like `1e0`. If `None`, do not relax
        the prior constraint and fix the annotation fraction at
        `annotation_prior`.
    decomposition
        Whether to decompose the annotation using information from injected
        in-silico type mixtures.
    deconvolution
        Which method to use for deconvolution of the cost based on similarity of
        different annotation profiles. If `False`, no deconvolution is done.
        Available methods are:
        
        - 'nnls': solves nnls to get only non-negative deconvolved projections
        - 'linear': solves a linear system to disentangle contributions; can
          result in negative values which makes sense for general vectors and
          amplitudes, i.e.
          
    **kw_args
        Additional keyword arguments are forwarded to
        :func:`~tacco.tools._helper.prep_distance`.
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """

    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location)

    # call typing without data integrity checks
    cell_type = _annotate_OTGFT(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
        annotation_prior=annotation_prior,
        epsilon=epsilon,
        lamb=lamb,
        decomposition=decomposition,
        deconvolution=deconvolution,
        **kw_args,
    )
    
    return cell_type

