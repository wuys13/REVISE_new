import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

from .morphology import calculate_morphology_probability, update_morphology_parameter, cal_prob_bayes, cal_prob_kmeans
from .sc_ref import construct_pure_reference


def run_vi_model(adata_st, sc_ref, 
                cells_on_spot, morphology_features, feature_list=None, lambda_morph=0.0,
                batch_prior=2,  adam_params=None, device='cuda', n_epoch=8000,  plot_file_name = None,):
    """
    Pyro 模型：结合基因表达和形态学信息进行变分推断
    Args:
        X: ST 数据，(n_spots, n_genes)
        sc_ref: Cell type 参考，(n_genes, n_types)
        cells_on_spot: 包含spot_name和all_cells_in_spot的字典
        morphology_features: 形态学特征
        feature_list: 特征列表
        batch_prior: 批次效应先验
        lambda_morph: 形态学正则化权重
        n_epoch: 训练轮数
    Returns:
        params: 变分参数
    """
    if adam_params is None:
        adam_params = {"lr": 0.003, "betas": (0.95, 0.999)}
    X = adata_st.X if type(adata_st.X) is np.ndarray else adata_st.X.toarray()
    max_exp = int(np.max(np.sum(X, axis=1)))
    X = torch.tensor(X, device=device, dtype=torch.int32)
    
    sc_ref = sc_ref.values
    sc_ref = torch.tensor(sc_ref, device=device, dtype=torch.float32)
    batch_prior = torch.tensor(batch_prior, device=device, dtype=torch.float64)
    assert X.shape[1] == sc_ref.shape[1], "Spatial data and SC reference data must have the same number of genes."

    n_spot = len(X)
    n_type, n_gene = sc_ref.shape
    if morphology_features is not None:
        n_feature = len(feature_list)
        
        mu_morph_init = np.random.rand(n_type, n_feature)
        sigma_morph_init = np.ones((n_type, n_feature))

    def model(sc_ref, X, batch_prior): 
        if batch_prior > 0:
            alpha = pyro.sample("Batch effect", dist.Uniform(torch.tensor(0., device=device),
                                                            torch.tensor(1., device=device)).expand([n_gene]).to_event(1))
            alpha_exp = torch.exp2(alpha * batch_prior)
        else:
            alpha_exp = torch.ones(n_gene, device=device, dtype=torch.float64)
        
        with pyro.plate("spot", n_spot, dim = -1):
            q = pyro.sample("contributions", dist.Dirichlet(torch.full((n_type,), 3., device=device)))
            rho = torch.matmul(q, sc_ref) * alpha_exp
            rho = rho / rho.sum(dim=-1).unsqueeze(-1)
            if torch.any(torch.isnan(rho)):
                print('rho contains NaN at')
                print(torch.where(torch.isnan(rho)))
            pyro.sample("Spatial RNA", dist.Multinomial(total_count=max_exp, probs=rho), obs=X)

    def guide(sc_ref, X, batch_prior):
        # n_spot = len(X)
        # n_type, n_gene = sc_ref.shape
        if batch_prior > 0:
            alpha_loc = pyro.param("alpha_loc", torch.zeros(n_gene, device=device))
            alpha_scale = pyro.param("alpha_scale", torch.ones(n_gene, device=device),
                                    constraint=constraints.positive)
            alpha_dist = dist.TransformedDistribution(dist.Normal(alpha_loc, alpha_scale),
                                                    [dist.transforms.SigmoidTransform()])
            alpha = pyro.sample("Batch effect", alpha_dist.to_event(1))

        with pyro.plate("spot", n_spot):
            sigma = pyro.param('sigma', lambda: torch.full((n_spot, n_type), 3., device=device),
                               constraint=constraints.positive)
            q = pyro.sample("contributions", dist.Dirichlet(sigma))

        if morphology_features is not None and lambda_morph > 0:
            # morphology features
            mu_morph = pyro.param("mu_morph", torch.tensor(mu_morph_init, device=device))
            sigma_morph = pyro.param("sigma_morph", torch.tensor(sigma_morph_init, device=device))
            
            # E 步：计算 PM_on_cell 和 PME_on_cell
            PM_on_cell = calculate_morphology_probability(morphology_features, feature_list,
                                                        mu_morph.cpu().detach().numpy(),
                                                        sigma_morph.cpu().detach().numpy())
            PE_on_spot = sigma / sigma.sum(dim=-1, keepdim=True)  # E[q]

            # M 步：更新 mu 和 sigma
            mu_new, sigma_new = update_morphology_parameter(PE_on_spot.cpu().detach().numpy(),
                                                            PM_on_cell, cells_on_spot, 
                                                            morphology_features, feature_list)
            pyro.param("mu_morph").data = torch.tensor(mu_new, device=device, dtype=torch.float32)
            pyro.param("sigma_morph").data = torch.tensor(sigma_new, device=device, dtype=torch.float32)


            # 形态学正则化：基于 PM_on_spot
            PM_on_spot = np.zeros((n_spot, n_type))
            spot_ids = cells_on_spot['spot_name'].astype(str).values
            unique_spots = np.unique(spot_ids)
            for s, spot in enumerate(unique_spots):
                mask = spot_ids == spot
                PM_on_spot[s, :] = PM_on_cell[mask, :].mean(axis=0)
            PM_on_spot = torch.tensor(PM_on_spot, device=device, dtype=torch.float32)

            if lambda_morph > 0:
                q_mean = sigma / sigma.sum(dim=-1, keepdim=True)
                morph_loss = (q_mean - PM_on_spot).pow(2).sum()
                pyro.factor("morphology_regularization", -lambda_morph * morph_loss)

    pyro.clear_param_store()
    optimizer = Adam(adam_params)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    loss_history = []
    for _ in tqdm(range(n_epoch), desc="Training"):
        loss = svi.step(sc_ref, X, batch_prior)
        loss_history.append(loss)
    if plot_file_name is not None:
        plt.plot(loss_history)
        plt.title("ELBO Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(plot_file_name, dpi=300)
        plt.show()

    # params = pyro.get_param_store()
    # mu = params['mu'].cpu().detach().numpy()
    # sigma = params['sigma'].cpu().detach().numpy()
    # return params, mu, sigma
    return pyro.get_param_store()

def get_cell_contributions(adata_st, adata_sc, sc_ref, key_type, 
                          cells_on_spot, morphology_features, feature_list, lambda_morph=0.0,
                          device='cuda', n_epoch=4000, adam_params=None,
                          batch_prior=2, plot_file_name = None,
                          ):
    type_list=sc_ref.index.tolist()

    if feature_list is None:
        if morphology_features is None:
            feature_list = None
        else:
            feature_list = morphology_features.columns.tolist()
            feature_list = [feature for feature in feature_list if feature not in ['x', 'y']]
    
    pyro_params = run_vi_model(adata_st, sc_ref, 
                              cells_on_spot, morphology_features, feature_list, lambda_morph=lambda_morph, 
                              device=device, n_epoch=n_epoch, adam_params=adam_params,
                              batch_prior=batch_prior, plot_file_name=plot_file_name)
    sigma = pyro_params['sigma'].cpu().detach().numpy()
    
    expr = np.array(adata_sc.X.toarray())
    mean_exp = np.array([np.mean(np.sum(expr[adata_sc.obs[key_type]==type_list[i]], axis=1))
                         for i in range(len(type_list))])
    
    cell_contributions = sigma/mean_exp
    cell_contributions = cell_contributions/np.sum(cell_contributions, axis=1)[:, np.newaxis]
    cell_contributions = pd.DataFrame(cell_contributions, columns=type_list, index=adata_st.obs_names)
    
    # mu_morph, sigma_morph = update_morphology_parameter(cell_contributions,
    #                                                         PM_on_cell, cells_on_spot, 
    #                                                         morphology_features, feature_list)
    # PM_on_cell = calculate_morphology_probability(morphology_features, feature_list,
    #                                                 mu_morph, sigma_morph)

    # print("cell_contributions: ", cell_contributions)



    # # Compute uncertainty (standard deviation) via posterior sampling
    # n_samples = 1000
    # sigma_torch = pyro.get_param_store()['sigma']
    # q_dist = torch.distributions.Dirichlet(sigma_torch)
    # q_samples = q_dist.sample((n_samples,))  # Shape: (n_samples, n_spot, n_type)
    # q_std = q_samples.std(dim=0).cpu().detach().numpy()
    # q_std_df = pd.DataFrame(q_std, columns=type_list, index=adata_st.obs_names)
    # adata_st.obsm[f'{key_type}_uncertainty_std'] = q_std_df

    # # Compute uncertainty (standard deviation) analytically
    # sigma_np = sigma
    # sigma_0 = np.sum(sigma_np, axis=1, keepdims=True)  # Shape: (n_spot, 1)
    # variance = sigma_np * (sigma_0 - sigma_np) / (sigma_0**2 * (sigma_0 + 1))
    # q_std_analytical = np.sqrt(variance)
    # q_std_analytical_df = pd.DataFrame(q_std_analytical, columns=type_list, index=adata_st.obs_names)
    # # adata_st.obsm[f'{key_type}_uncertainty_std_analytical'] = q_std_analytical_df

    # # Compute uncertainty (entropy) analytically
    # from scipy.special import psi, loggamma
    # entropy = np.zeros(sigma_np.shape[0])
    # for i in range(sigma_np.shape[0]):
    #     sigma_i = sigma_np[i]
    #     entropy[i] = (loggamma(sigma_0[i]) - np.sum(loggamma(sigma_i)) +
    #                   (sigma_0[i] - len(sigma_i)) * psi(sigma_0[i]) -
    #                   np.sum((sigma_i - 1) * psi(sigma_i)))
    # # adata_st.obs[f'{key_type}_entropy'] = entropy
    # entropy = pd.DataFrame(entropy, columns=[f'{key_type}_entropy'], index=adata_st.obs_names)


    return cell_contributions #, q_std_analytical_df, entropy


import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_cell_contributions_sp(adata_sp, adata_sc, ct_name, batch_prior=2, adam_params=None, device='cuda', n_epoch=4000, plot_file_name=None, confidence_ts=0.9, min_num=10):
    """
    Perform cell type annotation using variational inference with a unified reference.
    
    Args:
        adata_sp: Spatial transcriptomics AnnData object
        adata_sc: Single-cell reference AnnData object
        ct_name: Name of the cell type annotation to store
        batch_prior: Prior for batch effect
        adam_params: Parameters for Adam optimizer
        device: Device to run PyTorch ('cuda' or 'cpu')
        n_epoch: Number of training epochs
        plot_file_name: File name for saving loss plot
        confidence_ts: Confidence threshold for selecting pure spots
        min_num: Minimum number of spots per cell type for reference
    
    Returns:
        cell_contributions: DataFrame with cell type proportions
    """
    if adam_params is None:
        adam_params = {"lr": 0.003, "betas": (0.95, 0.999)}
    
    # Prepare data on CPU
    X = adata_sp.X if isinstance(adata_sp.X, np.ndarray) else adata_sp.X.toarray()
    max_exp = int(np.max(np.sum(X, axis=1)))
    
    # Get cell types and spot metadata
    type_list = sorted(adata_sc.obs[ct_name].unique().tolist())
    type_to_idx = {t: idx for idx, t in enumerate(type_list)}
    spot_cell_types = adata_sp.obs.get('Max_type', [None] * len(adata_sp)).values
    spot_confidences = adata_sp.obs.get('Confidence', [1.0] * len(adata_sp)).values
    
    n_spot, n_gene = X.shape
    n_type = len(type_list)
    
    # Filter pure spots and build unified reference on CPU
    sc_ref = np.zeros((n_type, n_gene), dtype=np.float32)
    for cell_type in type_list:
        type_idx = type_to_idx[cell_type]
        # Filter spots by confidence threshold
        valid_mask = (spot_cell_types == cell_type) & (spot_confidences >= confidence_ts)
        valid_indices = np.where(valid_mask)[0]
        num_valid = len(valid_indices)
        
        print(f"Cell type '{cell_type}': {num_valid} pure spots found (confidence >= {confidence_ts})")
        
        # If fewer than min_num spots, take top min_num by confidence
        if num_valid < min_num:
            type_mask = spot_cell_types == cell_type
            type_indices = np.where(type_mask)[0]
            type_confs = spot_confidences[type_mask]
            select_num = min(min_num, len(type_indices))
            if len(type_indices) > 0:
                top_indices = type_indices[np.argsort(type_confs)[-(select_num):]]
                print(f"  Cell type '{cell_type}': Increased from {num_valid} to {select_num} spots using top confidence")
                selected_indices = top_indices
            else:
                print(f"  Warning: Cell type '{cell_type}' has no spots; using zero reference")
                selected_indices = []
        else:
            selected_indices = valid_indices
        
        # Compute mean expression for selected spots
        if len(selected_indices) > 0:
            type_X = adata_sp.X[selected_indices] if isinstance(adata_sp.X, np.ndarray) else adata_sp.X[selected_indices].toarray()
            mean_expr = np.mean(type_X, axis=0)
            mean_expr = mean_expr / (np.sum(mean_expr) + 1e-10)  # Normalize
            sc_ref[type_idx, :] = mean_expr
    
    # Move data to device
    X_tensor = torch.tensor(X, device=device, dtype=torch.int32)
    sc_ref_tensor = torch.tensor(sc_ref, device=device, dtype=torch.float32)
    batch_prior_tensor = torch.tensor(batch_prior, device=device, dtype=torch.float64)
    
    # Initialize output array
    cell_contributions = np.zeros((n_spot, n_type))
    
    # Pyro model
    def model(sc_ref, X, batch_prior):
        if batch_prior > 0:
            alpha = pyro.sample("Batch effect", dist.Uniform(torch.tensor(0., device=device),
                                                            torch.tensor(1., device=device)).expand([n_gene]).to_event(1))
            alpha_exp = torch.exp2(alpha * batch_prior)
        else:
            alpha_exp = torch.ones(n_gene, device=device, dtype=torch.float64)
        
        with pyro.plate("spot", X.shape[0], dim=-1):
            q = pyro.sample("contributions", dist.Dirichlet(torch.full((n_type,), 3., device=device)))
            rho = torch.matmul(q, sc_ref)  # Simplified: sc_ref is (n_type, n_gene)
            rho = rho * alpha_exp
            rho = rho / rho.sum(dim=-1).unsqueeze(-1)
            if torch.any(torch.isnan(rho)):
                print('rho contains NaN at')
                print(torch.where(torch.isnan(rho)))
            pyro.sample("Spatial RNA", dist.Multinomial(total_count=max_exp, probs=rho), obs=X)
    
    # Pyro guide
    def guide(sc_ref, X, batch_prior):
        if batch_prior > 0:
            alpha_loc = pyro.param("alpha_loc", torch.zeros(n_gene, device=device))
            alpha_scale = pyro.param("alpha_scale", torch.ones(n_gene, device=device),
                                    constraint=pyro.distributions.constraints.positive)
            alpha_dist = dist.TransformedDistribution(dist.Normal(alpha_loc, alpha_scale),
                                                    [dist.transforms.SigmoidTransform()])
            alpha = pyro.sample("Batch effect", alpha_dist.to_event(1))
        
        with pyro.plate("spot", X.shape[0]):
            sigma = pyro.param('sigma', lambda: torch.full((X.shape[0], n_type), 3., device=device),
                               constraint=pyro.distributions.constraints.positive)
            q = pyro.sample("contributions", dist.Dirichlet(sigma))
    
    # Run SVI on entire dataset
    pyro.clear_param_store()
    optimizer = Adam(adam_params)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    loss_history = []
    
    for _ in tqdm(range(n_epoch), desc="Training SVI"):
        loss = svi.step(sc_ref_tensor, X_tensor, batch_prior_tensor)
        loss_history.append(loss)
    
    # Store contributions
    sigma = pyro.get_param_store()['sigma'].cpu().detach().numpy()
    cell_contributions = sigma
    
    if plot_file_name is not None:
        plt.plot(loss_history)
        plt.title("ELBO Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(plot_file_name, dpi=300)
        plt.close()
    
    # Normalize contributions
    expr = adata_sc.X.toarray() if hasattr(adata_sc.X, 'toarray') else adata_sc.X
    mean_exp = np.array([np.mean(np.sum(expr[adata_sc.obs[ct_name] == t], axis=1)) for t in type_list])
    cell_contributions = cell_contributions / mean_exp
    cell_contributions = cell_contributions / np.sum(cell_contributions, axis=1)[:, np.newaxis]
    cell_contributions = pd.DataFrame(cell_contributions, columns=type_list, index=adata_sp.obs_names)
    
    return cell_contributions

# import numpy as np
# import pandas as pd
# import torch
# import pyro
# import pyro.distributions as dist
# from pyro.infer import SVI, Trace_ELBO
# from pyro.optim import Adam
# from scipy.spatial import KDTree
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# def get_cell_contributions_sp(adata_sp, adata_sc, ct_name, batch_prior=2, adam_params=None, device='cuda', n_epoch=4000, plot_file_name=None, n_search_neighbors=5, n_choose_neighbors=3):
#     """
#     Perform cell type annotation using variational inference with spot-specific references.
    
#     Args:
#         adata_sp: Spatial transcriptomics AnnData object
#         adata_sc: Single-cell reference AnnData object
#         ct_name: Name of the cell type annotation to store
#         batch_prior: Prior for batch effect
#         adam_params: Parameters for Adam optimizer
#         device: Device to run PyTorch ('cuda' or 'cpu')
#         n_epoch: Number of training epochs
#         plot_file_name: File name for saving loss plot
#         n_search_neighbors: Number of nearest spots to search
#         n_choose_neighbors: Number of high-confidence neighbors to select
    
#     Returns:
#         cell_contributions: DataFrame with cell type proportions
#     """
#     if adam_params is None:
#         adam_params = {"lr": 0.003, "betas": (0.95, 0.999)}
    
#     # Prepare data on CPU
#     X = adata_sp.X if isinstance(adata_sp.X, np.ndarray) else adata_sp.X.toarray()
#     max_exp = int(np.max(np.sum(X, axis=1)))
    
#     # Get cell types and spot coordinates
#     type_list = sorted(adata_sc.obs[ct_name].unique().tolist())
#     type_to_idx = {t: idx for idx, t in enumerate(type_list)}
#     spot_coords = adata_sp.obsm['spatial']
#     spot_cell_types = adata_sp.obs.get('Max_type', [None] * len(adata_sp)).values
#     spot_confidences = adata_sp.obs.get('Confidence', [1.0] * len(adata_sp)).values
    
#     n_spot, n_gene = X.shape
#     n_type = len(type_list)
    
#     # Build KDTree for neighbor search on CPU
#     kdtree = KDTree(spot_coords)
    
#     # Precompute spot-specific references on CPU
#     spot_specific_refs = np.zeros((n_spot, n_type, n_gene), dtype=np.float32)
#     for spot_idx in tqdm(range(n_spot), desc="Building spot-specific references"):
#         spot_name = adata_sp.obs_names[spot_idx]
#         spot_coord = spot_coords[spot_idx]
        
#         # Query nearest neighbors
#         _, neighbor_indices = kdtree.query(spot_coord, k=n_search_neighbors)
        
#         for cell_type in type_list:
#             type_idx = type_to_idx[cell_type]
#             # Filter neighbors where Max_type matches cell_type
#             neighbor_types = spot_cell_types[neighbor_indices]
#             neighbor_confs = spot_confidences[neighbor_indices]
#             valid_mask = neighbor_types == cell_type
#             valid_indices = neighbor_indices[valid_mask]
#             valid_confs = neighbor_confs[valid_mask]
            
#             # Select top neighbors by confidence
#             if len(valid_confs) > 0:
#                 top_indices = valid_indices[np.argsort(valid_confs)[-n_choose_neighbors:]]
#                 type_X = adata_sp.X[top_indices] if isinstance(adata_sp.X, np.ndarray) else adata_sp.X[top_indices].toarray()
#                 mean_expr = np.mean(type_X, axis=0)
#                 mean_expr = mean_expr / (np.sum(mean_expr) + 1e-10)  # Normalize
#                 spot_specific_refs[spot_idx, type_idx, :] = mean_expr
    
#     # Move data to device (GPU or CPU)
#     X_tensor = torch.tensor(X, device=device, dtype=torch.int32)
#     sc_ref = torch.tensor(spot_specific_refs, device=device, dtype=torch.float32)
#     print("sc_ref shape: ", sc_ref.shape)
#     batch_prior_tensor = torch.tensor(batch_prior, device=device, dtype=torch.float64)
    
#     # Initialize output array
#     cell_contributions = np.zeros((n_spot, n_type))
    
#     # Pyro model
#     def model(sc_ref, X, batch_prior):
#         if batch_prior > 0:
#             alpha = pyro.sample("Batch effect", dist.Uniform(torch.tensor(0., device=device),
#                                                             torch.tensor(1., device=device)).expand([n_gene]).to_event(1))
#             alpha_exp = torch.exp2(alpha * batch_prior)
#         else:
#             alpha_exp = torch.ones(n_gene, device=device, dtype=torch.float64)
        
#         with pyro.plate("spot", X.shape[0], dim=-1):
#             q = pyro.sample("contributions", dist.Dirichlet(torch.full((n_type,), 3., device=device)))
#             rho = torch.matmul(q.unsqueeze(-2), sc_ref).squeeze(-2)
#             rho = rho * alpha_exp
#             rho = rho / rho.sum(dim=-1).unsqueeze(-1)
#             if torch.any(torch.isnan(rho)):
#                 print('rho contains NaN at')
#                 print(torch.where(torch.isnan(rho)))
#             pyro.sample("Spatial RNA", dist.Multinomial(total_count=max_exp, probs=rho), obs=X)
    
#     # Pyro guide
#     def guide(sc_ref, X, batch_prior):
#         if batch_prior > 0:
#             alpha_loc = pyro.param("alpha_loc", torch.zeros(n_gene, device=device))
#             alpha_scale = pyro.param("alpha_scale", torch.ones(n_gene, device=device),
#                                     constraint=pyro.distributions.constraints.positive)
#             alpha_dist = dist.TransformedDistribution(dist.Normal(alpha_loc, alpha_scale),
#                                                     [dist.transforms.SigmoidTransform()])
#             alpha = pyro.sample("Batch effect", alpha_dist.to_event(1))
        
#         with pyro.plate("spot", X.shape[0]):
#             sigma = pyro.param('sigma', lambda: torch.full((X.shape[0], n_type), 3., device=device),
#                                constraint=pyro.distributions.constraints.positive)
#             q = pyro.sample("contributions", dist.Dirichlet(sigma))
    
#     # Run SVI on entire dataset
#     pyro.clear_param_store()
#     optimizer = Adam(adam_params)
#     svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
#     loss_history = []
    
#     for _ in tqdm(range(n_epoch), desc="Training SVI"):
#         loss = svi.step(sc_ref, X_tensor, batch_prior_tensor)
#         loss_history.append(loss)
    
#     # Store contributions
#     sigma = pyro.get_param_store()['sigma'].cpu().detach().numpy()
#     cell_contributions = sigma
    
#     if plot_file_name is not None:
#         plt.plot(loss_history)
#         plt.title("ELBO Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.savefig(plot_file_name, dpi=300)
#         plt.close()
    
#     # Normalize contributions
#     expr = adata_sc.X.toarray() if hasattr(adata_sc.X, 'toarray') else adata_sc.X
#     mean_exp = np.array([np.mean(np.sum(expr[adata_sc.obs[ct_name] == t], axis=1)) for t in type_list])
#     cell_contributions = cell_contributions / mean_exp
#     cell_contributions = cell_contributions / np.sum(cell_contributions, axis=1)[:, np.newaxis]
#     cell_contributions = pd.DataFrame(cell_contributions, columns=type_list, index=adata_sp.obs_names)
    
#     return cell_contributions



def get_spot_cell_distribution(cell_contributions, SVC_obs, cell_completeness=True):
    """
    优化每个spot的cell type比例，使其符合实际细胞数
    Args:
        cell_contributions: array of cell type contributions for each spot (shape: [n_spots, n_cell_types])
        SVC_obs: DataFrame containing 'spot_name' and 'cell_id' columns
        cell_completeness: whether to ensure cell counts are integers
    Returns:
        optimized_contributions: array of optimized cell type contributions (shape: [n_spots, n_cell_types])
    """
    # 获取每个spot的实际细胞数
    spot_cell_counts = SVC_obs.groupby('spot_name')['cell_id'].count().to_dict()
    spot_names = cell_contributions.index
    
    spot_cell_distribution = []
    
    for spot_idx, spot_name in enumerate(spot_names):
        # 获取当前spot的实际细胞总数
        total_cells = spot_cell_counts[spot_name]
        
        # 当前spot的原始贡献比例
        original_contrib = cell_contributions.loc[spot_name].values
        
        # 计算基于当前比例的细胞数
        cell_counts = original_contrib * total_cells
        
        if cell_completeness:
            # 应用舍入以确保整数细胞计数（使用numpy.round保持总和接近）
            cell_counts = np.round(cell_counts)
            
            # 调整舍入误差以确保总和等于实际细胞数
            current_sum = np.sum(cell_counts)
            if current_sum != total_cells:
                # print(f"Adjusting cell counts for {spot_name}: {current_sum} -> {total_cells}")
                # 确定需要调整的差值
                adjustment = total_cells - current_sum
                if adjustment > 0:
                    # 找出最大概率的cell类型进行调整
                    max_idx = np.argmax(cell_counts)
                    cell_counts[max_idx] += adjustment
                elif adjustment < 0:
                    print(spot_name, adjustment, total_cells, current_sum, cell_counts)
                    # 找出非零元素的索引并按值从小到大排序
                    non_zero_indices = np.where(cell_counts >= 1)[0]
                    sorted_indices = non_zero_indices[np.argsort(cell_counts[non_zero_indices])]
                    
                    # 从最小的非零元素开始减1，直至adjustment消耗完
                    for idx in sorted_indices:
                        if adjustment >= 0:
                            break
                        cell_counts[idx] -= 1
                        adjustment += 1  # 因为减1会减少当前_sum的值，所以需要调整adjustment
                    print(spot_name, adjustment, total_cells, current_sum, cell_counts)
            
            spot_cell_distribution.append(cell_counts)

    spot_cell_distribution = pd.DataFrame(np.array(spot_cell_distribution), index=spot_names,
                                           columns=cell_contributions.columns)
        
    return spot_cell_distribution
        



def optimize_cell_contributions(cell_contributions, SVC_obs, cell_completeness=True):
    """
    优化每个spot的cell type比例，使其符合实际细胞数
    Args:
        cell_contributions: array of cell type contributions for each spot (shape: [n_spots, n_cell_types])
        SVC_obs: DataFrame containing 'spot_name' and 'cell_id' columns
        cell_completeness: whether to ensure cell counts are integers
    Returns:
        optimized_contributions: array of optimized cell type contributions (shape: [n_spots, n_cell_types])
    """
    # 获取每个spot的实际细胞数
    spot_cell_counts = SVC_obs.groupby('spot_name')['cell_id'].count().to_dict()
    
    # 获取spot名称有序列表，确保与cell_contributions的顺序一致
    spot_names = cell_contributions.index
    
    optimized_contributions = []
    
    for spot_idx, spot_name in enumerate(spot_names):
        # 获取当前spot的实际细胞总数
        total_cells = spot_cell_counts[spot_name]
        
        # 当前spot的原始贡献比例
        original_contrib = cell_contributions.loc[spot_name].values
        
        # 计算基于当前比例的细胞数
        cell_counts = original_contrib * total_cells
        
        if cell_completeness:
            # 应用舍入以确保整数细胞计数（使用numpy.round保持总和接近）
            cell_counts = np.round(cell_counts)
            
            # 调整舍入误差以确保总和等于实际细胞数
            current_sum = np.sum(cell_counts)
            if current_sum != total_cells:
                # print(f"Adjusting cell counts for {spot_name}: {current_sum} -> {total_cells}")
                # 确定需要调整的差值
                adjustment = total_cells - current_sum
                if adjustment > 0:
                    # 找出最大概率的cell类型进行调整
                    max_idx = np.argmax(cell_counts)
                    cell_counts[max_idx] += adjustment
                elif adjustment < 0:
                    print(spot_name, adjustment, total_cells, current_sum, cell_counts)
                    # 找出非零元素的索引并按值从小到大排序
                    non_zero_indices = np.where(cell_counts >= 1)[0]
                    sorted_indices = non_zero_indices[np.argsort(cell_counts[non_zero_indices])]
                    
                    # 从最小的非零元素开始减1，直至adjustment消耗完
                    for idx in sorted_indices:
                        if adjustment >= 0:
                            break
                        cell_counts[idx] -= 1
                        adjustment += 1  # 因为减1会减少当前_sum的值，所以需要调整adjustment
                    print(spot_name, adjustment, total_cells, current_sum, cell_counts)
            
        cell_counts_mask = cell_counts >= 0
        optimized_contrib = original_contrib[cell_counts_mask]
        optimized_contrib = optimized_contrib / np.sum(optimized_contrib)
              
        # # 归一化到新总和（如果启用了完整性检查）
        # if cell_completeness:
        #     optimized_contrib = cell_counts / total_cells
        # else:
        #     # 如果不需要完整性，保持原始比例但基于实际细胞数重新缩放
        #     optimized_contrib = cell_counts / np.sum(cell_counts)
        
        optimized_contributions.append(optimized_contrib)
    
    optimized_contributions = np.array(optimized_contributions)
    optimized_contributions = pd.DataFrame(optimized_contributions, index=spot_names,
                                           columns=cell_contributions.columns)
    
    return optimized_contributions


def assign_cell_types(SVC_obs, PM_on_cell, spot_cell_distribution):
    """    
    Args:
        SVC_obs: DataFrame，包含'spot_name'和'cell_id'列
        PM_on_cell: DataFrame，每个cell的cell类型概率
        spot_cell_distribution: DataFrame，每个spot中各个cell type的细胞数
        
    Returns:
        SVC_obs: 更新后的SVC_obs，包含'cell_type'列
    """
    SVC_obs = SVC_obs.copy()
    type_list = list(PM_on_cell.columns)  # 获取细胞类型列表
    
    # 添加新的列来存储cell types
    if 'cell_type' not in SVC_obs.columns:
        SVC_obs['cell_type'] = "Unknown"
    
    # 按spot分组处理
    spot_groups = SVC_obs.groupby('spot_name')
    
    # 为每个spot分配细胞类型
    for spot_name in tqdm(spot_cell_distribution.index, desc="Assigning cell types"):
        # 获取当前spot中的所有cell
        spot_cells_df = spot_groups.get_group(spot_name)
        
        # 确保cell_id作为索引存在于PM_on_cell中
        valid_cells = spot_cells_df[spot_cells_df['cell_id'].isin(PM_on_cell.index)]
        if len(valid_cells) == 0:
            print(f"Warning: No valid cells found for spot {spot_name}")
            # print(spot_cells_df)
            # print(PM_on_cell.head())
            # exit()
            continue
            
        # 获取每个cell对应的cell类型概率
        cell_probs = PM_on_cell.loc[valid_cells['cell_id']].values
        
        # 获取该spot需要的每种类型的细胞数量
        target_counts = spot_cell_distribution.loc[spot_name].astype(int)  # 确保是整数
        
        # 初始分配：每个cell选择最高概率的type
        cell_type_indices = np.argmax(cell_probs, axis=1)
        initial_types = np.array(type_list)[cell_type_indices]
        
        # 统计初始分配的细胞数
        type_counts = pd.Series(0, index=type_list)
        for t in initial_types:
            type_counts[t] += 1
        
        # 创建需要调整的类型列表
        adjustments = []
        for cell_type in type_list:
            target = int(target_counts[cell_type])
            current = type_counts[cell_type]
            if current != target:
                adjustments.append({
                    'cell_type': cell_type,
                    'difference': current - target,  # 正数表示过多，负数表示不足
                    'target': target
                })
        
        # 按照差异的绝对值排序，优先处理差异大的
        adjustments.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        # 处理需要调整的类型
        for adj in adjustments:
            cell_type = adj['cell_type']
            difference = adj['difference']
            
            if difference > 0:  # 需要减少的类型
                # 找到这个类型的所有细胞
                mask = initial_types == cell_type
                cells_of_type = np.where(mask)[0]
                
                if len(cells_of_type) > 0:
                    # 计算这些细胞对其他类型的概率
                    probs = cell_probs[cells_of_type]
                    probs[:, type_list.index(cell_type)] = -np.inf  # 排除当前类型
                    
                    # 选择最适合重新分配的细胞
                    best_alternative_scores = np.max(probs, axis=1)
                    cells_to_change = cells_of_type[np.argsort(best_alternative_scores)[-difference:]]
                    
                    # 为这些细胞重新分配类型
                    for cell_idx in cells_to_change:
                        new_type_idx = np.argmax(cell_probs[cell_idx])
                        initial_types[cell_idx] = type_list[new_type_idx]
                    
            elif difference < 0:  # 需要增加的类型
                # 找到其他类型的细胞
                mask = initial_types != cell_type
                other_cells = np.where(mask)[0]
                
                if len(other_cells) > 0:
                    # 计算这些细胞对当前类型的概率
                    probs = cell_probs[other_cells]
                    type_idx = type_list.index(cell_type)
                    
                    # 选择最适合改为当前类型的细胞
                    best_cells = other_cells[np.argsort(probs[:, type_idx])[-abs(difference):]]
                    
                    # 更新这些细胞的类型
                    initial_types[best_cells] = cell_type
        
        # 更新SVC_obs中的cell types
        SVC_obs.loc[valid_cells.index, 'cell_type'] = initial_types
    
    return SVC_obs


def assign_cell_types_easy(SVC_obs, cell_contributions, mode="max"):
    """
    Args:
        SVC_obs: DataFrame，包含'spot_name'和'cell_id'列
        cell_contributions: DataFrame，每个spot中存在的cell类型贡献
        mode: 模式，"max"或"random"
        
    Returns:
        SVC_obs: 更新后的SVC_obs，包含'cell_type'列
    """
    SVC_obs = SVC_obs.copy()
    cell_contributions = cell_contributions.copy()
    
    # 确保cell_contributions的index与SVC_obs中的spot_name匹配
    assert set(cell_contributions.index) == set(SVC_obs['spot_name'].unique()), "cell_contributions的index与SVC_obs中的spot_name不匹配"
    
    if 'cell_type' not in SVC_obs.columns:
        SVC_obs['cell_type'] = "Unknown"
    
    # 按spot分组处理
    spot_groups = SVC_obs.groupby('spot_name')
    
    for spot_name in tqdm(cell_contributions.index, desc="Assigning cell types"):
        # 获取当前spot中的所有cell
        spot_cells_df = spot_groups.get_group(spot_name)
        spot_cells = spot_cells_df['cell_id'].values
        
        # 获取当前spot的cell类型贡献
        spot_contributions = cell_contributions.loc[spot_name]
        
        if mode == "max":
            # 找出占比最大的cell类型
            max_type = spot_contributions.idxmax()
            # 为spot中的所有cell分配这个类型
            SVC_obs.loc[spot_cells_df.index, 'cell_type'] = max_type
        
        elif mode == "random":
            # 如果只有一个cell，直接分配占比最大的类型
            if len(spot_cells) == 1:
                max_type = spot_contributions.idxmax()
                SVC_obs.loc[spot_cells_df.index, 'cell_type'] = max_type
            else:
                # 找出前两种cell类型（最多贡献和次多贡献）
                sorted_types = spot_contributions.sort_values(ascending=False).index
                if len(sorted_types) >= 2:
                    max_type = sorted_types[0]
                    second_type = sorted_types[1]
                    
                    # 为大部分cell分配最大类型
                    SVC_obs.loc[spot_cells_df.index, 'cell_type'] = max_type
                    
                    # 随机选择一个cell改为次多类型
                    random_index = np.random.choice(spot_cells_df.index)
                    SVC_obs.loc[random_index, 'cell_type'] = second_type
                else:
                    # 如果只有1种类型，全部分配这个类型
                    max_type = sorted_types[0]
                    SVC_obs.loc[spot_cells_df.index, 'cell_type'] = max_type
        else:
            raise ValueError("mode必须是'max'或'random'")
    
    return SVC_obs



def sc_SVC_recon(adata_st, adata_sc, cell_contributions, SVC_obs, device='cuda', n_epochs=100, lr=0.1):
    """
    GPU-accelerated sc-like Spatial Virtual Cell (SVC) reconstruction with gradient-based optimization
    Args:
        adata_st: AnnData object of spatial transcriptomics data
        adata_sc: AnnData object of single-cell RNA-seq data with cell type annotations
        cell_contributions: array of cell type contributions for each spot (n_spots, n_types)
        SVC_obs: DataFrame containing 'spot_name', 'cell_id', and 'cell_type' columns
        device: str, 'cuda' or 'cpu' (default: 'cuda')
        n_epochs: int, number of optimization epochs (default: 100)
        lr: float, learning rate for gradient descent (default: 0.1)
    Returns:
        SVC_adata: AnnData object of reconstructed single-cell data
    """
    # 检查 GPU 可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
    
    # 获取细胞类型列表
    type_list = sorted(list(set(adata_sc.obs['cell_type'])))
    spots = SVC_obs['spot_name'].unique()
    
    # 构建 spot 到 index 的映射
    spot_to_idx = {spot: idx for idx, spot in enumerate(spots)}
    
    # 确保基因一致
    common_genes = np.intersect1d(adata_st.var_names, adata_sc.var_names)
    if len(common_genes) == 0:
        raise ValueError("No common genes found between adata_st and adata_sc.")
    
    adata_st = adata_st[:, common_genes].copy()
    adata_sc = adata_sc[:, common_genes].copy()
    
    # 获取表达矩阵
    X_st = adata_st.X if isinstance(adata_st.X, np.ndarray) else adata_st.X.toarray()
    X_sc = adata_sc.X if isinstance(adata_sc.X, np.ndarray) else adata_sc.X.toarray()
    
    # 转换为 PyTorch 张量并移到 GPU
    X_st = torch.from_numpy(X_st).float().to(device)
    X_sc = torch.from_numpy(X_sc).float().to(device)
    cell_contributions = torch.from_numpy(cell_contributions).float().to(device)
    
    # 按细胞类型分组单细胞
    cell_type_to_indices = {t: np.where(adata_sc.obs['cell_type'] == t)[0] for t in type_list}
    max_cells_per_type = max(len(indices) for indices in cell_type_to_indices.values())
    
    # 初始化分配概率矩阵 P: n_spots * n_types * max_cells_per_type
    P_logits = torch.randn(len(spots), len(type_list), max_cells_per_type, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([P_logits], lr=lr)
    
    # 构建单细胞表达矩阵和掩码
    X_sc_padded = torch.zeros(len(type_list), max_cells_per_type, len(common_genes), device=device)
    mask = torch.zeros(len(type_list), max_cells_per_type, device=device, dtype=torch.bool)
    for type_idx, cell_type in enumerate(type_list):
        indices = cell_type_to_indices[cell_type]
        X_sc_padded[type_idx, :len(indices)] = X_sc[indices]
        mask[type_idx, :len(indices)] = True
    
    # 优化循环
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 计算分配概率
        P = torch.softmax(P_logits, dim=2)  # n_spots * n_types * max_cells_per_type
        P = P * mask.unsqueeze(0)  # 屏蔽无效单细胞
        P = P / (P.sum(dim=2, keepdim=True) + 1e-10)  # 重新归一化
        
        # 计算加权表达
        Y = torch.einsum('ijk,jkl->ijl', P * cell_contributions.unsqueeze(2), X_sc_padded)  # n_spots * n_genes * n_types
        Y_sum = Y.sum(dim=2)  # n_spots * n_genes, 加权表达总和
        
        # 计算损失（负余弦相似度）
        X_st_norm = X_st / (torch.norm(X_st, dim=1, keepdim=True) + 1e-10)
        Y_norm = Y_sum / (torch.norm(Y_sum, dim=1, keepdim=True) + 1e-10)
        cosine_sim = (X_st_norm * Y_norm).sum(dim=1)
        loss = -cosine_sim.mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 获取最佳单细胞索引
    with torch.no_grad():
        P = torch.softmax(P_logits, dim=2) * mask.unsqueeze(0)
        best_sc_indices = torch.argmax(P, dim=2)  # n_spots * n_types
        # 转换为实际单细胞索引
        sc_indices_global = torch.zeros(len(spots), len(type_list), dtype=torch.long, device=device)
        for type_idx, cell_type in enumerate(type_list):
            indices = torch.tensor(cell_type_to_indices[cell_type], device=device)
            sc_indices_global[:, type_idx] = indices[best_sc_indices[:, type_idx]]
    
    # 构建 Y 矩阵
    Y = torch.zeros(len(spots), len(common_genes), len(type_list), device=device)
    for type_idx in range(len(type_list)):
        sc_expr = X_sc[sc_indices_global[:, type_idx]]  # n_spots * n_genes
        Y[:, :, type_idx] = sc_expr * cell_contributions[:, type_idx].unsqueeze(1)
    
    Y = Y * X_st.unsqueeze(2)
    
    # 根据 SVC_obs 索引基因表达
    spot_indices = torch.tensor([spot_to_idx[spot] for spot in SVC_obs['spot_name']], device=device)
    type_indices = torch.tensor([type_list.index(t) for t in SVC_obs['cell_type']], device=device)
    SVC_X = Y[spot_indices, :, type_indices]
    
    # 标准化基因表达
    SVC_X = SVC_X / (torch.sum(SVC_X, dim=1, keepdims=True) + 1e-10) * 1e6
    
    # 移回 CPU
    SVC_X = SVC_X.cpu().numpy()
    
    # 打印信息
    print(f"Number of cells processed: {len(SVC_obs)}")
    print(f"Number of unique spots: {len(spots)}")
    print(f"Shape of SVC_X: {SVC_X.shape}")
    
    # 构建 AnnData 对象
    SVC_adata = sc.AnnData(SVC_X)
    SVC_adata.var_names = common_genes
    SVC_adata.obs = SVC_obs.copy()
    
    return SVC_adata

def SVC_recon(adata_st, sc_ref, cell_contributions, SVC_obs, complete_mask=None):
    """
    sp-like Spatial Virtual Cell (SVC) reconstruction
    Args:
        adata_st: AnnData object of spatial transcriptomics data
        sc_ref: DataFrame of cell type reference profiles
        cell_contributions: array of cell type contributions for each spot
        SVC_obs: DataFrame containing 'spot_name', 'cell_id', and 'cell_type' columns
    Returns:
        SVC_adata: AnnData object of reconstructed single-cell data
    """
    cell_contributions = cell_contributions.values
    type_list = sorted(list(sc_ref.index))  # list of cell types
    spots = SVC_obs['spot_name'].unique()  # 获取唯一的spots
    
    # 构建spot到index的映射
    spot_to_idx = {spot: idx for idx, spot in enumerate(spots)}
    
    # 计算基因表达
    adata_spot = adata_st.copy()
    X = adata_spot.X if type(adata_spot.X) is np.ndarray else adata_spot.X.toarray()
    
    # 计算每个spot每个cell type的基因表达
    if complete_mask is None:
        Y = cell_contributions[:, np.newaxis, :] * sc_ref.values.T
        Y = Y / (np.sum(Y, axis=2, keepdims=True) + 1e-10)
        Y = Y * X[:, :, np.newaxis]  # n_spot * n_gene * n_type
    else: # todo: try to consider cell type
        complete_contributions = cell_contributions * complete_mask
        incomplete_contributions = cell_contributions * (1 - complete_mask)
        
        # 计算完整细胞的表达量并归一化
        Y_complete = complete_contributions[:, np.newaxis, :] * sc_ref.values.T
        Y_complete = Y_complete / (np.sum(Y_complete, axis=2, keepdims=True) + 1e-10)
        Y_complete = Y_complete * X[:, :, np.newaxis]

        # 计算不完整细胞的表达量（不归一化）
        Y_incomplete = incomplete_contributions[:, np.newaxis, :] * sc_ref.values.T
        Y_incomplete = Y_incomplete * X[:, :, np.newaxis]

        # 合并
        Y = Y_complete + Y_incomplete  # n_spot * n_gene * n_type

    
    # 直接根据 SVC_obs 中的spot和cell_type信息索引基因表达值
    spot_indices = np.array([spot_to_idx[spot] for spot in SVC_obs['spot_name']])
    type_indices = np.array([type_list.index(t) for t in SVC_obs['cell_type']])
    
    # 一次性获取所有细胞的基因表达
    SVC_X = Y[spot_indices, :, type_indices]
    
    # 标准化基因表达
    SVC_X = SVC_X / (np.sum(SVC_X, axis=1, keepdims=True) + 1e-10) * 1e4
    
    print(f"Number of cells processed: {len(SVC_obs)}")
    print(f"Number of unique spots: {len(spots)}")
    print(f"Shape of SVC_X: {SVC_X.shape}")
    
    # 构建新的AnnData对象
    SVC_adata = sc.AnnData(SVC_X)
    SVC_adata.var_names = adata_st.var_names
    SVC_adata.obs = SVC_obs.copy()  # 直接使用SVC_obs作为观测数据
    
    return SVC_adata



# import numpy as np
# import pandas as pd
# from scipy.sparse import csr_matrix
# from scipy.spatial import KDTree
# from anndata import AnnData
# from tqdm import tqdm

# def sp_SVC_recon(adata_st, SVC_obs, n_search_neighbors = 5, n_choose_neighbors=3, cell_type_col='cell_type'):
#     """
#     sp-like Spatial Virtual Cell (SVC) reconstruction with spot-specific sc_ref using matrix operations.
    
#     Args:
#         adata_st: AnnData object of spatial transcriptomics data, obs contains 'Max_type' and 'Confidence'
#         SVC_obs: DataFrame containing 'spot_name', 'cell_id', 'cell_type' columns
#         n_neighbors: Number of nearest spots to select based on highest confidence (out of 5 neighbors)
#         cell_type_col: Column name for cell types in SVC_obs
#     Returns:
#         SVC_adata: AnnData object of reconstructed single-cell data
#     """
#     # Get unique spots and cell types from SVC_obs
#     spots = SVC_obs['spot_name'].unique()
#     type_list = sorted(SVC_obs[cell_type_col].unique())

#     # Create spot to index mapping
#     spot_to_idx = {spot: idx for idx, spot in enumerate(spots)}
#     type_to_idx = {t: idx for idx, t in enumerate(type_list)}
    
#     # Get spot expression data
#     adata_spot = adata_st.copy()
#     X = adata_spot.X if isinstance(adata_spot.X, np.ndarray) else adata_spot.X.toarray()
    
#     # Get all spots with Max_type and Confidence
#     spot_coords = adata_spot.obsm['spatial']
#     spot_cell_types = adata_spot.obs['Max_type'].values
#     spot_confidences = adata_spot.obs['Confidence'].values
    
#     # Build spot-specific sc_ref for each cell type
#     spot_specific_refs = {spot: {} for spot in spots}
    
#     # Determine required cell types per spot
#     spot_cell_types_required = {spot: set(SVC_obs[SVC_obs['spot_name'] == spot][cell_type_col]) for spot in spots}
    
#     # Build KDTree for all spots and select high-confidence neighbors
#     kdtree = KDTree(spot_coords)
    
#     for cell_type in tqdm(type_list, desc="Building spot-specific references"):
#         # Get relevant spots requiring this cell type
#         relevant_spots = [spot for spot in spots if cell_type in spot_cell_types_required[spot]]
#         if not relevant_spots:
#             continue
        
#         # Get coordinates for relevant spots
#         relevant_coords = np.array([adata_spot.obs.loc[spot, ["x", "y"]] for spot in relevant_spots])
        
#         # Query 5 nearest neighbors
#         _, neighbor_indices = kdtree.query(relevant_coords, k=n_search_neighbors)
        
#         # Select top 3 neighbors based on confidence where Max_type matches cell_type
#         for spot, indices in zip(relevant_spots, neighbor_indices):
#             if len(indices) == 0:
#                 continue
#             # Get confidences and types for neighbors
#             neighbor_types = spot_cell_types[indices]
#             neighbor_confs = spot_confidences[indices]
            
#             # Filter neighbors where Max_type matches required cell_type
#             valid_mask = neighbor_types == cell_type
#             valid_indices = indices[valid_mask]
#             valid_confs = neighbor_confs[valid_mask]
            
#             # Select top 3 by confidence
#             if len(valid_confs) > 0:
#                 top_indices = valid_indices[np.argsort(valid_confs)[-n_choose_neighbors:]]
#                 if len(top_indices) > 0:
#                     # Get expression data for selected neighbors
#                     type_X = adata_spot.X[top_indices] if isinstance(adata_spot.X, np.ndarray) else adata_spot.X[top_indices].toarray()
#                     mean_expr = np.mean(type_X, axis=0)
#                     mean_expr = mean_expr / (np.sum(mean_expr) + 1e-10)  # Normalize
#                     spot_specific_refs[spot][cell_type] = mean_expr 
    
#     # Initialize Y array for spot-specific references
#     n_spots = len(spots)
#     n_genes = len(adata_st.var_names)
#     n_types = len(type_list)
#     Y = np.zeros((n_spots, n_genes, n_types))
    
#     # Fill Y with spot-specific reference profiles
#     for spot in spots:
#         spot_idx = spot_to_idx[spot]
#         for cell_type in spot_cell_types_required[spot]:
#             if cell_type in spot_specific_refs[spot]:
#                 type_idx = type_to_idx[cell_type]
#                 Y[spot_idx, :, type_idx] = spot_specific_refs[spot][cell_type]
    
#     # Step 1: Compute spot-level reconstructed expression (sum over cell types)
#     Y_total = np.sum(Y, axis=2)  # n_spot × n_gene
    
#     # Step 2: Compute scale factor
#     with np.errstate(divide='ignore', invalid='ignore'):
#         scale = X / (Y_total + 1e-10)  # n_spot × n_gene
#         scale[np.isinf(scale) | np.isnan(scale)] = 0
    
#     # Step 3: Apply scale to all cell types
#     Y_scaled = Y * scale[:, :, np.newaxis]  # n_spot × n_gene × n_type
    
#     # Step 4: Assign expressions to cells using matrix operations
#     spot_indices = np.array([spot_to_idx[spot] for spot in SVC_obs['spot_name']])
#     type_indices = np.array([type_to_idx[t] for t in SVC_obs[cell_type_col]])
    
#     # Initialize output array
#     SVC_X = np.zeros((len(SVC_obs), n_genes))
    
#     # Assign scaled expressions to each cell
#     SVC_X = Y_scaled[spot_indices, :, type_indices]  # n_cells × n_genes
    
#     # Step 5: Normalize final gene expression
#     SVC_X = SVC_X / (np.sum(SVC_X, axis=1, keepdims=True) + 1e-10) * 1e4
    
#     # Step 6: Create AnnData object
#     SVC_adata = AnnData(
#         X=csr_matrix(SVC_X),
#         obs=SVC_obs.copy(),
#         var=pd.DataFrame(index=adata_st.var_names)
#     )
    
#     print(f"Number of cells processed: {len(SVC_obs)}")
#     print(f"Number of unique spots: {len(spots)}")
#     print(f"Shape of SVC_X: {SVC_X.shape}")
    
#     return SVC_adata





import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from sklearn.neighbors import KDTree
from tqdm import tqdm

def sp_SVC_recon(adata_st, SVC_obs, sc_ref=None, 
                 mode="ref", 
                 n_search_neighbors=5, n_choose_neighbors=3, 
                 confidence_ts=0.9, min_num=10,
                 cell_type_col='cell_type'):
    """
    sp-like Spatial Virtual Cell (SVC) reconstruction with spot-specific or unified sc_ref using matrix operations.
    
    Args:
        adata_st: AnnData object of spatial transcriptomics data, obs contains 'Max_type' and 'Confidence'
        SVC_obs: DataFrame containing 'spot_name', 'cell_id', 'cell_type' columns
        sc_ref: DataFrame with cell types as index and genes as columns (used when mode="ref")
        mode: "ref" to use sc_ref, "sp-a" for unified confidence-based reference, "sp-s" for neighbor-based strategy
        n_search_neighbors: Number of nearest spots to search for neighbors (sp-s mode)
        n_choose_neighbors: Number of high-confidence neighbors to select (sp-s mode)
        confidence_ts: Confidence threshold for selecting pure spots (sp-a mode)
        min_num: Minimum number of spots per cell type for reference (sp-a mode)
        cell_type_col: Column name for cell types in SVC_obs
    Returns:
        SVC_adata: AnnData object of reconstructed single-cell data
    """
    # Input validation
    if mode == "ref" and sc_ref is None:
        raise ValueError("sc_ref must be provided when mode='ref'")
    
    # Get unique spots and cell types from SVC_obs
    spots = SVC_obs['spot_name'].unique()
    type_list = sorted(SVC_obs[cell_type_col].unique())
    
    # Create spot to index mapping
    spot_to_idx = {spot: idx for idx, spot in enumerate(spots)}
    type_to_idx = {t: idx for idx, t in enumerate(type_list)}
    
    # Get spot expression data
    adata_spot = adata_st.copy()
    X = adata_spot.X if isinstance(adata_spot.X, np.ndarray) else adata_spot.X.toarray()
    
    # Initialize spot-specific references
    spot_specific_refs = {spot: {} for spot in spots}
    
    # Determine required cell types per spot
    spot_cell_types_required = {spot: set(SVC_obs[SVC_obs['spot_name'] == spot][cell_type_col]) for spot in spots}
    
    if mode == "ref":
        # Validate sc_ref
        if not isinstance(sc_ref, pd.DataFrame):
            raise ValueError("sc_ref must be a pandas DataFrame with cell types as index and genes as columns")
        
        # Ensure sc_ref columns align with adata_st.var_names
        common_genes = np.intersect1d(sc_ref.columns, adata_st.var_names)
        if len(common_genes) == 0:
            raise ValueError("No common genes found between sc_ref and adata_st")
        
        # Reindex sc_ref to match adata_st.var_names
        sc_ref_reindexed = np.zeros((len(sc_ref.index), len(adata_st.var_names)))
        for i, gene in enumerate(adata_st.var_names):
            if gene in sc_ref.columns:
                sc_ref_reindexed[:, i] = sc_ref[gene].values
        
        for cell_type in tqdm(type_list, desc="Building sc_ref-based references"):
            if cell_type not in sc_ref.index:
                continue
            # Get expression profile for this cell type
            mean_expr = sc_ref_reindexed[sc_ref.index.get_loc(cell_type)]
            mean_expr = mean_expr / (np.sum(mean_expr) + 1e-10)  # Normalize
            # Assign to all relevant spots
            for spot in spots:
                if cell_type in spot_cell_types_required[spot]:
                    spot_specific_refs[spot][cell_type] = mean_expr
    
    elif mode == "sp-a":
        # Unified reference based on confidence threshold
        spot_cell_types = adata_spot.obs['Max_type'].values
        spot_confidences = adata_spot.obs['Confidence'].values
        
        # Build unified reference for each cell type
        sc_ref_unified = np.zeros((len(type_list), len(adata_st.var_names)), dtype=np.float32)
        for cell_type in tqdm(type_list, desc="Building unified references"):
            type_idx = type_to_idx[cell_type]
            # Filter spots by confidence threshold
            valid_mask = (spot_cell_types == cell_type) & (spot_confidences >= confidence_ts)
            valid_indices = np.where(valid_mask)[0]
            num_valid = len(valid_indices)
            
            print(f"Cell type '{cell_type}': {num_valid} pure spots found (confidence >= {confidence_ts})")
            
            # If fewer than min_num spots, take top min_num by confidence
            if num_valid < min_num:
                type_mask = spot_cell_types == cell_type
                type_indices = np.where(type_mask)[0]
                type_confs = spot_confidences[type_mask]
                if len(type_indices) > 0:
                    top_indices = type_indices[np.argsort(type_confs)[-(min_num):]]
                    print(f"  Cell type '{cell_type}': Increased from {num_valid} to {min_num} spots using top confidence")
                    selected_indices = top_indices
                else:
                    print(f"  Warning: Cell type '{cell_type}' has no spots; using zero reference")
                    selected_indices = []
            else:
                selected_indices = valid_indices
            
            # Compute mean expression for selected spots
            if len(selected_indices) > 0:
                type_X = adata_spot.X[selected_indices] if isinstance(adata_spot.X, np.ndarray) else adata_spot.X[selected_indices].toarray()
                mean_expr = np.mean(type_X, axis=0)
                mean_expr = mean_expr / (np.sum(mean_expr) + 1e-10)  # Normalize
                sc_ref_unified[type_idx, :] = mean_expr
            
            # Assign unified reference to all relevant spots
            for spot in spots:
                if cell_type in spot_cell_types_required[spot]:
                    spot_specific_refs[spot][cell_type] = sc_ref_unified[type_idx, :]
    
    elif mode == "sp-s":
        # Neighbor-based strategy
        spot_coords = adata_spot.obsm['spatial']
        spot_cell_types = adata_spot.obs['Max_type'].values
        spot_confidences = adata_spot.obs['Confidence'].values
        
        # Build KDTree for all spots
        kdtree = KDTree(spot_coords)
        
        for cell_type in tqdm(type_list, desc="Building spot-specific references"):
            # Get relevant spots requiring this cell type
            relevant_spots = [spot for spot in spots if cell_type in spot_cell_types_required[spot]]
            if not relevant_spots:
                continue
            
            # Get coordinates for relevant spots
            relevant_coords = np.array([adata_spot.obsm['spatial'][adata_spot.obs.index.get_loc(spot)] for spot in relevant_spots])
            
            # Query nearest neighbors
            _, neighbor_indices = kdtree.query(relevant_coords, k=n_search_neighbors)
            
            # Select top neighbors based on confidence
            for spot, indices in zip(relevant_spots, neighbor_indices):
                if len(indices) == 0:
                    continue
                # Get confidences and types for neighbors
                neighbor_types = spot_cell_types[indices]
                neighbor_confs = spot_confidences[indices]
                
                # Filter neighbors where Max_type matches required cell_type
                valid_mask = neighbor_types == cell_type
                valid_indices = indices[valid_mask]
                valid_confs = neighbor_confs[valid_mask]
                
                # Select top by confidence
                if len(valid_confs) > 0:
                    top_indices = valid_indices[np.argsort(valid_confs)[-n_choose_neighbors:]]
                if len(top_indices) > 0:
                    # Get expression data for selected neighbors
                    type_X = adata_spot.X[top_indices] if isinstance(adata_spot.X, np.ndarray) else adata_spot.X[top_indices].toarray()
                    mean_expr = np.mean(type_X, axis=0)
                    mean_expr = mean_expr / (np.sum(mean_expr) + 1e-10)  # Normalize
                    spot_specific_refs[spot][cell_type] = mean_expr
    
    else:
        raise ValueError("Mode must be 'ref', 'sp-a', or 'sp-s'")
    
    # Initialize Y array for spot-specific references
    n_spots = len(spots)
    n_genes = len(adata_st.var_names)
    n_types = len(type_list)
    Y = np.zeros((n_spots, n_genes, n_types))
    
    # Fill Y with spot-specific reference profiles
    for spot in spots:
        spot_idx = spot_to_idx[spot]
        for cell_type in spot_cell_types_required[spot]:
            if cell_type in spot_specific_refs[spot]:
                type_idx = type_to_idx[cell_type]
                Y[spot_idx, :, type_idx] = spot_specific_refs[spot][cell_type]
    
    # Step 1: Compute spot-level reconstructed expression
    Y_total = np.sum(Y, axis=2)  # n_spot × n_gene
    
    # Step 2: Compute scale factor
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = X / (Y_total + 1e-10)  # n_spot × n_gene
        scale[np.isinf(scale) | np.isnan(scale)] = 0
    
    # Step 3: Apply scale to all cell types
    Y_scaled = Y * scale[:, :, np.newaxis]  # n_spot × n_gene × n_type
    
    # Step 4: Assign expressions to cells
    spot_indices = np.array([spot_to_idx[spot] for spot in SVC_obs['spot_name']])
    type_indices = np.array([type_to_idx[t] for t in SVC_obs[cell_type_col]])
    
    # Initialize output array
    SVC_X = np.zeros((len(SVC_obs), n_genes))
    
    # Assign scaled expressions
    SVC_X = Y_scaled[spot_indices, :, type_indices]  # n_cells × n_genes
    
    # Step 5: Normalize final gene expression
    SVC_X = SVC_X / (np.sum(SVC_X, axis=1, keepdims=True) + 1e-10) * 1e4
    
    # Step 6: Create AnnData object
    SVC_adata = AnnData(
        X=csr_matrix(SVC_X),
        obs=SVC_obs.copy(),
        var=pd.DataFrame(index=adata_st.var_names)
    )
    
    return SVC_adata


# import pandas as pd
# import numpy as np
# from scipy.sparse import issparse, csr_matrix
# from scipy.spatial import KDTree
# from anndata import AnnData
# from tqdm import tqdm

# def sp_SVC_recon_sp(adata_st, SVC_obs, n_neighbors=3, cell_type_col='Level1'):
#     """
#     sp-like Spatial Virtual Cell (SVC) reconstruction with spot-specific sc_ref using matrix operations.
    
#     Args:
#         adata_st: AnnData object of spatial transcriptomics data, obs['is_pure'] indicates pure spots
#         SVC_obs: DataFrame containing 'spot_name', 'cell_id', 'true_cell_type' columns
#         n_neighbors: Number of nearest pure spots to consider for each cell type
#         cell_type_col: Column name in adata_st.obs containing cell type information
    
#     Returns:
#         SVC_adata: AnnData object of reconstructed single-cell data
#     """
#     # Get unique spots and cell types from SVC_obs
#     spots = SVC_obs['spot_name'].unique()
#     type_list = sorted(SVC_obs['true_cell_type'].unique())

#     # Create spot to index mapping
#     spot_to_idx = {spot: idx for idx, spot in enumerate(spots)}
#     type_to_idx = {t: idx for idx, t in enumerate(type_list)}
    
#     # Get spot expression data
#     adata_spot = adata_st.copy()
#     X = adata_spot.X if isinstance(adata_spot.X, np.ndarray) else adata_spot.X.toarray()
    
#     # Identify pure spots
#     pure_spots = adata_st[adata_st.obs['is_pure']].copy()
#     pure_coords = pure_spots.obsm['spatial']
#     pure_cell_types = pure_spots.obs[cell_type_col].values
    
#     # Build spot-specific sc_ref for each cell type
#     spot_specific_refs = {spot: {} for spot in spots}
    
#     # Determine required cell types per spot
#     spot_cell_types = {spot: set(SVC_obs[SVC_obs['spot_name'] == spot]['cell_type']) for spot in spots}
    
#     # Build KDTree for each cell type's pure spots and compute mean expressions
#     for cell_type in tqdm(type_list, desc="Building spot-specific references"):
#         # Get pure spots for this cell type
#         type_mask = pure_cell_types == cell_type
#         if not np.any(type_mask):
#             continue  # Skip if no pure spots for this cell type
#         type_coords = pure_coords[type_mask]
#         type_X = pure_spots.X[type_mask] if isinstance(pure_spots.X, np.ndarray) else pure_spots.X[type_mask].toarray()
        
#         # Build KDTree for this cell type's pure spots
#         kdtree = KDTree(type_coords)
        
#         # Query n_neighbors nearest pure spots for all relevant spots
#         relevant_spots = [spot for spot in spots if cell_type in spot_cell_types[spot]]
#         if not relevant_spots:
#             continue
#         relevant_coords = np.array([adata_st.obs.loc[spot, ["x", "y"]] for spot in relevant_spots])
        
#         # Find n_neighbors nearest neighbors
#         _, neighbor_indices = kdtree.query(relevant_coords, k=n_neighbors)
        
#         # Compute mean expression for each spot
#         for spot, indices in zip(relevant_spots, neighbor_indices):
#             if len(indices) > 0:
#                 # Ensure indices are valid
#                 valid_indices = [i for i in indices if i < len(type_X)]
#                 if valid_indices:
#                     mean_expr = np.mean(type_X[valid_indices], axis=0)
#                     mean_expr = mean_expr / (np.sum(mean_expr) + 1e-10)  # Normalize
#                     # total_count = SVC_obs[SVC_obs['spot_name'] == spot]['total_counts'].mean()
#                     total_count = 1
#                     spot_specific_refs[spot][cell_type] = mean_expr * total_count
    
#     # Initialize Y array for spot-specific references
#     n_spots = len(spots)
#     n_genes = len(adata_st.var_names)
#     n_types = len(type_list)
#     Y = np.zeros((n_spots, n_genes, n_types))
    
#     # Fill Y with spot-specific reference profiles
#     for spot in spots:
#         spot_idx = spot_to_idx[spot]
#         for cell_type in spot_cell_types[spot]:
#             if cell_type in spot_specific_refs[spot]:
#                 type_idx = type_to_idx[cell_type]
#                 Y[spot_idx, :, type_idx] = spot_specific_refs[spot][cell_type]
    
#     # Step 1: Compute spot-level reconstructed expression (sum over cell types)
#     Y_total = np.sum(Y, axis=2)  # n_spot × n_gene
    
#     # Step 2: Compute scale factor
#     with np.errstate(divide='ignore', invalid='ignore'):
#         scale = X / (Y_total + 1e-10)  # n_spot × n_gene
#         scale[np.isinf(scale) | np.isnan(scale)] = 0
    
#     # Step 3: Apply scale to all cell types
#     Y_scaled = Y * scale[:, :, np.newaxis]  # n_spot × n_gene × n_type
    
#     # Step 4: Assign expressions to cells using matrix operations
#     spot_indices = np.array([spot_to_idx[spot] for spot in SVC_obs['spot_name']])
#     type_indices = np.array([type_to_idx[t] for t in SVC_obs['true_cell_type']])
#     # total_counts = SVC_obs['total_counts'].values
    
#     # Initialize output array
#     SVC_X = np.zeros((len(SVC_obs), n_genes))
    
#     # Assign scaled expressions to each cell
#     SVC_X = Y_scaled[spot_indices, :, type_indices]  # n_cells × n_genes
    
#     # Step 5: Normalize final gene expression
#     SVC_X = SVC_X / (np.sum(SVC_X, axis=1, keepdims=True) + 1e-10) * 1e4
    
#     # Step 6: Create AnnData object
#     SVC_adata = AnnData(
#         X=csr_matrix(SVC_X),
#         obs=SVC_obs.copy(),
#         var=pd.DataFrame(index=adata_st.var_names)
#     )
    
#     print(f"Number of cells processed: {len(SVC_obs)}")
#     print(f"Number of unique spots: {len(spots)}")
#     print(f"Shape of SVC_X: {SVC_X.shape}")
    
#     return SVC_adata



from scipy.spatial import KDTree
from scipy.sparse import issparse
import numpy as np
import pandas as pd
def create_meta_cells(adata, sc_pixel, n_neighbors=5, cell_type_col='Level1', 
                     bandwidth=None, min_neighbors=3, agg_method='weighted_mean'):
    """
    通过聚合同类型邻近细胞的表达数据来创建meta cells
    
    参数:
    - adata: AnnData对象，包含归一化表达矩阵和空间坐标
    - sc_pixel: 空间分辨率，用于定义邻居搜索半径
    - n_neighbors: 最大邻居数量
    - cell_type_col: 细胞类型列名
    - bandwidth: 高斯核带宽，用于空间权重计算（如果为None则自动估计）
    - min_neighbors: 构建meta cell所需的最小邻居数量
    - agg_method: 聚合方法，可选 'weighted_mean' 或 'concat'
    
    返回:
    - meta_adata: 包含meta cells的新AnnData对象
    """
    
    sc_pixel = sc_pixel * n_neighbors
    # 复制原始AnnData对象
    meta_adata = adata.copy()
    
    # 基本参数检查
    if 'spatial' not in adata.obsm:
        try:
            adata.obsm['spatial'] = adata.obs[["x","y"]].values.astype(float)
        except KeyError:
            raise ValueError("Missing spatial coordinates")
    if cell_type_col not in adata.obs:
        raise ValueError(f"Missing cell type column: {cell_type_col}")
    
    spatial_coords = adata.obsm['spatial']
    cell_types = adata.obs[cell_type_col].values
    
    # 获取表达矩阵
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    
    # 自动估计带宽（如果未提供）
    if bandwidth is None:
        # 使用sc_pixel作为参考
        bandwidth = sc_pixel / 2
        print(f"Set bandwidth to sc_pixel/2: {bandwidth:.4f}")
    
    # 初始化结果数组，保持原始顺序
    if agg_method == 'weighted_mean':
        meta_expressions = X.copy()  # 默认保持原始表达
    else:  # concat模式
        # 初始化为最大可能大小
        meta_expressions = np.zeros((len(adata), n_neighbors + 1, X.shape[1]))
        meta_expressions[:, 0, :] = X  # 第一个位置放原始细胞
    
    meta_cell_sizes = np.ones(len(adata), dtype=int)  # 默认大小为1
    meta_cell_indices = [[i] for i in range(len(adata))]  # 每个细胞初始只包含自己
    
    # 按细胞类型处理
    unique_cell_types = np.unique(cell_types)
    for cell_type in tqdm(unique_cell_types, desc="Processing cell types"):
        # 获取当前类型的细胞
        type_mask = cell_types == cell_type
        type_indices = np.where(type_mask)[0]
        type_coords = spatial_coords[type_mask]
        type_expr = X[type_mask]
        
        # 构建KDTree
        kdtree = KDTree(type_coords)
        
        # 处理每个细胞
        for local_idx, global_idx in enumerate(type_indices):
            # 在sc_pixel范围内搜索邻居
            neighbors = kdtree.query_ball_point(type_coords[local_idx], sc_pixel)
            
            # 排除自身
            neighbors = [n for n in neighbors if n != local_idx]
            
            # 如果邻居数量超过n_neighbors，随机选择n_neighbors个
            if len(neighbors) > n_neighbors:
                neighbors = np.random.choice(neighbors, n_neighbors, replace=False)
            
            if len(neighbors) < min_neighbors:
                continue  # 保持原始表达值
            
            # 计算到中心细胞的距离
            distances = np.array([np.linalg.norm(type_coords[local_idx] - type_coords[n]) for n in neighbors])
            
            # 计算空间权重
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            weights = weights / (np.sum(weights) + 1e-6)
            
            # 获取邻居表达数据
            neighbor_expr = type_expr[neighbors]
            
            if agg_method == 'weighted_mean':
                # 计算加权平均表达
                meta_expr = np.sum(neighbor_expr * weights[:, None], axis=0)
                meta_expressions[global_idx] = meta_expr
            
            elif agg_method == 'concat':
                # 填充邻居表达数据
                n_neighbors_actual = len(neighbors)
                meta_expressions[global_idx, 1:n_neighbors_actual+1] = neighbor_expr
            
            # 更新meta cell信息
            meta_cell_sizes[global_idx] = len(neighbors) + 1
            meta_cell_indices[global_idx].extend([type_indices[n] for n in neighbors])
    
    # 更新meta_adata
    if agg_method == 'weighted_mean':
        if issparse(adata.X):
            meta_expressions = csr_matrix(meta_expressions)
    else:  # concat模式
        if issparse(adata.X):
            meta_expressions = csr_matrix(meta_expressions.reshape(len(adata), -1))
        else:
            meta_expressions = meta_expressions.reshape(len(adata), -1)
    
    # 更新表达矩阵
    meta_adata.X = meta_expressions
    
    # 添加meta cell信息
    meta_adata.obs['meta_cell_size'] = meta_cell_sizes
    meta_adata.uns['meta_cell_indices'] = meta_cell_indices
    
    # 打印统计信息
    enhanced_cells = np.sum(meta_cell_sizes > 1)
    print(f"Enhanced {enhanced_cells} cells ({enhanced_cells/len(adata)*100:.1f}%) with neighbors")
    print(f"Average meta cell size: {np.mean(meta_cell_sizes):.2f} cells")
    
    return meta_adata


def normalize_SVC_adata(adata_st, SVC_adata):
    """
    Normalize SVC_adata by aligning with adata_st and redistributing gene expressions based on initial proportions.
    
    Args:
        adata_st: AnnData object of spatial transcriptomics data with obs containing 'spot'
        SVC_adata: AnnData object with obs containing 'spot_name' and 'cell_id', and X as initial expression proportions
    
    Returns:
        SVC_adata: AnnData object with normalized single-cell expression data
    """
    # Ensure inputs are valid
    if 'spot' not in adata_st.obs.columns:
        adata_st.obs['spot'] = adata_st.obs.index
    if 'spot_name' not in SVC_adata.obs.columns or 'cell_id' not in SVC_adata.obs.columns:
        raise ValueError("SVC_adata.obs must contain 'spot_name' and 'cell_id' columns")

    # Get unique spots from SVC_adata
    spots = SVC_adata.obs['spot_name'].unique()
    
    # Create spot to index mapping
    spot_to_idx = {spot: idx for idx, spot in enumerate(spots)}
    
    # Get spot expression data from adata_st
    adata_spot = adata_st.copy()
    X = adata_spot.X if isinstance(adata_spot.X, np.ndarray) else adata_spot.X.toarray()
    
    # Align spots between adata_st and SVC_adata
    valid_spots = adata_st.obs['spot'].isin(spots)
    if not valid_spots.any():
        raise ValueError("No matching spots found between adata_st and SVC_adata")
    adata_spot = adata_spot[valid_spots].copy()
    X = adata_spot.X if isinstance(adata_spot.X, np.ndarray) else adata_spot.X.toarray()
    
    # Get initial proportions from SVC_adata
    SVC_X = SVC_adata.X if isinstance(SVC_adata.X, np.ndarray) else SVC_adata.X.toarray()
    
    # Initialize output array for normalized single-cell expressions
    normalized_SVC_X = np.zeros_like(SVC_X)
    
    # Assign spot indices for each cell in SVC_adata
    spot_indices = np.array([spot_to_idx[spot] for spot in SVC_adata.obs['spot_name']])
    
    # Normalize expressions for each spot
    for spot in tqdm(spots, desc = "Normalizing SVC_adata"):
        spot_idx = spot_to_idx[spot]
        # Get cells in this spot from SVC_adata
        spot_mask = SVC_adata.obs['spot_name'] == spot
        cell_indices = np.where(spot_mask)[0]
        
        if len(cell_indices) == 0:
            continue
        
        # Get proportions for cells in this spot
        cell_proportions = SVC_X[cell_indices]
        
        # Normalize proportions to sum to 1 across cells in the spot
        cell_proportions_sum = np.sum(cell_proportions, axis=0, keepdims=True)
        cell_proportions = cell_proportions / (cell_proportions_sum + 1e-10)
        
        # Get corresponding spot expression from adata_st
        st_spot_idx = np.where(adata_st.obs['spot'] == spot)[0]
        if len(st_spot_idx) == 0:
            continue
        spot_expr = X[st_spot_idx[0]]
        
        # Distribute spot expression to cells based on proportions
        for i, cell_idx in enumerate(cell_indices):
            normalized_SVC_X[cell_idx] = cell_proportions[i] * spot_expr
    
    # Normalize final gene expression to a fixed total count (following sp_SVC_recon_sp)
    normalized_SVC_X = normalized_SVC_X / (np.sum(normalized_SVC_X, axis=1, keepdims=True) + 1e-10) * 1e4
    
    # Create new AnnData object
    normalized_SVC_adata = sc.AnnData(normalized_SVC_X, obs=SVC_adata.obs.copy(), var=SVC_adata.var.copy())
    
    return normalized_SVC_adata




# todo: just display here now
def two_step_REVISE(X, adata_sc, sc_ref, type_list, key_type,
                cells_on_spot, morphology_features, feature_list=None, lambda_morph=0.0,
                batch_prior=2, adam_params=None, device='cuda', n_epoch=8000, plot=False,
                threshold=0.9, top_n=3, max_n=10):
    """
    两次变分流程，结合形态学信息
    Args:
        X: ST数据矩阵
        adata_sc: 单细胞数据AnnData对象
        sc_ref: 单细胞参考矩阵
        type_list: 细胞类型列表
        key_type: 单细胞数据中细胞类型的key
        threshold: 判定为纯净spot的阈值
        top_n: 当某类型没有纯净spot时，选择比例最高的前n个spot
        max_n: 每种细胞类型最多使用的spot数量
    Returns:
        cell_contributions_pure: 第二次变分结果
        sc_ref_pure: 纯净的单细胞参考矩阵
    """
    if plot:
        plot_file_name_1 = "first.png"
        plot_file_name_2 = "second.png"
    # Step 1: 第一次变分
    print("First deconvolution...")
    cell_contributions = get_cell_contributions(X, adata_sc, sc_ref, type_list, key_type,
                                         cells_on_spot, morphology_features, feature_list,
                                         lambda_morph=lambda_morph, device=device,
                                         n_epoch=n_epoch, adam_params=adam_params,
                                         batch_prior=batch_prior, plot_file_name=plot_file_name_1)
    cell_contributions_first = cell_contributions.copy()


    # Step 2: 构建纯净参考矩阵
    print("\nConstructing pure reference matrix...")
    sc_ref_pure, pure_spots_info = construct_pure_reference(X, cell_contributions, sc_ref, type_list,
                                                          threshold=threshold, top_n=top_n, max_n=max_n)
    
    # 打印统计信息
    print("\nSpot statistics and similarities:")
    print("Cell Type | #Spots | Source | Ref Sim | contributions (mean/min/max/median) | Spot Sims (mean/min/max/median)")
    print("-" * 120)
    for info in pure_spots_info:
        print(f"{info['cell_type']:15} | {info['n_spots']:6d} | {info['source']:5} | {info['ref_similarity']:8.3f} | "
              f"{info['mean_prop']:6.3f}/{info['min_prop']:6.3f}/{info['max_prop']:6.3f}/{info['median_prop']:6.3f} | "
              f"{info['mean_spot_sim']:6.3f}/{info['min_spot_sim']:6.3f}/{info['max_spot_sim']:6.3f}/{info['median_spot_sim']:6.3f}")

    # Step 3: 第二次变分，使用batch_prior=0
    print("\nSecond deconvolution...")
    cell_contributions_pure = get_cell_contributions(X, adata_sc, sc_ref_pure, type_list, key_type,
                                              cells_on_spot, morphology_features, feature_list,
                                              lambda_morph=lambda_morph, device=device,
                                              n_epoch=n_epoch, adam_params=adam_params,
                                              batch_prior=0, plot_file_name=plot_file_name_2)

    return cell_contributions_pure, sc_ref_pure, cell_contributions_first


