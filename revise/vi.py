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
                          device='cuda', n_epoch=8000, adam_params=None,
                          batch_prior=2, plot_file_name = None,
                          ):
    type_list=sc_ref.index.tolist()

    if feature_list is None:
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

    return cell_contributions
