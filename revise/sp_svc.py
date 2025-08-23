from revise.base_svc import BaseSVC
from revise.tacco import tacco_anno
from revise.graph_smooth import get_spatial_graph
from revise.graph_smooth import get_expression_graph
from revise.graph_smooth import laplacian_smooth_expression
import scanpy as sc
from tqdm import tqdm


class SpSVC(BaseSVC):
    def __init__(self, st_adata, sc_ref_adata, config):
        super().__init__(st_adata, sc_ref_adata, config)

    def annotate(self, *args, **kwargs):
        # Implement the annotation logic here
        self.st_adata = tacco_anno(self.st_adata, self.sc_ref_adata, self.config.celltype_col)

    def reconstruct(self, *args, **kwargs):
        # Implement the reconstruction logic here
        adata_by_cell_type = []
        for cell_type in tqdm(self.st_adata.obs['max_type'].unique().tolist()):
            adata_tmp = self.st_adata[self.st_adata.obs['max_type'] == cell_type]
            adata_tmp = get_spatial_graph(adata_tmp, n_neighbors=self.config.graph_n_neighbors)
            # adata_tmp = get_expression_graph(adata_tmp, n_pca=self.config.graph_n_pca, n_neighbors=self.config.graph_n_neighbors)
            print(f"cell type: {cell_type}, n_spots: {adata_tmp.n_obs}, connectivites: {adata_tmp.obsp['spatial_connectivities'].shape}")
            for _ in range(self.config.iter_num):
                adata_tmp = laplacian_smooth_expression(adata_tmp, alpha=self.config.graph_st_alpha,
                                                        obsp_key='spatial_connectivities')
            adata_by_cell_type.append(adata_tmp)
        self.st_adata = sc.concat(adata_by_cell_type)

    def write_result(self):
        self.st_adata.write(self.config.st_result_file)
