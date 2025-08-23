from revise.base_svc import BaseSVC
from revise.tacco import tacco_anno


class SpSVC(BaseSVC):
    def __init__(self, st_adata, sc_ref_adata, config):
        super().__init__(st_adata, sc_ref_adata, config)

    def annotate(self, *args, **kwargs):
        # Implement the annotation logic here
        self.st_adata = tacco_anno(self.st_adata, self.sc_ref_adata, self.config.celltype_col)

    def reconstruct(self, *args, **kwargs):
        # Implement the reconstruction logic here
        pass
