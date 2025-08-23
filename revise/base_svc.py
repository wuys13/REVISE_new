class BaseSVC:
    def __init__(self, st_adata, sc_ref_adata, config):
        self.st_adata = st_adata
        self.sc_ref_adata = sc_ref_adata
        self.config = config

    def annotate(self, *args, **kwargs):
        raise NotImplementedError("Annotate method not implemented.")

    def reconstruct(self, *args, **kwargs):
        raise NotImplementedError("Reconstruct method not implemented.")
