from dataclasses import dataclass
from dataclasses import field
import os


@dataclass
class VisiumHDConf:
    patient_id: str = "P1CRC"
    data_root_path: str = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/wuyushuai/store_data/CRC_processed"
    sc_file: str = "adata_sc_all_reanno.h5ad"
    celltype_col: str = "Level1"
    result_root_path: str = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/jiaoyifeng/code/REVISE_new/0_records/run_REVISE_seg"
    result_dir: str = field(init=False)
    graph_n_pca: int = 30
    graph_n_neighbors: int = 5
    graph_st_alpha: float = 0.2
    iter_num: int = 2
    single_cell_size: float = 8.0
    pixel_size: float = 0.27401259157195484
    use_cell_size: bool = False
    bandwidth_temp: float = 0.5

    def __post_init__(self):
        self.result_dir = os.path.join(self.result_root_path, f"{self.patient_id}_real")
        self.st_file = f"{self.patient_id}_HD.h5ad"
        self.st_file_path = os.path.join(self.data_root_path, self.st_file)
        self.sc_file_path = os.path.join(self.data_root_path, self.sc_file)
        self.st_result_file = os.path.join(self.result_dir, f"{self.st_file}_REVISE.h5ad")
