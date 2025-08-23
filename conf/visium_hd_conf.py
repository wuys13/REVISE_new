from dataclasses import dataclass
from dataclasses import field
import os


@dataclass
class VisiumHDConf:
    patient_id: str = "P1CRC"
    data_root_path: str = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/wuyushuai/store_data/CRC_processed"
    sc_file: str = "adata_sc_all_reanno.h5ad"
    celltype_col: str = "Level1"
    mode: str = "tacco_revise"
    dist_mode: str = "spatial"
    result_root_path: str = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/jiaoyifeng/code/REVISE_new/0_records/run_REVISE_seg"
    result_dir: str = field(init=False)

    def __post_init__(self):
        self.result_dir = os.path.join(self.result_root_path, f"{self.patient_id}_real")
        self.st_file = os.path.join(self.data_root_path, self.sc_file)
        self.sc_file = os.path.join(self.data_root_path, self.sc_file)
