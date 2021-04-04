from dataclasses import MISSING, dataclass
from secret_sauce.config.data.dataset import SongsDatasetConfig
import torch


@dataclass
class Config:
    name: str = "TensorBeat Generation"
    dataset: SongsDatasetConfig = SongsDatasetConfig()
    cuda_avail: bool = torch.cuda.is_available()

    epochs: int = 100

    save_name: str = "v2"
    save_dir: str = f"outputs/{save_name}"
