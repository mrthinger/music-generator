from dataclasses import MISSING, dataclass
from secret_sauce.config.data.dataset import SongsDatasetConfig
import torch
import os
from datetime import datetime


@dataclass
class Config:
    name: str = "TensorBeat Generation"
    dataset: SongsDatasetConfig = SongsDatasetConfig()
    cuda_avail: bool = torch.cuda.is_available()

    epochs: int = 100

    save_name: str = (
        os.getenv("SAVE_NAME")
        if os.getenv("SAVE_NAME") != None
        else datetime.now().strftime("%d%m%Y%H%M%S")
    )
    save_dir: str = f"outputs/{save_name}"
