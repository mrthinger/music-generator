from dataclasses import MISSING, dataclass
from typing import Optional
from secret_sauce.config.basic_transformer_config import BasicTransformerConfig
from secret_sauce.config.vqvae_config import VQVAEConfig
from secret_sauce.config.data.dataset import SongsDatasetConfig
import torch
import os
from datetime import datetime


@dataclass
class Config:
    name: str = "TensorBeat Generation"
    dataset: SongsDatasetConfig = SongsDatasetConfig()
    cuda_avail: bool = torch.cuda.is_available()
    vqvae: VQVAEConfig = VQVAEConfig()
    transformer: BasicTransformerConfig = BasicTransformerConfig()

    epochs: int = 10000

    save_name: str = (
        os.getenv("SAVE_NAME")
        if os.getenv("SAVE_NAME") != None
        else datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    )
    save_dir: str = f"outputs/{save_name}"
    save_every_epochs: int = 500
    save_zeroth_epoch: bool = False


    load_dir: Optional[str] = None
    load_tag: Optional[str] = None

    generate_every_epochs: int = 1
