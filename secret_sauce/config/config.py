from dataclasses import MISSING, dataclass
from secret_sauce.config.data.dataset import SongsDatasetConfig


@dataclass
class Config:
    name: str = "ss"
    dataset: SongsDatasetConfig = SongsDatasetConfig()