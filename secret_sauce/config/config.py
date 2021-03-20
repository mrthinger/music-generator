from dataclasses import MISSING, dataclass
from secret_sauce.config.data.dataset import SongsDatasetConfig
from secret_sauce.config.data.disk_datasource import DiskDataSourceConfig


@dataclass
class Config:
    name: str = "ss"
    disk_datasource: DiskDataSourceConfig = DiskDataSourceConfig()
    dataset: SongsDatasetConfig = SongsDatasetConfig()