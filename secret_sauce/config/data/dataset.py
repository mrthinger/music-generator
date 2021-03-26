from dataclasses import dataclass
from secret_sauce.config.data.disk_datasource import DiskDataSourceConfig


@dataclass
class SongsDatasetConfig:
    sample_len: float = 3
    sample_rate: int = 22000
    disk_datasource: DiskDataSourceConfig = DiskDataSourceConfig()
