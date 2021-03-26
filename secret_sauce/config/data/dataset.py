from dataclasses import dataclass
from secret_sauce.config.data.disk_datasource import DiskDataSourceConfig


@dataclass
class SongsDatasetConfig:
    sample_len: float = 3.0
    sample_rate: int = 22000
    train_test_split: float = 0.9
    batch_size: int = 32
    disk_datasource: DiskDataSourceConfig = DiskDataSourceConfig()
