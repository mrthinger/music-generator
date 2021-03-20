from dataclasses import dataclass
import os

import hydra


@dataclass
class DiskDataSourceConfig:
    data_path: str = hydra.utils.to_absolute_path("data")
