from abc import ABCMeta, abstractmethod
import os
from secret_sauce.config.data.disk_datasource import DiskDataSourceConfig
import glob


class IDataSource:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_num_songs(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_song_uri(self, idx: int) -> str:
        raise NotImplementedError


class DiskDataSource(IDataSource):
    def __init__(self, cfg: DiskDataSourceConfig) -> None:
        super().__init__()
        self.songs: list[str] = glob.glob(f"${cfg.data_path}/*.wav")

    def get_num_songs(self) -> int:
        return len(self.songs)

    def get_song_uri(self, idx: int) -> str:
        return self.songs[idx]
