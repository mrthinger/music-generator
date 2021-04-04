from abc import ABCMeta, abstractmethod
import os
from secret_sauce.util.io import get_duration_sec
from secret_sauce.config.data.dataset import SongsDatasetConfig
from secret_sauce.config.data.disk_datasource import DiskDataSourceConfig
import glob
import numpy as np
import torch
import torchaudio


class IDataSource:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_song(self, idx: int, offset: float) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_total_duration(self) -> tuple[np.ndarray]:
        raise NotImplementedError


class DiskDataSource(IDataSource):
    def __init__(self, cfg: SongsDatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.songs: list[str] = glob.glob(f"{cfg.disk_datasource.data_path}/*.wav")

        if self.cfg.disk_datasource.cache:
            self.songs_cache = torch.Tensor()
            for song in self.songs:
                wave, _ = torchaudio.load(song)
                self.songs_cache = torch.cat((self.songs_cache, wave), dim=-1)

    def get_song(self, idx: int, offset: float) -> torch.Tensor:

        frame_offset = int(offset * self.cfg.sample_rate)
        num_frames = int(self.cfg.sample_len * self.cfg.sample_rate)

        if self.cfg.disk_datasource.cache:
            return self.songs_cache[..., frame_offset : frame_offset + num_frames]

        else:
            with open(self.songs[idx], mode="rb") as song:
                wave, sample_rate = torchaudio.load(
                    song, frame_offset=frame_offset, num_frames=num_frames
                )
                assert (
                    sample_rate == self.cfg.sample_rate
                ), f"samplerate off!: {self.songs[idx]}"

            return wave

    def get_total_duration(self) -> tuple[np.ndarray]:
        durations = np.array([get_duration_sec(song) for song in self.songs])
        cumsum = np.cumsum(durations)

        return durations, cumsum
