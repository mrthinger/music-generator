import torch
import random
from secret_sauce.dataset.datasources import IDataSource
from secret_sauce.config.data.dataset import SongsDatasetConfig
from torch.utils.data import Dataset
import numpy as np


class SongsDataset(Dataset):
    def __init__(self, cfg: SongsDatasetConfig, datasource: IDataSource) -> None:
        super().__init__()
        self.datasource = datasource
        self.cfg = cfg

        self.durations, self.cumsum = datasource.get_total_duration()

    def preprocess_sample(self, wave: torch.Tensor) -> torch.Tensor:
        mix = random.uniform(0.4, 0.6)
        mono = mix * wave[0, ...] + (1 - mix) * wave[1, ...]
        return mono.unsqueeze(0)  # add back chan dim [T] -> [C, T]

    def __getitem__(self, item_idx: int):
        song_idx, sec_offset = self.get_index_offset(item_idx)
        wave = self.datasource.get_song(song_idx, sec_offset)
        mono = self.preprocess_sample(wave)
        return mono

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.cfg.sample_len))

    def get_index_offset(self, item) -> tuple[int, float]:
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.cfg.sample_len / 2
        shift = np.random.randint(-half_interval, half_interval)
        offset = (
            item * self.cfg.sample_len + shift
        )  # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert (
            0 <= midpoint < self.cumsum[-1]
        ), f"Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}"
        index = np.searchsorted(
            self.cumsum, midpoint
        )  # index <-> midpoint of interval lies in this song
        start, end = (
            self.cumsum[index - 1] if index > 0 else 0.0,
            self.cumsum[index],
        )  # start and end of current song
        assert (
            start <= midpoint <= end
        ), f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        if offset > end - self.cfg.sample_len:  # Going over song
            offset = max(start, offset - self.cfg.sample_len)  # Now should fit
        elif offset < start:  # Going under song
            offset = min(
                end - self.cfg.sample_len, offset + half_interval
            )  # Now should fit
        assert (
            start <= offset <= end - self.cfg.sample_len
        ), f"Offset {offset} not in [{start}, {end - self.cfg.sample_len}]. End: {end}, SL: {self.cfg.sample_len}, Index: {index}"
        offset = offset - start
        return index, offset