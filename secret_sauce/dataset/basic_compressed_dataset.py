from secret_sauce.config.config import Config
from torch.utils.data import Dataset
import torch
import numpy as np


class BasicCompressedDataset(Dataset):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.data: torch.Tensor = torch.load(
            cfg.transformer.data_path, map_location="cpu"
        )
        # data -> [ (Dim) ] * Songs

        self.durations = np.array([song.shape[-1] for song in self.data])
        self.cumsum = np.cumsum(self.durations)

        self.sample_size = self.cfg.transformer.window_size + self.cfg.transformer.shift

    def __getitem__(self, item_idx: int):
        song_idx, offset = self.get_index_offset(item_idx)
        start = offset
        end = offset + self.sample_size

        clip = self.data[song_idx][start : end]
        x = clip[: -self.cfg.transformer.shift]
        y = clip[self.cfg.transformer.shift :]
        x_pos = torch.arange(start, start+self.cfg.transformer.window_size)
        return torch.stack((x, x_pos, y))

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_size))

    def get_index_offset(self, item) -> tuple[int, int]:
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_size // 2
        shift = np.random.randint(-half_interval, half_interval)
        offset = (
            item * self.sample_size + shift
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
        if offset > end - self.sample_size:  # Going over song
            offset = max(start, offset - self.sample_size)  # Now should fit
        elif offset < start:  # Going under song
            offset = min(
                end - self.sample_size, offset + half_interval
            )  # Now should fit
        assert (
            start <= offset <= end - self.sample_size
        ), f"Offset {offset} not in [{start}, {end - self.sample_size}]. End: {end}, SL: {self.sample_size}, Index: {index}"
        offset = offset - start
        return index, int(offset)
