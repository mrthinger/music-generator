from torch.utils.data import Dataset
import torch
import numpy as np
class BasicCompressedDataset(Dataset):

    def __init__(self, path: str, window_size: int = 32, shift: int = 1) -> None:
        super().__init__()
        self.data: torch.Tensor = torch.load(path, map_location='cpu')
        self.num_tokens = self.data.shape[0]
        self.window_size = window_size
        self.shift = shift

    
    def __getitem__(self, item_idx: int):
        offset = item_idx * self.window_size
        start = offset
        end = offset+self.window_size

        clip = self.data[start:end+self.shift]
        x = clip[:-self.shift]
        y = clip[self.shift:]
        x_pos = torch.arange(start, end)
        return torch.stack((x, x_pos, y))

    def __len__(self):
        return int(np.floor(self.num_tokens / self.window_size)) - 1
