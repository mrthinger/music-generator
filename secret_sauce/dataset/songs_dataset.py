from torch.tensor import Tensor
from secret_sauce.dataset.datasources import IDataSource
from secret_sauce.config.data.dataset import SongsDatasetConfig
from torch.utils.data import Dataset


class SongsDataset(Dataset):
    def __init__(self, cfg: SongsDatasetConfig, datasource: IDataSource) -> None:
        super().__init__()
        self.datasource = datasource
        self.cfg = cfg

    def __getitem__(self, idx: int):
        path = self.datasource.get_song_uri(idx)

    def __len__(self):
        return len(self.datasource.get_num_songs())