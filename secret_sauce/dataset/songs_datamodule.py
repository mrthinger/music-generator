from torch.utils.data.dataloader import DataLoader
from secret_sauce.dataset.songs_dataset import SongsDataset
import pytorch_lightning as pl
from secret_sauce.config.data.dataset import SongsDatasetConfig
from torch.utils.data import Dataset


class OffsetDataset(Dataset):
    def __init__(self, dataset: Dataset, start: int, end: int):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, item):
        return self.dataset[self.start + item]


class SongsDataModule(pl.LightningDataModule):
    def __init__(self, cfg: SongsDatasetConfig, dataset: SongsDataset):
        super().__init__()
        self.cfg = cfg

        train_start = 0
        train_end = int(len(dataset) * cfg.train_test_split)

        test_start = train_end
        test_end = len(dataset)

        self.train_ds = OffsetDataset(dataset, train_start, train_end)
        self.test_ds = OffsetDataset(dataset, test_start, test_end)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.cfg.batch_size)