from secret_sauce.network.vqvae.vqvae import VQVAE
from secret_sauce.dataset.songs_datamodule import SongsDataModule
from secret_sauce.dataset.songs_dataset import SongsDataset
from secret_sauce.dataset.datasources import DiskDataSource
from secret_sauce.config.config import Config
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import Trainer


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    disk = DiskDataSource(cfg.dataset)

    ds = SongsDataset(cfg.dataset, disk)

    print(len(ds))

    datamod = SongsDataModule(cfg.dataset, ds)

    vqvae = VQVAE()
    # exdata: torch.Tensor = ds[0]
    # exdata = exdata.view(1, 1, -1)
    # encoded = vqvae(exdata)
    # print(encoded)

    trainer = Trainer(gpus=1)
    trainer.fit(vqvae, datamod)


if __name__ == "__main__":
    main()