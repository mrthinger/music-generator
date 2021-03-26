from secret_sauce.dataset.songs_dataset import SongsDataset
from secret_sauce.dataset.datasources import DiskDataSource
from secret_sauce.config.config import Config
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    disk = DiskDataSource(cfg.dataset)

    ds = SongsDataset(cfg.dataset, disk)
    for i in range(len(ds)):
        print(ds.get_index_offset(i))

    o = ds[1]
    print(o)


if __name__ == "__main__":
    main()