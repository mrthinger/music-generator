from secret_sauce.config.config import Config
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()