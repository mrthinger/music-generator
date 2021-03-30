from secret_sauce.network.vqvae.vqvae import VQVAE
from secret_sauce.dataset.songs_datamodule import SongsDataModule
from secret_sauce.dataset.songs_dataset import SongsDataset
from secret_sauce.dataset.datasources import DiskDataSource
from secret_sauce.config.config import Config
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

deepspeed_config = {
    "zero_allow_untested_optimizer": True,
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 3e-5,
            "betas": [0.998, 0.999],
            "eps": 1e-5,
            "weight_decay": 1e-9,
            "cuda_aware": True,
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "last_batch_iteration": -1,
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 100,
        },
    },
    "zero_optimization": {
        "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
        "cpu_offload": True,  # Enable Offloading optimizer state/calculation to the host CPU
        "contiguous_gradients": True,  # Reduce gradient fragmentation.
        "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
        "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
    },
}


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