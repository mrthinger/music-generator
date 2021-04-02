from secret_sauce.network.vqvae.vqvae import VQVAE
from secret_sauce.dataset.songs_dataset import SongsDataset
from secret_sauce.dataset.datasources import DiskDataSource
from secret_sauce.config.config import Config
from omegaconf import OmegaConf
import deepspeed
import argparse
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser(description="VAE Train")

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    args = parser.parse_args()
    return args


def main():
    cfg = Config()
    args = parse_args()

    deepspeed.init_distributed(dist_backend="gloo")
    print(args)
    print(OmegaConf.to_yaml(cfg))

    disk = DiskDataSource(cfg.dataset)

    ds = SongsDataset(cfg.dataset, disk)

    print(len(ds))

    # datamod = SongsDataModule(cfg.dataset, ds)
    # vqvae = VQVAE()

    # parameters = filter(lambda p: p.requires_grad, vqvae.parameters())
    # exdata: torch.Tensor = ds[0]
    # exdata = exdata.view(1, 1, -1)
    # encoded = vqvae(exdata)
    # print(encoded)

    seq = nn.Sequential(nn.Linear(1, 1))

    model, optimizer, _, _ = deepspeed.initialize(args=args, model=seq)

    # for step, batch in enumerate(training_dataloader):
    #     loss = model(batch)
    #     model.backward(loss)
    #     model.step()
    #     lr_scheduler.step()


if __name__ == "__main__":

    main()