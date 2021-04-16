import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.engine import DeepSpeedEngine
from secret_sauce.network.vqvae.vqvae import VQVAE
from secret_sauce.dataset.songs_dataset import SongsDataset
from secret_sauce.dataset.datasources import DiskDataSource
from secret_sauce.config.config import Config
from omegaconf import OmegaConf
import deepspeed
import argparse
from torch.utils.tensorboard import SummaryWriter

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
    deepspeed.init_distributed()

    print(torch.distributed.get_rank())

    print(args)
    print(OmegaConf.to_yaml(cfg))

    disk = DiskDataSource(cfg.dataset)

    ds = SongsDataset(cfg.dataset, disk)

    vqvae = VQVAE(cfg)

    print(f"num ds elems: {len(ds)}")
    print(f"num params: {sum(p.numel() for p in vqvae.parameters())}")

    model, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args, model=vqvae, model_parameters=vqvae.parameters(), training_data=ds
    )

    model: DeepSpeedEngine = model
    training_dataloader: DeepSpeedDataLoader = training_dataloader

    # model.load_checkpoint("./outputs/04-04-2021-16-02-28", tag="epoch-300")

    if model.global_rank == 0:
        writer = SummaryWriter(cfg.save_dir)
        writer.add_scalar("Dataset Elements", len(ds))
        writer.add_scalar("Parameters", sum(p.numel() for p in vqvae.parameters()))

    for epoch in range(cfg.epochs):

        epoch_loss = 0

        for step, batch in enumerate(training_dataloader):
            if model.fp16_enabled:
                batch = batch.type(torch.HalfTensor)
            batch: torch.Tensor = batch.to(model.local_rank)
            y, loss = model(batch)
            epoch_loss += loss.item()
            model.backward(loss)
            model.step()
            lr_scheduler.step()

        epoch_loss /= len(training_dataloader)

        if model.global_rank == 0:
            writer.add_scalar("loss/train", epoch_loss, global_step=model.global_steps)

            if epoch % 10 == 0:
                song: torch.Tensor = (
                    y[0].detach().cpu().type(torch.FloatTensor).clip(-1, 1)
                )
                writer.add_audio(
                    "Reconstruction",
                    song,
                    sample_rate=22000,
                    global_step=model.global_steps,
                )

        if epoch % 100 == 0 and epoch != 0:
            model.save_checkpoint(cfg.save_dir, tag=f"epoch-{epoch}")


if __name__ == "__main__":

    main()