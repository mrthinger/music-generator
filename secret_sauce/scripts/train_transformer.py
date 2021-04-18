from secret_sauce.network.basic_transformer.transformer import BasicTransformer
from secret_sauce.dataset.basic_compressed_dataset import BasicCompressedDataset
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
from secret_sauce.util.util import print_master
from secret_sauce.config.config import Config

from omegaconf import OmegaConf
import deepspeed
import argparse
from torch.utils.tensorboard import SummaryWriter
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
    deepspeed.init_distributed()

    print_master(args)
    print_master(OmegaConf.to_yaml(cfg))

    ds = BasicCompressedDataset(
        "/root/secret_sauce/nggyu22000-compressed.pt", window_size=8192
    )

    transformer = BasicTransformer(cfg)
    t = torch.load(
        r"/root/secret_sauce/vaewts/04-16-2021-10-58-09/epoch-900/mp_rank_00_model_states.pt",
        map_location="cpu",
    )
    embed_wts: torch.tensor = t["module"]["vector_quantizer.embedding.weight"]
    transformer.load_embeddings(embed_wts)

    del t
    t = None

    print_master(f"num ds elems: {len(ds)}")
    print_master(f"num params: {sum(p.numel() for p in transformer.parameters())}")

    model, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=transformer,
        model_parameters=transformer.parameters(),
        training_data=ds,
    )

    model: DeepSpeedEngine = model
    training_dataloader: DeepSpeedDataLoader = training_dataloader

    if cfg.load_dir != None and cfg.load_tag != None:
        model.load_checkpoint(cfg.load_dir, tag=cfg.load_tag)

    if model.global_rank == 0:
        writer = SummaryWriter(cfg.save_dir)
        writer.add_scalar("Dataset Elements", len(ds))
        writer.add_scalar(
            "Parameters", sum(p.numel() for p in transformer.parameters())
        )

    for epoch in range(cfg.epochs):

        epoch_loss = 0

  
        for step, batch in enumerate(training_dataloader):
            batch: torch.Tensor = batch.to(model.local_rank, dtype=torch.long)

            src, src_pos, target = batch[:, 0, :], batch[:, 1, :], batch[:, 2, :]
            prediction, loss = model(src, src_pos, target)  # [B, T, F]

            epoch_loss += loss.item()
            model.backward(loss)
            model.step()
            lr_scheduler.step()

        epoch_loss /= len(training_dataloader)

        if model.global_rank == 0:
            writer.add_scalar("loss/train", epoch_loss, global_step=model.global_steps)

        if epoch % 100 == 0 and epoch != 0:
            model.save_checkpoint(cfg.save_dir, tag=f"epoch-{epoch}")


if __name__ == "__main__":
    main()