import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


from secret_sauce.network.basic_transformer.transformer import BasicTransformer
from secret_sauce.dataset.basic_compressed_dataset import BasicCompressedDataset
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.engine import DeepSpeedEngine
from secret_sauce.util.util import parse_args, print_master
from secret_sauce.config.config import Config

from omegaconf import OmegaConf
import deepspeed
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch import nn


def main():
    cfg = Config()
    args = parse_args()
    deepspeed.init_distributed()

    print_master(args)
    print_master(OmegaConf.to_yaml(cfg))

    ds = BasicCompressedDataset(cfg)

    transformer = BasicTransformer(cfg)

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

        if epoch % cfg.save_every_epochs == 0 and epoch != 0:
            model.save_checkpoint(cfg.save_dir, tag=f"epoch-{epoch}")


if __name__ == "__main__":
    main()