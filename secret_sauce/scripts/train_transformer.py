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
from secret_sauce.util.util import is_master, parse_args, print_master, wait_for_debugger_on_master
from secret_sauce.config.config import Config

from omegaconf import OmegaConf
import deepspeed
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper


def main():
    cfg = Config()
    args = parse_args()
    deepspeed.init_distributed()
    wait_for_debugger_on_master()

    print_master(args)
    print_master(OmegaConf.to_yaml(cfg))

    ds = BasicCompressedDataset(cfg)

    model = BasicTransformer(cfg)

    model = PerformerLM(
        num_tokens=cfg.vqvae.num_embeddings + 1,  # +1 is for start token
        max_seq_len=cfg.transformer.window_size + cfg.transformer.shift,
        dim=cfg.transformer.width,
        depth=cfg.transformer.blocks_num,
        heads=cfg.transformer.heads_num,
        causal=True,
        use_scalenorm = True,
    )
    model = AutoregressiveWrapper(model)

    print_master(f"num ds elems: {len(ds)}")
    print_master(f"num params: {sum(p.numel() for p in model.parameters())}")

    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=ds,
    )

    model_engine: DeepSpeedEngine = model_engine
    training_dataloader: DeepSpeedDataLoader = training_dataloader

    if cfg.load_dir != None and cfg.load_tag != None:
        model_engine.load_checkpoint(cfg.load_dir, tag=cfg.load_tag)

    if is_master():
        writer = SummaryWriter(cfg.save_dir)
        writer.add_scalar("Dataset Elements", len(ds))
        writer.add_scalar("Parameters", sum(p.numel() for p in model.parameters()))

    for epoch in range(cfg.epochs):

        epoch_loss = 0
        num_batches = 0

        for step, batch in enumerate(training_dataloader):
            model_engine.train()
            batch: torch.Tensor = batch.to(model_engine.local_rank, dtype=torch.long)

            loss = model_engine(batch)

            epoch_loss += loss.item()
            num_batches += 1
            model_engine.backward(loss)
            model_engine.step()
            lr_scheduler.step()

        epoch_loss /= num_batches

        if is_master:
            writer.add_scalar(
                "loss/train", epoch_loss, global_step=model_engine.global_steps
            )

        if epoch % cfg.save_every_epochs == 0:
            model_engine.save_checkpoint(cfg.save_dir, tag=f"epoch-{epoch}")


if __name__ == "__main__":
    main()