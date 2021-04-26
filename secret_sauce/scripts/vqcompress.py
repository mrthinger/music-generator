import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.engine import DeepSpeedEngine
from secret_sauce.network.vqvae.vqvae import VQVAE
from secret_sauce.dataset.songs_dataset import SongClipDataset
from secret_sauce.dataset.datasources import DiskDataSource
from secret_sauce.util.util import is_master, print_master, wait_for_debugger_on_master
from secret_sauce.config.config import Config
import torch.distributed as dist
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
import deepspeed
import argparse

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


    #configure params
    cfg.load_dir = './weights/04'
    cfg.load_tag = 'epoch-29'

    deepspeed.init_distributed()

    print_master(args)
    print_master(OmegaConf.to_yaml(cfg))

    disk = DiskDataSource(cfg.dataset)

    ds = SongClipDataset(cfg.dataset, disk)

    vqvae = VQVAE(cfg)

    print_master(f"num ds elems: {len(ds)}")
    print_master(f"num params: {sum(p.numel() for p in vqvae.parameters())}")

    model, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args, model=vqvae, model_parameters=vqvae.parameters(), training_data=ds
    )


    model: DeepSpeedEngine = model
    training_dataloader: DeepSpeedDataLoader = training_dataloader

    if cfg.load_dir != None and cfg.load_tag != None:
        model.load_checkpoint(cfg.load_dir, tag=cfg.load_tag, load_optimizer_states=False, load_lr_scheduler_states=False)


    if is_master():
        data = []

    for step, batch in enumerate(training_dataloader):
        if model.fp16_enabled:
            batch = batch.type(torch.HalfTensor)
        batch: torch.Tensor = batch.to(model.local_rank)


        tensor_list = [torch.zeros((16,20000), dtype=torch.float32).to(model.local_rank) for _ in range(model.world_size)]

        with torch.no_grad():
        
            y: torch.Tensor = model(batch, encode_only=True)
            y = y.detach().type(torch.float32)

        print_master((y.shape, tensor_list[0].shape, len(tensor_list)))

        dist.all_gather(tensor_list, y)

        if is_master():
            data.append(tensor_list)
    


    if is_master():
        wait_for_debugger_on_master()

        import itertools
        data = list(itertools.chain(*data))
        data = torch.stack(data).view(-1).type(torch.int16)
        print(data)
        torch.save(data, "./savant32000-compressed.pt")




if __name__ == "__main__":
    main()