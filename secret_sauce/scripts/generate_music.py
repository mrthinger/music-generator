import torch

torch.manual_seed(0)
import random

random.seed(0)
import numpy as np

np.random.seed(0)

from omegaconf import OmegaConf
from secret_sauce.config.config import Config
from secret_sauce.dataset.datasources import DiskDataSource
from secret_sauce.dataset.songs_dataset import SongDataset
from secret_sauce.network.vqvae.vqvae import VQVAE
from secret_sauce.util.util import is_master, wait_for_debugger
import itertools

from performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
                                    
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    cfg = Config()
    print(OmegaConf.to_yaml(cfg))
    torch.set_autocast_enabled(True)


    wait_for_debugger()

    # vqvae = VQVAE(cfg)
    # vq_save = torch.load('/root/secret_sauce/weights/04/epoch-29/mp_rank_00_model_states.pt')['module']
    # vqvae.load_state_dict(vq_save)
    # vqvae.to(device=0, dtype=torch.float16)
    # y: torch.Tensor = vqvae(chunk, encode_only=True)


    transformer = PerformerLM(
        num_tokens=cfg.vqvae.num_embeddings + 1,  # +1 is for start token
        max_seq_len=cfg.transformer.window_size + cfg.transformer.shift,
        dim=cfg.transformer.width,
        depth=cfg.transformer.blocks_num,
        heads=cfg.transformer.heads_num,
        causal=True,
        use_scalenorm = True,
        reversible= True,
        emb_dropout=cfg.transformer.dropout,
    )
    transformer = AutoregressiveWrapper(transformer)
    transformer_save = torch.load('/root/secret_sauce/outputs/04-27-2021-07-20-40/epoch-3/mp_rank_00_model_states.pt')['module']
    transformer.load_state_dict(transformer_save)
    transformer.to(device=1, dtype=torch.float16)

    transformer.eval()

    start_token = torch.tensor(cfg.vqvae.num_embeddings)
    start_token = start_token.unsqueeze(0).to(device=1, dtype=torch.long)
    sample = transformer.generate(start_token, 4000*8, temperature=.98)

    torch.save(sample, './sample.pt')










if __name__ == "__main__":
    main()
