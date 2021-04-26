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

                                    
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    cfg = Config()
    print(OmegaConf.to_yaml(cfg))

    disk = DiskDataSource(cfg.dataset)

    ds = SongDataset(disk)
    songs_dataloader = DataLoader(ds)

    # wait_for_debugger()

    vqvae = VQVAE(cfg)

    print(f"num ds elems: {len(ds)}")


    model_save = torch.load('/root/secret_sauce/weights/04/epoch-29/mp_rank_00_model_states.pt')['module']
    vqvae.load_state_dict(model_save)
    vqvae.to(device='cuda', dtype=torch.float16)


    processed_songs = []

    for batch in tqdm(songs_dataloader):
        song = batch.to(device='cuda', dtype=torch.float16)

        chunks = []
        B, C, D = song.shape
        chunk_size = 8000000
        while D > chunk_size:
            chunk = song[..., :chunk_size]
            song = song[..., chunk_size:]
            B, C, D = song.shape
            chunks.append(chunk)
        chunks.append(song)


        processed_chunks = []

        for chunk in chunks:
            with torch.no_grad():
                y: torch.Tensor = vqvae(chunk, encode_only=True)
                y = y.detach().type(torch.int16)
                processed_chunks.append(y)

        processed_song = torch.cat(processed_chunks, dim=-1).view(-1)
        processed_songs.append(processed_song)


    # processed songs -> [ (D) ] * S
    torch.save(processed_songs, './savant-32000-compressed.pt')






if __name__ == "__main__":
    main()
