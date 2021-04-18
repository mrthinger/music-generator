import math
from secret_sauce.util.timer import Timer
from secret_sauce.config.config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self,
        x: TensorType["batch", "timestep", "feature"],
        position_inds: TensorType["batch", "timestep"],
    ):

        positional_encodings = self.pe[position_inds, :]

        x = x + positional_encodings
        return x


class BasicTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super(BasicTransformer, self).__init__()
        self.cfg = cfg
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(cfg.vqvae.embedding_dim, max_len=600000)
        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                cfg.vqvae.embedding_dim,
                cfg.transformer.heads_num,
                cfg.transformer.ff_dim,
                cfg.transformer.dropout,
            )
            for _ in range(cfg.transformer.blocks_num)
        ])

        self.codebook = nn.Embedding(cfg.vqvae.num_embeddings, cfg.vqvae.embedding_dim)
        self.predictions = nn.Linear(cfg.vqvae.embedding_dim, cfg.vqvae.num_embeddings)
        self.critereon = nn.CrossEntropyLoss()


    def generate_square_subsequent_mask(self, sz: int, device, dtype):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.to(dtype)
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def load_embeddings(self, embeddings: torch.Tensor):
        self.codebook.weight.data = embeddings

    def forward(
        self,
        src: TensorType["batch", "timestep"],
        src_positions: TensorType["batch", "timestep"],
        target: TensorType["batch", "timestep"],
    ):

        device = src.device
        src: torch.Tensor = src
        B, T = src.shape

        # dont have gradients flow into codebook
        with torch.no_grad():
            src = self.codebook(src) * math.sqrt(self.cfg.vqvae.embedding_dim)
            src = self.pos_encoder(src, src_positions)

            output = src.permute(1, 0, 2)
            src_mask = self.generate_square_subsequent_mask(T, device=device, dtype=output.dtype)

        for block in self.encoder_blocks:
            output = block(output, src_mask)


        output = output.permute(1, 0, 2)
        output = self.predictions(output)



        prediction = output.view(-1, self.cfg.vqvae.num_embeddings)
        target = target.reshape(-1)
        loss = self.critereon(prediction, target)


        return output, loss