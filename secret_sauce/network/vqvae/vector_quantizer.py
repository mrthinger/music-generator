import deepspeed
import torch
from torch import nn
from torch.nn import functional as F

# Source (Apache License 2.0): 
# https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/models/vq_vae.py#L7
# Adapted for sound data & deepspeed
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.02):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
        # for stage 3 (https://www.deepspeed.ai/tutorials/zero/)
        # deepspeed.zero.register_external_parameter(self,
        #                                            self.language_model.embedding.word_embeddings.weight)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1).contiguous()  # [B x D x T] -> [B x T x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BT x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BT x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BT, 1]

        # Convert to one-hot encodings
        device = latents.device
        dtype = latents.dtype
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device, dtype=dtype)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BT x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BT, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x T x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 2, 1).contiguous(), vq_loss  # [B x T x D] -> [B x D x T]