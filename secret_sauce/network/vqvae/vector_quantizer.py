import deepspeed
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

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
                 beta: float = 0.98):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
        # for stage 3 (https://www.deepspeed.ai/tutorials/zero/)
        # deepspeed.zero.register_external_parameter(self,
        #                                            self.language_model.embedding.word_embeddings.weight)

    def get_inds(self, latents: torch.Tensor):
        latents = latents.permute(0, 2, 1).contiguous()  # [B x D x T] -> [B x T x D]
        latents_shape = latents.shape
        B, T, D = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BT x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BT x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BT, 1]

        return encoding_inds.view(B, T, 1).permute(0,2,1).contiguous()


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



#https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-3):
    return (x + eps) / (x.sum() + n_categories * eps)

class VectorQuantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.8, commitment=.02, eps=1e-3):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = torch.randn(dim, n_embed)
        embed.uniform_(-1 / self.n_embed, 1 / self.n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):

        input = input.permute(0, 2, 1).contiguous()


        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        distance = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-distance).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:

            if dist.is_initialized():
                size = float(dist.get_world_size())


            embed_onehot_sum = embed_onehot.type(torch.float32).sum(0)

            ema_inplace(self.cluster_size, embed_onehot_sum, self.decay)

            if dist.is_initialized():
                self.cluster_size.data /= size
                dist.all_reduce(self.cluster_size.data)

            embed_sum = flatten.transpose(0, 1).type(torch.float32) @ embed_onehot.type(torch.float32)
            ema_inplace(self.embed_avg, embed_sum, self.decay)

            if dist.is_initialized():
                self.embed_avg.data /= size
                dist.all_reduce(self.embed_avg.data)

            cs_f32 = self.cluster_size.type(torch.float32)

            cluster_size = laplace_smoothing(cs_f32, self.n_embed, self.eps) * cs_f32.sum()

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            if dist.is_initialized():
                self.embed.data /= size
                dist.all_reduce(self.embed.data)




        loss = F.mse_loss(quantize.detach(), input) * self.commitment
        quantize = input + (quantize - input).detach()

        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-5)))
        perplexity = embed_ind.unique().shape[-1]

        return quantize.permute(0, 2, 1).contiguous(), embed_ind, loss, perplexity