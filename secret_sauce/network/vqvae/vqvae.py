from secret_sauce.util.util import print_master
from secret_sauce.network.vqvae.vector_quantizer import VectorQuantize, VectorQuantizer
from secret_sauce.network.vqvae.resnet import Resnet1dBlock
import torch
from torch.nn import functional as F
import torchaudio
from torch import nn, optim
from secret_sauce.config.config import Config
from torchaudio import transforms as T


class VQVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.res_width = 64

        self.encoder = nn.Sequential(
            nn.Conv1d(1, self.res_width, 4, 2, 1),
            self.make_res1d(),

            # nn.Conv1d(self.res_width, self.res_width, 4, 2, 1),
            # self.make_res1d(),
            nn.Conv1d(self.res_width, self.res_width, 4, 2, 1),
            self.make_res1d(),

            nn.Conv1d(self.res_width, self.res_width, 4, 2, 1),
            self.make_res1d(),

        )

        self.vector_quantizer = VectorQuantize( self.cfg.vqvae.embedding_dim, self.cfg.vqvae.num_embeddings)

        self.decoder = nn.Sequential(
            self.make_res1d(),
            nn.ConvTranspose1d(self.res_width, self.res_width, 4, 2, 1),
            self.make_res1d(),

            # nn.ConvTranspose1d(self.res_width, self.res_width, 4, 2, 1),
            # self.make_res1d(),
            nn.ConvTranspose1d(self.res_width, self.res_width, 4, 2, 1),
            self.make_res1d(),

            nn.ConvTranspose1d(self.res_width, self.res_width, 4, 2, 1),
            nn.Conv1d(self.res_width, 1, 3, 1, 1),
        )
        # n_fft = (2048, 1024, 512)
        # win_length = (1200, 600, 240)
        # hop_length = (240, 120, 50)
        # n_mels = (128, 128, 64)  
        # 
        n_fft = (2048,)
        win_length = (240,)
        hop_length = (50,)
        n_mels = (128,)

        self.specs: list[T.MelSpectrogram] = []
        for i in range(len(n_fft)):
            spec = self.make_mel_spec(n_fft[i], win_length[i], hop_length[i], n_mels[i])
            self.specs.append(spec)


    def make_mel_spec(self, n_fft, win_length, hop_length, n_mels):
        return T.Spectrogram(
            # sample_rate=self.cfg.dataset.sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            # center=False,
            power=2.0,
            # norm="slaney",
            # onesided=True,
            # n_mels=n_mels,
            normalized=True,
        )

    def make_res1d(self):
        return nn.Sequential(
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 1),
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 2),
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 3),
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 4),
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 5),
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 6),
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 7),
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 8),
            Resnet1dBlock(self.res_width, self.res_width, dilation=3 * 9),
            nn.BatchNorm1d(self.res_width),
        )

    def spec_loss(self, input: torch.Tensor, target: torch.Tensor):
        
        # loss = torch.tensor(0, dtype=torch.float, device=input.device)
        input = input.to(input.device, dtype=torch.float)
        target = target.to(target.device, dtype=torch.float)

        spec = self.specs[0]
        # for spec in self.specs:

        spec.to(input.device, dtype=torch.float)


        spec_input = spec(input)
        spec_target = spec(target)

        loss = F.mse_loss(spec_input, spec_target) / len(self.specs)
        return loss

    def forward(self, x: torch.Tensor, encode_only: bool = False):
        y = x

        y = self.encoder(y)

        



        y, embed_ind, vqloss, usage_perplexity  = self.vector_quantizer(y)

        if encode_only:
            return embed_ind

        y = self.decoder(y)


        spec_loss = self.spec_loss(y, x)
        # print(spec_loss)

        loss = F.mse_loss(y, x)
        loss += spec_loss
        loss += vqloss * .1

        return y, vqloss, usage_perplexity

    def encode(self, x: torch.Tensor):
        y = x

        y = self.encoder(y)
        y, _ = self.vector_quantizer(y)

        return y

    def decode(self, x: torch.Tensor):
        y = x
        y = self.decoder(y)

        return y