from secret_sauce.util.util import print_master
from secret_sauce.network.vqvae.vector_quantizer import VectorQuantizer
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

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 4, 2, 1),
            self.make_res1d(),
            # nn.Conv1d(64, 64, 4, 2, 1),
            # self.make_res1d(),
            nn.Conv1d(64, 64, 4, 2, 1),
            self.make_res1d(),
            nn.Conv1d(64, 64, 4, 2, 1),
            self.make_res1d(),
        )

        self.vector_quantizer = VectorQuantizer(self.cfg.vqvae.num_embeddings, self.cfg.vqvae.embedding_dim)

        self.decoder = nn.Sequential(
            self.make_res1d(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1),
            self.make_res1d(),
            # nn.ConvTranspose1d(64, 64, 4, 2, 1),
            # self.make_res1d(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1),
            self.make_res1d(),
            nn.ConvTranspose1d(64, 64, 4, 2, 1),
            nn.Conv1d(64, 1, 3, 1, 1),
        )
        n_fft = 2048
        win_length = None
        hop_length = 128
        n_mels = 512
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.cfg.dataset.sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=False,
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            normalized=True,
        )
        # self.sequential_model = nn.Sequential(*list(self.encoder.modules()),*list(self.decoder.modules()))

    def make_res1d(self):
        return nn.Sequential(
            Resnet1dBlock(64, 64, dilation=3 * 1),
            Resnet1dBlock(64, 64, dilation=3 * 2),
            Resnet1dBlock(64, 64, dilation=3 * 3),
            Resnet1dBlock(64, 64, dilation=3 * 4),
            Resnet1dBlock(64, 64, dilation=3 * 5),
            Resnet1dBlock(64, 64, dilation=3 * 6),
            Resnet1dBlock(64, 64, dilation=3 * 7),
            Resnet1dBlock(64, 64, dilation=3 * 8),
            Resnet1dBlock(64, 64, dilation=3 * 9),
            Resnet1dBlock(64, 64, dilation=3 * 10),
            nn.BatchNorm1d(64),
        )

    def spec_loss(self, input: torch.Tensor, target: torch.Tensor):

        self.mel_spectrogram.type(torch.FloatTensor).to(input.device)

        input = input.type(torch.FloatTensor).to(input.device)
        target = target.type(torch.FloatTensor).to(target.device)

        input = self.mel_spectrogram(input)
        target = self.mel_spectrogram(target)

        loss = F.mse_loss(input, target)
        return loss

    def forward(self, x: torch.Tensor, encode_only: bool = False):
        y = x

        y = self.encoder(y)
        y, vqloss = self.vector_quantizer(y)

        if encode_only:
            return y

        y = self.decoder(y)


        # print_master(y.shape[2])
        spec_loss = self.spec_loss(y, x)
        # print(spec_loss)

        loss = F.mse_loss(y, x)
        loss += spec_loss
        loss += vqloss

        return y, loss

    def encode(self, x: torch.Tensor):
        y = x

        y = self.encoder(y)
        y, _ = self.vector_quantizer(y)

        return y

    def decode(self, x: torch.Tensor):
        y = x
        y = self.decoder(y)

        return y