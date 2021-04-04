import torch
from torch.nn import functional as F
import torchaudio
from torch import nn, optim


class Resnet1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=3):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, 3, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(out_channels, in_channels, 1, 1, 0),
        )

    def forward(self, x):
        return x + self.model(x)