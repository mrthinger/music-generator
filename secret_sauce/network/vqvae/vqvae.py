import torch
from torch.nn import functional as F
import torchaudio
from torch import nn, optim


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 4, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 1, 1, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 512, 1, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 256, 4, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )

        # self.sequential_model = nn.Sequential(*list(self.encoder.modules()),*list(self.decoder.modules()))

    def forward(self, x):
        y = x
        y = self.encoder(y)
        y = self.decoder(y)

        loss = F.mse_loss(y, x)
        return y, loss

    # def training_step(self, batch, batch_idx):
    #     x = batch
    #     y_hat = self(x)
    #     loss = F.mse_loss(y_hat, x)

    #     self.log("train_loss", loss)

    #     return {"loss": loss, "preds": y_hat}

    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
    #     if batch_idx == 0 and self.current_epoch % 5 == 0:
    #         preds = outputs[0][0]["extra"]["preds"]
    #         song = preds[0].detach().cpu()
    #         tensorboard = self.logger.experiment
    #         tensorboard.add_audio(
    #             tag=str(self.current_epoch), snd_tensor=song, sample_rate=22000
    #         )

    # def configure_optimizers(self):
    #     opt = optim.Adam(self.parameters(), lr=0.001)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt)
    #     return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "train_loss"}
