from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.engine import DeepSpeedEngine
from secret_sauce.network.vqvae.vqvae import VQVAE
from secret_sauce.dataset.songs_dataset import SongsDataset
from secret_sauce.dataset.datasources import DiskDataSource
from secret_sauce.config.config import Config
from omegaconf import OmegaConf
import deepspeed
import argparse
from torch.utils.tensorboard import SummaryWriter


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

    print(args)
    print(OmegaConf.to_yaml(cfg))

    disk = DiskDataSource(cfg.dataset)

    ds = SongsDataset(cfg.dataset, disk)

    print(len(ds))

    vqvae = VQVAE()
    parameters = filter(lambda p: p.requires_grad, vqvae.parameters())

    model, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args, model=vqvae, model_parameters=parameters, training_data=ds
    )

    model: DeepSpeedEngine = model
    training_dataloader: DeepSpeedDataLoader = training_dataloader

    print(optimizer)

    writer = SummaryWriter(cfg.save_dir)

    for epoch in range(cfg.epochs):

        epoch_loss = 0

        for step, batch in enumerate(training_dataloader):
            batch = batch.to(model.local_rank)
            y, loss = model(batch)
            epoch_loss += loss.item()
            model.backward(loss)
            model.step()
            lr_scheduler.step()

        epoch_loss /= len(training_dataloader)
        writer.add_scalar("loss/train", epoch_loss, global_step=model.global_steps)

        if epoch % 10 == 0:
            writer.add_audio(
                "Reconstruction",
                y[0].detach().cpu(),
                sample_rate=22000,
                global_step=model.global_steps,
            )


if __name__ == "__main__":

    main()