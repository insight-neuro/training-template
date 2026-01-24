import torch
import torch.nn as nn
import torch.nn.functional as F
from crane import BrainModel
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from omegaconf import DictConfig


class BFMLightning(LightningModule):
    def __init__(self, cfg: DictConfig, model: BrainModel):
        super().__init__()
        self.cfg = cfg
        self.spectrogram = instantiate(cfg.spectrogram)
        self.model = model
        self.head = nn.Linear(cfg.model.transformer.d_model, cfg.model.input_dim)

    def take_step(self, batch: dict) -> torch.Tensor:
        inputs = self.spectrogram(batch)  # [batch_size, num_electrodes, num_timebins, n_freqs]
        labels = inputs[:, :, 1:]  # [batch_size, num_electrodes, num_timebins - 1, n_freqs]

        features = self.model(inputs)
        preds = self.head(features.last_hidden_state)  # [batch_size, num_electrodes, num_timebins, in_channels]
        preds = preds[:, :, :-1, :]  # Align predictions with labels

        loss = F.mse_loss(preds, labels)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.take_step(batch)
        batch_size = batch["ieeg"]["data"].shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self.take_step(batch)
        batch_size = batch["ieeg"]["data"].shape[0]
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = instantiate(
            self.cfg.optimizer,
            params=self.parameters(),
        )
        return optimizer
