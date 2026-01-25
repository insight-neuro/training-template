import torch
import torch.nn as nn
import torch.nn.functional as F
from crane import BrainModel
from crane.data import CraneBatch
from crane.preprocess import laplacian_rereference, subset_electrodes
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from omegaconf import DictConfig


class BFMLightning(LightningModule):
    def __init__(self, cfg: DictConfig, model: BrainModel):
        super().__init__()
        self.cfg = cfg
        self.spectrogram = instantiate(cfg.preprocess.spectrogram)
        self.model = model
        self.head = nn.Linear(cfg.model.transformer.d_model, cfg.model.input_dim)

    def take_step(self, batch: CraneBatch) -> torch.Tensor:
        data, channels = batch.data, batch.channels
        if self.cfg.rereference:
            data, channels = laplacian_rereference(data, channels)
        data, _ = subset_electrodes(data, channels, self.cfg.preprocess.max_n_electrodes)

        inputs = self.spectrogram(data, batch.sampling_rate)  # [batch_size, num_electrodes, num_timebins, n_freqs]
        labels = inputs[:, :, 1:]  # [batch_size, num_electrodes, num_timebins - 1, n_freqs]

        features = self.model(inputs).last_hidden_state  # [batch_size, num_electrodes, num_timebins, d_model]
        preds = self.head(features)  # [batch_size, num_electrodes, num_timebins, in_channels]
        preds = preds[:, :, :-1, :]  # [batch_size, num_electrodes, num_timebins - 1, in_channels]

        loss = F.mse_loss(preds, labels)
        return loss

    def training_step(self, batch: CraneBatch, batch_idx: int) -> torch.Tensor:
        loss = self.take_step(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.n_samples)
        return loss

    def validation_step(self, batch: CraneBatch, batch_idx: int) -> None:
        loss = self.take_step(batch)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch.n_samples)

    def configure_optimizers(self):
        optimizer = instantiate(
            self.cfg.optimizer,
            params=self.parameters(),
        )
        return optimizer
