import torch
import torch.nn as nn
import torch.nn.functional as F
from crane import BrainModel
from crane.core.featurizer import BrainFeature
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from omegaconf import DictConfig


class BFMLightning(LightningModule):
    """PyTorch Lightning module for training BrainModel on iEEG data. Defines training and validation steps.

    Args:
        cfg: Configuration dictionary for training.
        model: BrainModel to be trained.
        output_dim: Dimensionality of the model output (e.g., number of frequency bins).
    """

    def __init__(self, cfg: DictConfig, model: BrainModel, output_dim: int):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.head = nn.Linear(cfg.model.transformer.d_model, output_dim)

    def take_step(self, batch: BrainFeature) -> torch.Tensor:
        inputs = batch.spec  # [batch_size, num_electrodes, num_timebins, n_freqs]
        labels = inputs[:, :, 1:]  # [batch_size, num_electrodes, num_timebins - 1, n_freqs]

        features = self.model(inputs).last_hidden_state  # [batch_size, num_electrodes, num_timebins, d_model]
        preds = self.head(features)  # [batch_size, num_electrodes, num_timebins, output_dim]
        preds = preds[:, :, :-1, :]  # [batch_size, num_electrodes, num_timebins - 1, output_dim]

        loss = F.mse_loss(preds, labels)
        return loss

    def training_step(self, batch: BrainFeature, batch_idx: int) -> torch.Tensor:
        loss = self.take_step(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.spec.shape[0])  # type: ignore[attr-defined]
        return loss

    def validation_step(self, batch: BrainFeature, batch_idx: int) -> None:
        loss = self.take_step(batch)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch.spec.shape[0])  # type: ignore[attr-defined]

    def configure_optimizers(self):
        optimizer = instantiate(
            self.cfg.optim,
            params=self.parameters(),
        )
        return optimizer
