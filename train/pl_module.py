import torch
import torch.nn as nn
import torch.nn.functional as F
from crane import BrainFeatureExtractor, BrainModel
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from temporaldata import Data


class BFMLightning(LightningModule):
    """PyTorch Lightning module for training BrainModel on iEEG data. Defines training and validation steps.

    Args:
        cfg: Configuration dictionary for training.
        featurizer: BrainFeatureExtractor to preprocess raw iEEG data.
        model: BrainModel to be trained.
    """

    def __init__(self, cfg: DictConfig, featurizer: BrainFeatureExtractor, model: BrainModel):
        super().__init__()
        self.cfg = cfg
        self.featurizer = featurizer
        self.model = model
        self.head = nn.Linear(cfg.model.transformer.d_model, featurizer.feature_size)

    def take_step(self, batch: Data) -> torch.Tensor:
        inputs = self.featurizer(batch)  # [batch_size, num_electrodes, num_timebins, n_freqs]
        labels = inputs[:, :, 1:]  # [batch_size, num_electrodes, num_timebins - 1, n_freqs]

        features = self.model(inputs).last_hidden_state  # [batch_size, num_electrodes, num_timebins, d_model]
        preds = self.head(features)  # [batch_size, num_electrodes, num_timebins, in_channels]
        preds = preds[:, :, :-1, :]  # [batch_size, num_electrodes, num_timebins - 1, in_channels]

        loss = F.mse_loss(preds, labels)
        return loss

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        loss = self.take_step(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.n_samples)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        loss = self.take_step(batch)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch.n_samples)

    def configure_optimizers(self):
        optimizer = instantiate(
            self.cfg.optimizer,
            params=self.parameters(),
        )
        return optimizer
