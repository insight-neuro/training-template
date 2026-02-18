import logging

import lightning.pytorch as pl
from crane import BrainFeatureExtractor
from crane.data import CraneDataset, Subjects
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import DistributedStitchingFixedWindowSampler, RandomFixedWindowSampler

logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for iEEG data. Handles loading and batching of data for training and validation.

    Args:
        cfg: Configuration object containing all settings.
        featurizer: BrainFeatureExtractor to preprocess raw iEEG data.
    """

    def __init__(self, cfg: DictConfig, featurizer: BrainFeatureExtractor):
        super().__init__()
        self.cfg = cfg

        eval_subjects = Subjects(*cfg.eval_subjects)
        train_subjects = ~eval_subjects  # Use all subjects not in eval_subjects for training

        self.train_ds = CraneDataset(cfg.dataset_dir, featurizer=featurizer, select=train_subjects)
        self.eval_ds = CraneDataset(cfg.dataset_dir, featurizer=featurizer, select=eval_subjects)

    def train_dataloader(self):
        """Returns the training dataloader."""
        train_sampler = RandomFixedWindowSampler(sampling_intervals=self.train_ds.get_sampling_intervals(), window_length=self.cfg.context_length)

        train_loader = DataLoader(self.train_ds, sampler=train_sampler, collate_fn=collate, **self.cfg.dataloader)

        logger.info("Created training dataloader with %d samples.", len(train_loader))
        return train_loader

    def val_dataloader(self):
        """Returns the validation dataloader."""
        assert self.trainer is not None
        val_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.eval_ds.get_sampling_intervals(),
            window_length=self.cfg.context_length,
            batch_size=self.cfg.dataloader.batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        val_loader = DataLoader(self.eval_ds, sampler=val_sampler, collate_fn=collate, **self.cfg.dataloader, shuffle=False, drop_last=False)

        logger.info("Created validation dataloader with %d samples.", len(val_loader))
        return val_loader
