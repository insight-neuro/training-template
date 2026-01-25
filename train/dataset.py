import logging
import os

import lightning.pytorch as pl
from crane.data import MultiSessionDataset, SessionBatchSampler, collate_crane_batches
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class iEEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for iEEG data.

    This module encapsulates all data loading logic including:
    - Loading training subject trials from config
    - Creating train/validation splits
    - Creating dataloaders with appropriate settings

    Args:
        cfg: Configuration object containing all settings
    """

    def __init__(self, cfg: DictConfig, root_dir: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.root_dir = root_dir or os.environ["ROOT_DIR"]
        if not root_dir and "ROOT_DIR" not in os.environ:
            raise ValueError("ROOT_DIR environment variable not set and no root_dir provided.")

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None):
        """
        Load data and create train/validation splits.
        This is called on every GPU/process.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Load training subject trials

        self.full_dataset = MultiSessionDataset(self.cfg.train_subject_trials, self.cfg.context_length, data_root_dir=self.root_dir)

        # Extract individual session sizes from the ConcatDataset
        self.session_sizes = [len(dataset) for dataset in self.full_dataset.datasets]

        # Note: Session-aware batching requires using the full dataset
        # Train/val split would need to be done at the session level to maintain homogeneous batches
        # For now, we use all data for both training and validation with session-aware batching
        # TODO: Implement session-level train/val split if needed
        logger.info(f"Total samples across {len(self.session_sizes)} sessions: {len(self.full_dataset)}")
        logger.info(f"Session sizes: {self.session_sizes}")

    def train_dataloader(self):
        """Create and return the training dataloader with session-aware batching."""
        # Create session-aware batch sampler
        batch_sampler = SessionBatchSampler(dataset_sizes=self.session_sizes, **self.cfg.sampler)

        return DataLoader(
            self.full_dataset,  # Use full_dataset, not train_dataset, because sampler handles indices
            batch_sampler=batch_sampler,
            **self.cfg.dataloader,
            collate_fn=collate_crane_batches,
        )

    def val_dataloader(self):
        """Create and return the validation dataloader with session-aware batching."""
        # Create session-aware batch sampler (no shuffling for validation)
        batch_sampler = SessionBatchSampler(dataset_sizes=self.session_sizes, **self.cfg.sampler)

        return DataLoader(
            self.full_dataset,  # Use full_dataset for now - train/val split needs session-level splitting
            batch_sampler=batch_sampler,
            **self.cfg.dataloader,
            collate_fn=collate_crane_batches,
        )
