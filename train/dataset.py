from __future__ import annotations  # allow compatibility for Python 3.9

import logging
import os
from functools import partial

import lightning.pytorch as pl
import torch
from crane.data.multi_session import MultiSessionDataset
from crane.data.sampler import SessionBatchSampler
from crane.preprocess import laplacian_rereference, subset_electrodes
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def ieeg_collate_fn(batch: list, rereference: bool, max_n_electrodes: int) -> dict:
    """
    Custom collate function to handle mixed data types in iEEG dataset.
    Optionally applies preprocessing after batching if cfg is provided.

    Handles:
    - Tensors: stacked normally
    - Integers: converted to tensors
    - Strings: kept as lists
    - Numpy string arrays: converted to lists of numpy arrays (one per batch item)

    Args:
        batch: List of samples from dataset
    """
    if not batch:
        return {}

    # Get the structure from the first item
    first_item = batch[0]

    collated = {}

    # Handle "ieeg" data
    collated["ieeg"] = {
        "data": torch.stack([item["ieeg"]["data"] for item in batch]),
        "sampling_rate": first_item["ieeg"]["sampling_rate"],
    }

    # Handle "channels" - keep as list of arrays (can't stack string arrays)
    collated["channels"] = {"id": first_item["channels"]["id"]}

    # Handle "metadata" - keep strings as lists
    collated["metadata"] = {
        "brainset": first_item["metadata"]["brainset"],
        "subject": first_item["metadata"]["subject"],
        "session": first_item["metadata"]["session"],
    }

    if rereference:
        collated = laplacian_rereference(collated, inplace=True)
    collated = subset_electrodes(collated, max_n_electrodes, inplace=True)

    return collated


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
        self.root_dir = root_dir or os.environ["DATA_ROOT_DIR"]
        if not root_dir and "DATA_ROOT_DIR" not in os.environ:
            raise ValueError("DATA_ROOT_DIR environment variable not set and no root_dir provided.")

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
        self.session_sizes = [len(dataset) for dataset in self.full_dataset.datasets]  # type: ignore

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
            collate_fn=partial(
                ieeg_collate_fn,
                rereference=self.cfg.rereference,
                max_n_electrodes=self.cfg.max_n_electrodes,
            ),
        )

    def val_dataloader(self):
        """Create and return the validation dataloader with session-aware batching."""
        # Create session-aware batch sampler (no shuffling for validation)
        batch_sampler = SessionBatchSampler(dataset_sizes=self.session_sizes, **self.cfg.sampler)

        return DataLoader(
            self.full_dataset,  # Use full_dataset for now - train/val split needs session-level splitting
            batch_sampler=batch_sampler,
            **self.cfg.dataloader,
            collate_fn=partial(
                ieeg_collate_fn,
                rereference=self.cfg.rereference,
                max_n_electrodes=self.cfg.max_n_electrodes,
            ),
        )
