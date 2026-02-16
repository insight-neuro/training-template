import logging

import lightning.pytorch as pl
from crane.data import Subjects
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import DistributedStitchingFixedWindowSampler, RandomFixedWindowSampler

logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for iEEG data. Handles loading and batching of data for training and validation.

    Args:
        cfg: Configuration object containing all settings.
        root_dir: Optional root directory for data. If not provided, will use ROOT_DIR environment variable.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        eval_subjects = Subjects(*cfg.eval_subjects)
        train_subjects = ~eval_subjects  # Use all subjects not in eval_subjects for training

        self.train_ds = instantiate(cfg.train_ds, select=train_subjects)
        self.eval_ds = instantiate(cfg.eval_ds, select=eval_subjects)

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
            step=self.cfg.eval_step_size,
            batch_size=self.cfg.dataloader.batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        val_loader = DataLoader(self.eval_ds, sampler=val_sampler, collate_fn=collate, **self.cfg.dataloader, shuffle=False, drop_last=False)

        logger.info("Created validation dataloader with %d samples.", len(val_loader))
        return val_loader
