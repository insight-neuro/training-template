import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from train.dataset import iEEGDataModule
from train.model import iEEGTransformer
from train.pl_module import BFMLightning

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    hydra_wd = HydraConfig.get().runtime.output_dir

    # Set random seed for reproducibility
    cfg.seed = cfg.get("seed", 42)
    seed_everything(cfg.seed)

    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = iEEGDataModule(cfg)

    # Initialize model
    logger.info("Initializing model...")
    model = iEEGTransformer(cfg.model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")

    # Set up callbacks
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=f"{hydra_wd}/checkpoints",
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=cfg.save_model_every_n_epochs,
        )
    ]

    # CSV Logger (always active) for local logging
    loggers: list[Logger] = [CSVLogger(save_dir=f"{hydra_wd}/logs", name="training_logs")]
    logger.info(f"CSV logging enabled: {hydra_wd}/logs/training_logs")

    # WandB Logger (optional)
    if cfg.wandb_project and cfg.wandb_entity:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.get("run_name"),
            save_dir=hydra_wd,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)
        logger.info(f"W&B logging enabled: {cfg.wandb_entity}/{cfg.wandb_project}")
    else:
        callbacks += [DeviceStatsMonitor(), LearningRateMonitor(logging_interval="step")]
        logger.info("W&B logging disabled: (wandb_project or wandb_entity are empty)")

    # Initialize Lightning module
    module = BFMLightning(cfg, model)
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        **cfg.trainer,
    )

    # Train
    logger.info("=" * 80 + "\nStarting training...\n" + "=" * 80)
    trainer.fit(module, datamodule=datamodule)
    logger.info("=" * 80 + "\nTraining complete!\n" + "=" * 80)
