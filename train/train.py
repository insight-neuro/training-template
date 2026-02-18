import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from train.datamodule import DataModule
from train.model import iEEGTransformer
from train.pl_module import BFMLightning

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    hydra_wd = HydraConfig.get().runtime.output_dir

    # Set random seed for reproducibility
    cfg.seed = cfg.get("seed", 42)
    seed_everything(cfg.seed)

    # Initialize featurizer
    logger.info("Setting up featurizer...")
    featurizer = instantiate(cfg.preprocess)

    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = DataModule(cfg.data, featurizer)

    # Initialize model
    logger.info("Initializing model...")
    model = iEEGTransformer(cfg.model, input_dim=featurizer.feature_size)

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
    if cfg.wandb and cfg.wandb.project and cfg.wandb.entity:
        wandb_logger = WandbLogger(
            save_dir=hydra_wd,
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.wandb,
        )
        loggers.append(wandb_logger)

        run = wandb_logger.experiment
        logger.info(f"W&B logging enabled: {run.name} at {run.entity}/{run.project} (id={run.id})")
    else:
        callbacks += [DeviceStatsMonitor(), LearningRateMonitor(logging_interval="step")]
        logger.info("W&B logging disabled: (wandb not set in config.)")

    # Initialize Lightning module
    module = BFMLightning(cfg, model, output_dim=featurizer.feature_size)
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        **cfg.trainer,
    )

    # Train
    logger.info("=" * 80 + "\nStarting training...\n" + "=" * 80)
    trainer.fit(module, datamodule=datamodule)
    logger.info("=" * 80 + "\nTraining complete!\n" + "=" * 80)
