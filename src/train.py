import hydra
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import logging

import wandb

log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> Trainer:
    """Train the model using the provided configuration.

    Args:
        cfg: DictConfig object containing the configuration.

    Returns:
        The Lightning Trainer object.
    """
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    logger: Logger = hydra.utils.instantiate(cfg.logger)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    try:
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
    except Exception as e:
        log.error(f"An error occurred during training: {e}")
        raise
    finally:
        # Properly close wandb run
        if wandb.run:
            wandb.finish()

    return trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
