import hydra
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import wandb


def train(cfg: DictConfig) -> None:
    """Train the model using the provided configuration.

    Args:
        cfg: DictConfig object containing the configuration.
    """
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    logger: Logger = hydra.utils.instantiate(cfg.logger)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
