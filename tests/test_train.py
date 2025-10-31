from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from train import train


def test_train_model():
    """Test the training pipeline with a single batch."""
    GlobalHydra.instance().clear()  # Clear Hydra's global state
    with initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = compose(
            config_name="config.yaml",
            overrides=[
                "+trainer.fast_dev_run=True",  # Runs a single batch for train, val, and test
                "datamodule.batch_size=2",
                "data_dir=./data",
                "original_work_dir=.",
            ],
        )
        train(cfg)
