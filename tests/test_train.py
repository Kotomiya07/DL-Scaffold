from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

import glob
import os

from train import train


def test_train_model():
    """Test the training pipeline with a single batch."""
    GlobalHydra.instance().clear()  # Clear Hydra's global state
    with initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = compose(
            config_name="config.yaml",
            overrides=[
                "trainer.max_epochs=1",
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "+trainer.limit_test_batches=1",
                "trainer.default_root_dir=./test_outputs",
                "datamodule.batch_size=2",
                "data_dir=./data",
                "original_work_dir=.",
                # disable wandb logging for tests
                "logger=wandb",
                "+logger.mode=offline",
            ],
        )

        trainer = train(cfg)

        # lightweight verification
        assert trainer.checkpoint_callback.dirpath is not None
        output_dir = trainer.checkpoint_callback.dirpath
        assert os.path.exists(output_dir), (
            f"Output directory {output_dir} does not exist."
        )

        ckpt_files = glob.glob(os.path.join(output_dir, "**/*.ckpt"), recursive=True)
        assert len(ckpt_files) > 0, f"No checkpoint files found in {output_dir}."
