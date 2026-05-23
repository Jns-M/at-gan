"""Inference-only synthesizer that loads a trained generator and produces synthetic tabular data."""

from pathlib import Path

import pandas as pd
import tensorflow as tf

from at_gan.data.preprocessor import TabularPreprocessor
from at_gan.utils.logger import get_logger

logger = get_logger(__name__)

class GANSynthesizer:
    """Wraps a saved Keras generator and a fitted ``TabularPreprocessor`` for post-training data synthesis."""

    def __init__(self, generator_path: str | Path, scaler_dir: str | Path, config: dict):
        """Loads the generator model and restores the preprocessor scalers from disk.

        Args:
            generator_path: Path to a saved ``.keras`` generator model file.
            scaler_dir: Directory containing serialized scaler and encoder ``.bin`` files.
            config: Full experiment config dict; uses ``model.latent_dim`` and ``data.*`` keys.
        """
        self.config = config
        self.latent_dim = self.config["model"]["latent_dim"]

        logger.info(f"Loading generator from {generator_path}...")
        self.generator = tf.keras.models.load_model(generator_path, compile=False)

        logger.info(f"Loading preprocessor scalers from {scaler_dir}...")

        # treat_bin_as_cat must match training config to reproduce identical column routing
        self.preprocessor = TabularPreprocessor(
            continuous_cols=self.config["data"].get("continuous_cols"),
            binary_cols=self.config["data"].get("binary_cols"),
            categorical_cols=self.config["data"].get("categorical_cols"),
            discrete_count_cols=self.config["data"].get("discrete_count_cols"),
            treat_bin_as_cat=self.config["data"].get("treat_bin_as_cat", False),
        )
        self.preprocessor.load_scalers(scaler_dir)

    def sample(self, num_samples: int) -> pd.DataFrame:
        """Generates a batch of synthetic samples by sampling from the latent space.

        Args:
            num_samples: Number of synthetic rows to generate.

        Returns:
            ``DataFrame`` of inverse-transformed synthetic data in original feature space.
        """
        logger.info(f"Generating {num_samples} synthetic samples...")

        noise = tf.random.normal(shape=(num_samples, self.latent_dim))

        synthetic_df = self.preprocessor.inverse_transform(
            self.generator(noise, training=False).numpy()
        )

        return synthetic_df