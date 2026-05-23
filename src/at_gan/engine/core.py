"""Top-level orchestration engine for the full GAN experiment lifecycle."""

import random
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from sklearn.model_selection import train_test_split

from at_gan.data.preprocessor import TabularPreprocessor
from at_gan.training.trainer import GANTrainer
from at_gan.utils.logger import get_logger
from at_gan.utils.paths import ExperimentPathManager

logger = get_logger(__name__)

class GANCoreEngine:
    """Manages the end-to-end pipeline: config loading, hardware setup, data prep, training, and export."""

    def __init__(
            self,
            config_data: str | Path | dict,
            enable_wandb: bool = True,
            export_generator: bool = True,
            generate_samples: int = 0,
    ) -> None:
        """Loads config, seeds all RNGs, configures hardware, and initializes W&B.

        Args:
            config_data: Path to a YAML config file, or a pre-parsed config ``dict`` (e.g. from a sweep).
            enable_wandb: Whether to initialize and log to a W&B run.
            export_generator: Whether to save generator ``.keras`` files after training.
            generate_samples: Number of synthetic rows to auto-generate post-training. ``0`` disables.
        """
        self.config_data = config_data
        self.config = self.load_config()
        self.enable_wandb_flag = enable_wandb
        self.export_generator_flag = export_generator
        self.generate_samples = generate_samples

        self.experiment_name = self.config.get("experiment_name", "default_experiment")
        self.path_manager = None
        self.preprocessor = None
        self.dataset = None
        self.eval_data = None
        self.gan = None

        self.set_global_seed()
        self.setup_hardware()
        self.setup_wandb()

    def load_config(self) -> dict:
        """Loads the experiment configuration from a YAML file or returns a pre-parsed dict directly.

        Returns:
            Config dictionary with all experiment parameters.
        """
        # Return dict immediately when called from a sweep agent to avoid re-parsing
        if isinstance(self.config_data, dict):
            return self.config_data

        with open(self.config_data, "r") as file:
            return yaml.safe_load(file)

    def set_global_seed(self) -> None:
        """Seeds Python, NumPy, and TensorFlow RNGs for reproducibility.

        Args:
            None — reads ``seed`` from ``self.config`` (default ``1130``).
        """
        seed = self.config.get("seed", 1130)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Global random seed locked to: {seed}")

    def setup_hardware(self) -> bool:
        """Configures TensorFlow to use GPU or CPU based on the ``training.device`` config key.

        Returns:
            ``True`` if one or more GPUs were successfully configured, ``False`` otherwise.
        """
        target_device = self.config["training"].get("device", "cpu").lower()

        if target_device == "gpu":
            try:
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Detected GPU: {gpu.name}")
                    tf.config.optimizer.set_jit(False)
                    logger.info(f"GPU Mode Active: {len(gpus)} device(s) configured.")
                    return True
                else:
                    logger.warning("GPU requested but not found. Falling back to CPU.")
                    return False
            except Exception as e:
                logger.warning(f"GPU setup failed, falling back to CPU: {e}")
                return False

        logger.info("CPU Mode Active (requested via config).")
        # Explicitly hide GPUs to guarantee a pure CPU environment
        tf.config.set_visible_devices([], "GPU")
        return False

    def setup_wandb(self) -> None:
        """Initializes a W&B run (or attaches to an active sweep run) and creates experiment directories."""
        run_id = self.config.get("resume_run_id")
        base_output = self.config["data"]["output_path"]

        if not self.enable_wandb_flag:
            logger.info("WandB tracking disabled. Running locally.")
            self.path_manager = ExperimentPathManager(base_output, self.experiment_name, "offline_run")
            self.path_manager.create_dirs()
            return

        logger.info("Initializing Weights & Biases...")

        if wandb.run is None:
            wandb.init(
                project="at-gan-framework",
                group=self.experiment_name,
                name=None,
                id=str(run_id) if run_id else None,
                resume="allow" if run_id else None,
                config=self.config,
                job_type="training",
            )

            if run_id and wandb.run.id == run_id:
                logger.info(f"Resuming training from run ID: {run_id}...")
            elif run_id and wandb.run.id != run_id:
                logger.warning(f"Provided run ID {run_id} does not match current run ID {wandb.run.id}. Starting new run...")
        else:
            # Sweep agent already initialized the run; sync the full config into it
            wandb.config.update(self.config, allow_val_change=True)
            logger.info("Attached to active WandB Sweep run.")

        logger.info("Successfully set up Weights and Biases.")

        self.path_manager = ExperimentPathManager(base_output, self.experiment_name, wandb.run.id)
        self.path_manager.create_dirs()

    def prepare_data(self) -> None:
        """Reads the raw CSV, creates a holdout split, fits preprocessing on train data, and builds the training dataset."""
        raw_df = pd.read_csv(self.config["data"]["dataset_path"])

        train_cfg = self.config["training"]
        test_split_pct = train_cfg.get("test_split_pct", 0.2)

        if not 0 < test_split_pct < 1:
            raise ValueError(
                f"training.test_split_pct must be a float between 0 and 1, got {test_split_pct}."
            )

        seed = self.config.get("seed", 1130)
        train_df, eval_df = train_test_split(
            raw_df,
            test_size=test_split_pct,
            random_state=seed,
            shuffle=True,
        )

        self.preprocessor = TabularPreprocessor(
            continuous_cols=self.config["data"].get("continuous_cols"),
            binary_cols=self.config["data"].get("binary_cols"),
            categorical_cols=self.config["data"].get("categorical_cols"),
            discrete_count_cols=self.config["data"].get("discrete_count_cols"),
            treat_bin_as_cat=self.config["data"].get("treat_bin_as_cat", False),
            beta_noise=self.config["data"].get("beta_noise", False),
            smooth_categorical=self.config["data"].get("smooth_categorical", False),
        )

        preprocessed_train_df = self.preprocessor.fit_transform(train_df)
        preprocessed_eval_df = self.preprocessor.transform(eval_df)

        self.eval_data = tf.convert_to_tensor(
            preprocessed_eval_df.values.astype("float32")
        )

        self.preprocessor.dump_scalers(self.path_manager.scalers_dir)
        self.preprocessor.summarize_data(preprocessed_train_df)

        batch_size = self.config["training"]["batch_size"]
        self.dataset = tf.data.Dataset.from_tensor_slices(preprocessed_train_df.values.astype("float32"))

        # drop_remainder=True ensures every batch has an identical shape for the packed discriminator reshape
        self.dataset = (
            self.dataset
            .shuffle(buffer_size=len(preprocessed_train_df), seed=seed, reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    def export_generators(self) -> None:
        """Saves the latest in-memory generator and restores then saves the best checkpoint generator."""
        if self.gan is None or self.gan.generator is None:
            logger.error("Cannot export. The GAN has not been built or trained yet.")
            return

        self.gan.generator.save(self.path_manager.latest_generator_path)
        logger.info(f"Latest generator successfully saved to: {self.path_manager.latest_generator_path}")

        best_ckpt_path = tf.train.latest_checkpoint(str(self.path_manager.best_checkpoints_dir))

        if best_ckpt_path:
            # Checkpoint object must mirror the exact structure used in GANCallback to restore correctly
            checkpoint = tf.train.Checkpoint(
                g_optimizer=self.gan.g_optimizer,
                d_optimizer=self.gan.d_optimizer,
                generator=self.gan.generator,
                discriminator=self.gan.discriminator,
            )
            # Restore the best checkpoint weights
            checkpoint.restore(best_ckpt_path).expect_partial()

            self.gan.generator.save(self.path_manager.best_generator_path)
            logger.info(f"Best generator successfully saved to: {self.path_manager.best_generator_path}")
        else:
            logger.warning("No 'best' checkpoint found to export.")

    def train(self) -> None:
        """Instantiates a ``GANTrainer`` and runs the full Keras training loop."""
        trainer = GANTrainer(
            self.config,
            self.preprocessor,
            self.dataset,
            self.eval_data,
            self.path_manager,
        )
        self.gan = trainer.train()

    def auto_generate_data(self) -> None:
        """Generates synthetic samples from the active in-memory generator and saves them to CSV."""
        logger.info(f"Auto-generating {self.generate_samples} synthetic samples...")


        latent_dim = self.config["model"]["latent_dim"]
        noise = tf.random.normal(shape=(self.generate_samples, latent_dim))

        synthetic_df = self.preprocessor.inverse_transform(
            self.gan.generator(noise, training=False).numpy()
        )

        output_file = self.path_manager.synthetic_data_path
        synthetic_df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved auto-generated samples to: {output_file}")

    def run_experiment(self) -> None:
        """Executes the full experiment pipeline: data prep → train → export → generate → W&B finish."""
        self.prepare_data()
        self.train()
        logger.info("Experiment Pipeline Complete. GAN is ready for inference.")

        if self.export_generator_flag:
            self.export_generators()

        if self.generate_samples > 0:
            # uses the best generator weights loaded from checkpoint
            self.auto_generate_data()

        if self.enable_wandb_flag:
            wandb.finish()