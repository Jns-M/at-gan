"""Orchestrates GAN model construction, optimizer configuration, and the Keras training loop."""

import tensorflow as tf
import wandb
from keras import Model

from at_gan.data.preprocessor import TabularPreprocessor
from at_gan.models.discriminator import build_discriminator
from at_gan.models.generator import build_generator
from at_gan.models.gan import TabularGAN
from at_gan.callbacks.console_callback import ConsoleCallback
from at_gan.callbacks.gan_callback import GANCallback
from at_gan.callbacks.wand_callback import WandbCallback
from at_gan.utils.logger import get_logger
from at_gan.utils.paths import ExperimentPathManager

logger = get_logger(__name__)

class GANTrainer:
    """Builds, compiles, and trains a ``TabularGAN`` from a config dict and a preprocessed dataset."""

    def __init__(
            self,
            config: dict,
            preprocessor: TabularPreprocessor,
            dataset: tf.data.Dataset,
            eval_data: tf.Tensor,
            path_manager: ExperimentPathManager,
    ):
        """
        Initializes the GAN training module, sets up optimizers, learning rate schedules, the loss
        function, and evaluation callbacks required for training.

        Attributes:
            config: Parsed YAML configuration dictionary.
            preprocessor: Fitted ``TabularPreprocessor`` used to inverse-transform generated samples.
            dataset: Training dataset for the GAN.
            eval_data: Evaluation dataset for the GAN.
            path_manager: ``ExperimentPathManager`` providing checkpoint and log directory paths.
        """
        self.config = config
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.eval_data = eval_data
        self.start_epoch = 0

        train_cfg = config["training"]
        g_lr = train_cfg["g_lr"]
        d_lr = train_cfg["d_lr"]

        train_cfg = config["training"]
        g_lr = train_cfg["g_lr"]
        d_lr = train_cfg["d_lr"]
        adam_beta_1 = train_cfg.get("adam_beta_1", 0.9)
        cosine_decay = train_cfg.get("lr_cosine_decay", False)
        cosine_decay_restart_epochs = train_cfg.get("lr_cosine_decay_restart_epochs", 2000)
        g_decay_alpha = train_cfg.get("g_lr_decay_alpha", 0.01)
        d_decay_alpha = train_cfg.get("d_lr_decay_alpha", 0.01)

        if cosine_decay:
            try:
                steps_per_epoch = len(dataset)
            except TypeError:
                # Fallback for datasets without a known cardinality
                steps_per_epoch = 1000

            steps_per_restart = steps_per_epoch * cosine_decay_restart_epochs
            logger.info(f"Cosine Decay with Warm Restarts Active: Restarting every {cosine_decay_restart_epochs} epochs.")

            g_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=g_lr,
                first_decay_steps=steps_per_restart,
                # Keeps each restart interval identical in length
                t_mul=1.0,
                # Decays the peak LR by 10% on each restart
                m_mul=0.9,
                alpha=g_decay_alpha,
            )
            d_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=d_lr,
                first_decay_steps=steps_per_restart,
                t_mul=1.0,
                m_mul=0.9,
                alpha=d_decay_alpha,
            )

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=adam_beta_1, clipnorm=1.0)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=adam_beta_1, clipnorm=1.0)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.gan = self.build_models()
        self.gan.compile(
            d_optimizer=self.d_optimizer,
            g_optimizer=self.g_optimizer,
            loss_fn=self.loss_fn,
        )

        max_eval = train_cfg.get("test_split_pct", 2000)
        eval_batches = []
        collected = 0

        # prepare batches for evaluation
        for batch in dataset:
            eval_batches.append(batch)
            collected += batch.shape[0]
            if collected >= max_eval:
                break

        sample_batch = tf.concat(eval_batches, axis=0)

        self.callback = GANCallback(
            gan_model=self.gan,
            real_data_sample=self.eval_data,
            preprocessor=preprocessor,
            path_manager=path_manager,
            save_frequency=train_cfg.get("checkpoint_frequency", 100),
            eval_frequency=train_cfg.get("eval_frequency", 100),
        )

    def build_models(self) -> TabularGAN:
        """Instantiates the generator, discriminator, and assembled ``TabularGAN``.

        Returns:
            Uncompiled ``TabularGAN`` with both submodels built and summarized.
        """
        model_cfg = self.config["model"]
        train_cfg = self.config["training"]

        generator_cfg = model_cfg["generator"]
        discriminator_cfg = model_cfg["discriminator"]
        pack_size = discriminator_cfg.get("pack_size", 1)

        generator = build_generator(
            config=generator_cfg,
            latent_dim=model_cfg["latent_dim"],
            continuous_dim=self.preprocessor.continuous_dim,
            binary_dim=self.preprocessor.binary_dim,
            discrete_count_dim=self.preprocessor.discrete_count_dim,
            categorical_dims=self.preprocessor.categorical_dims,
        )
        generator.summary()

        # Discriminator input width is multiplied by pack_size to accommodate packed row inputs
        discriminator = build_discriminator(
            input_dim=self.preprocessor.total_dim * pack_size,
            config=discriminator_cfg,
        )
        discriminator.summary()

        gan = TabularGAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=model_cfg["latent_dim"],
            pack_size=pack_size,
            label_smoothing_min=discriminator_cfg.get("label_smoothing_min", 0.9),
            label_flipping=discriminator_cfg.get("label_flipping", 0.0),
            g_updates_per_epoch=train_cfg.get("g_updates_per_epoch", 1),
        )
        #gan.summary()
        return gan

    def restore_session(self) -> None:
        """Restores the latest checkpoint into the GAN and sets ``start_epoch`` accordingly."""
        if self.callback.latest_manager.latest_checkpoint:
            self.callback.checkpoint.restore(self.callback.latest_manager.latest_checkpoint)
            latest_ckpt_path = self.callback.latest_manager.latest_checkpoint
            # Checkpoint filenames encode the epoch number as the suffix after the final '-'
            self.start_epoch = int(latest_ckpt_path.split("-")[-1])
            logger.info(f"Successfully restored from previous session at epoch {self.start_epoch}.")

        else:
            logger.info("Initializing fresh training session.")

    def train(self) -> Model:
        """Runs the full Keras training loop and returns the trained GAN.

        Returns:
            The trained ``TabularGAN`` model after ``epochs`` iterations.
        """
        train_cfg = self.config["training"]

        if self.config.get("resume_run_id"):
            self.restore_session()

        callbacks: list[tf.keras.callbacks.Callback] = [ConsoleCallback(), self.callback]

        if wandb.run is not None:
            callbacks.append(WandbCallback())

        self.gan.fit(
            self.dataset,
            epochs=train_cfg["epochs"],
            initial_epoch=self.start_epoch,
            callbacks=callbacks,
            verbose=0,
        )

        return self.gan