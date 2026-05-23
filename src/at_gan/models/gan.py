"""Tabular GAN model implementing a custom packed training loop with one-sided label smoothing and flipping."""

import tensorflow as tf


class TabularGAN(tf.keras.Model):
    """Custom Keras Model implementing alternating discriminator/generator training for tabular data."""

    def __init__(
            self,
            generator: tf.keras.Model,
            discriminator: tf.keras.Model,
            latent_dim: int,
            pack_size: int = 1,
            label_smoothing_min: float = 0.9,
            label_flipping: float = 0.0,
            g_updates_per_epoch: int = 1,
            **kwargs,
    ):
        """Initializes the GAN with submodels, training regularization parameters, and metric trackers.

        Args:
            generator: Keras generator model mapping latent noise to synthetic tabular rows.
            discriminator: Keras discriminator model mapping (packed) rows to a real/fake score.
            latent_dim: Dimensionality of the generator's input noise vector.
            pack_size: Number of adjacent rows concatenated per discriminator input (PacGAN trick).
            label_smoothing_min: Lower bound of the uniform real-label smoothing range (``[min, 1.0]``).
            label_flipping: Fraction of real labels randomly flipped to fake per step (one-sided).
            g_updates_per_epoch: Number of generator gradient update passes per discriminator pass.
            **kwargs: Passed to ``tf.keras.Model.__init__``.
        """
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.pack_size = pack_size
        self.label_smoothing_min = label_smoothing_min
        self.label_flipping = label_flipping
        self.g_updates_per_epoch = g_updates_per_epoch

        self.g_optimizer = None
        self.d_optimizer = None
        self.loss_fn = None

        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_acc_tracker = tf.keras.metrics.BinaryAccuracy(name="d_acc")
        self.d_tnr_tracker = tf.keras.metrics.BinaryAccuracy(name="d_tnr")
        self.d_tpr_tracker = tf.keras.metrics.BinaryAccuracy(name="d_tpr")
        self.d_to_g_loss_ratio = tf.keras.metrics.Mean(name="d_to_g_loss_ratio")
        self.d_fake_pred_mean = tf.keras.metrics.Mean(name="d_fake_pred_mean")
        self.d_real_pred_mean = tf.keras.metrics.Mean(name="d_real_pred_mean")
        self.d_predicts_true_rate_tracker = tf.keras.metrics.Mean(name="d_predicts_true_rate")

    # noinspection PyMethodOverriding
    def compile(
            self,
            d_optimizer: tf.keras.optimizers.Optimizer,
            g_optimizer: tf.keras.optimizers.Optimizer,
            loss_fn,
    ):
        """Attaches optimizers and the loss function to the model.

        Args:
            d_optimizer: Optimizer for the discriminator.
            g_optimizer: Optimizer for the generator.
            loss_fn: Binary loss callable (e.g. ``BinaryCrossentropy``).
        """
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """Exposes all tracked metrics so Keras resets them automatically between epochs.

        Returns:
            List of all ``tf.keras.metrics.Metric`` instances tracked by this model.
        """
        return [
            self.d_loss_tracker,
            self.g_loss_tracker,
            self.d_acc_tracker,
            self.d_tnr_tracker,
            self.d_tpr_tracker,
            self.d_to_g_loss_ratio,
            self.d_fake_pred_mean,
            self.d_real_pred_mean,
            self.d_predicts_true_rate_tracker,
        ]

    @tf.function
    def train_step(self, real_data: tf.Tensor) -> dict[str, tf.Tensor]:
        """Executes one alternating D→G training step with packing, label smoothing, and label flipping.

        Args:
            real_data: Float tensor of shape ``(batch, feature_dim)`` from the training dataset.

        Returns:
            Dict of metric name to scalar tensor for the completed step.
        """
        raw_batch_size = tf.shape(real_data)[0]
        feature_dim = tf.shape(real_data)[1]

        # Floor-divide to produce complete packs; remainder rows are discarded
        packed_batch_size = raw_batch_size // self.pack_size
        valid_rows = packed_batch_size * self.pack_size
        real_data = real_data[:valid_rows]

        # Flatten pack_size adjacent rows into a single wide discriminator input
        real_data_packed = tf.reshape(real_data, [packed_batch_size, feature_dim * self.pack_size])

        # One-sided label smoothing: real targets are drawn from [label_smoothing_min, 1.0]
        real_labels = tf.random.uniform((packed_batch_size, 1), minval=self.label_smoothing_min, maxval=1.0)
        metric_real_labels = tf.ones((packed_batch_size, 1))
        fake_labels = tf.zeros((packed_batch_size, 1))

        # Randomly flip a fraction of real labels to zero to prevent discriminator overconfidence
        if self.label_flipping > 0.0:
            flip_real_mask = tf.random.uniform((packed_batch_size, 1)) < self.label_flipping
            real_labels = tf.where(flip_real_mask, 0.0, real_labels)

        # --- Discriminator update ---
        noise = tf.random.normal(shape=(valid_rows, self.latent_dim))
        fake_data = self.generator(noise, training=False)
        fake_data_packed = tf.reshape(fake_data, [packed_batch_size, feature_dim * self.pack_size])

        with tf.GradientTape() as d_tape:
            real_predictions = self.discriminator(real_data_packed, training=True)
            fake_predictions = self.discriminator(fake_data_packed, training=True)

            d_loss_real = self.loss_fn(real_labels, real_predictions)
            d_loss_fake = self.loss_fn(fake_labels, fake_predictions)
            # Average real and fake losses to balance gradient scale
            d_loss = (d_loss_real + d_loss_fake) / 2.0

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # --- Generator update (repeated g_updates_per_epoch times) ---
        for _ in range(max(1, self.g_updates_per_epoch)):
            fresh_noise = tf.random.normal(shape=(valid_rows, self.latent_dim))

            with tf.GradientTape() as g_tape:
                fresh_fake_data = self.generator(fresh_noise, training=True)
                fresh_fake_data_packed = tf.reshape(fresh_fake_data, [packed_batch_size, feature_dim * self.pack_size])
                fresh_fake_predictions = self.discriminator(fresh_fake_data_packed, training=False)
                g_loss = self.loss_fn(metric_real_labels, fresh_fake_predictions)

            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
            self.g_loss_tracker.update_state(g_loss)

        self.d_loss_tracker.update_state(d_loss)

        # Use the running g_loss average so the ratio accounts for multiple generator passes
        current_g_loss_avg = self.g_loss_tracker.result()
        ratio = d_loss / (current_g_loss_avg + tf.keras.backend.epsilon())
        self.d_to_g_loss_ratio.update_state(ratio)

        true_labels = tf.concat([tf.ones((packed_batch_size, 1)), tf.zeros((packed_batch_size, 1))], axis=0)
        all_predictions = tf.concat([real_predictions, fake_predictions], axis=0)
        predicted_true_flags = tf.cast(all_predictions > 0.5, tf.float32)

        self.d_predicts_true_rate_tracker.update_state(predicted_true_flags)
        self.d_acc_tracker.update_state(true_labels, all_predictions)
        self.d_tnr_tracker.update_state(fake_labels, fake_predictions)
        self.d_tpr_tracker.update_state(metric_real_labels, real_predictions)
        self.d_real_pred_mean.update_state(real_predictions)
        self.d_fake_pred_mean.update_state(fake_predictions)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
            "d/g_loss": self.d_to_g_loss_ratio.result(),
            "d_acc": self.d_acc_tracker.result(),
            "d_tnr": self.d_tnr_tracker.result(),
            "d_tpr": self.d_tpr_tracker.result(),
            "d_tricked": 1.0 - self.d_tnr_tracker.result(),
            "d_fake_pred": self.d_fake_pred_mean.result(),
            "d_real_pred": self.d_real_pred_mean.result(),
            "d_bias": self.d_predicts_true_rate_tracker.result(),
        }