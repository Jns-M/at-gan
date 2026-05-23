"""Keras callback for forwarding per-epoch GAN metrics to Weights & Biases."""

import tensorflow as tf
import wandb


class WandbCallback(tf.keras.callbacks.Callback):
    """Logs GAN training metrics to a W&B run at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        """Packages and ships per-epoch metrics to W&B.

        Args:
            epoch: Integer index of the completed epoch (0-based).
            logs: Dict of metric names to scalar values produced by the GAN's
                ``train_step``. Expected keys: ``d_loss``, ``g_loss``,
                ``d/g_loss``, ``d_acc``, ``d_tnr``, ``d_tpr``, ``d_tricked``,
                ``d_fake_pred``, ``d_real_pred``, ``d_bias``.
                Missing keys default to ``0.0``.
        """
        if logs is None:
            return

        wandb_logs = {
            "Loss/Discriminator":        logs.get("d_loss", 0.0),
            "Loss/Generator":            logs.get("g_loss", 0.0),
            "Loss/D:G-Ratio":            logs.get("d/g_loss", 0.0),
            "Metrics/D-ACC":             logs.get("d_acc", 0.0),
            "Metrics/D-TNR":             logs.get("d_tnr", 0.0),
            "Metrics/D-TPR":             logs.get("d_tpr", 0.0),
            "Metrics/D-Trick-Rate":      logs.get("d_tricked", 0.0),
            "Other/D-Fake-Pred":         logs.get("d_fake_pred", 0.0),
            "Other/D-Real-Pred":         logs.get("d_real_pred", 0.0),
            "Other/D-Positive-Rate":     logs.get("d_bias", 0.0),
        }

        wandb.log(wandb_logs, step=epoch)