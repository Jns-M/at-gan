"""Keras callback for structured GAN training progress output to stdout."""

import time

import tensorflow as tf


class ConsoleCallback(tf.keras.callbacks.Callback):
    """Prints a formatted per-epoch summary of GAN losses and discriminator metrics to stdout."""

    def __init__(self):
        """Initializes the callback with a null epoch timer."""
        super().__init__()
        self.epoch_time_start = None

    def on_epoch_begin(self, epoch, logs=None):
        """Records the wall-clock start time of the epoch.

        Args:
            epoch: Integer index of the current epoch (0-based).
            logs: Unused. Passed by the Keras training loop.
        """
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Prints the formatted training summary for the completed epoch.

        Args:
            epoch: Integer index of the completed epoch (0-based).
            logs: Dict of metric names to scalar values produced by the GAN's
                ``train_step``. Expected keys: ``d/g_loss``, ``d_loss``,
                ``g_loss``, ``d_acc``, ``d_tnr``, ``d_tpr``, ``d_tricked``,
                ``d_fake_pred``, ``d_real_pred``, ``d_bias``.
                Missing keys default to ``0.0``.
        """
        if logs is None:
            return

        total_epochs = self.params.get("epochs", "?")

        # Guard against None start time if on_epoch_begin was skipped (e.g. resumed training)
        epoch_duration = (time.time() - self.epoch_time_start) * 1000 if self.epoch_time_start is not None else 0.0
        time_str = f"{epoch_duration:.0f}ms"

        dg_loss = logs.get("d/g_loss", 0.0)
        d_loss = logs.get("d_loss", 0.0)
        g_loss = logs.get("g_loss", 0.0)
        d_acc = logs.get("d_acc", 0.0)
        d_tnr = logs.get("d_tnr", 0.0)
        d_tpr = logs.get("d_tpr", 0.0)
        d_trick = logs.get("d_tricked", 0.0)
        d_fake_pred = logs.get("d_fake_pred", 0.0)
        d_real_pred = logs.get("d_real_pred", 0.0)
        d_bias = logs.get("d_bias", 0.0)

        print(
            f"Epoch {epoch + 1:>5}/{total_epochs:<5} - {time_str:>6}     "
            f"D/G_Loss: {dg_loss:.4f} - "
            f"D_Loss: {d_loss:.4f} - "
            f"G_Loss: {g_loss:.4f}     "
            f"D_Acc: {d_acc:.4f} - "
            f"D_TNR: {d_tnr:.4f} - "
            f"D_TPR: {d_tpr:.4f} - "
            f"D_Tricked: {d_trick:.4f}     "
            f"D_Fake_Pred: {d_fake_pred:.4f} - "
            f"D_Real_Pred: {d_real_pred:.4f} - "
            f"D_Bias: {d_bias:.4f}"
        )