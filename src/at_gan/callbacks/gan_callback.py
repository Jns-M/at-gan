"""Keras callback for periodic GAN evaluation, checkpointing, and W&B reporting."""

import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import wandb
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from at_gan.data.preprocessor import TabularPreprocessor
from at_gan.utils.paths import ExperimentPathManager


class GANCallback(tf.keras.callbacks.Callback):
    """Handles checkpointing, multi-metric fidelity evaluation, and W&B logging during GAN training."""

    def __init__(
        self,
        gan_model,
        preprocessor: TabularPreprocessor,
        real_data_sample,
        path_manager: ExperimentPathManager,
        save_frequency: int,
        eval_frequency: int,
    ):
        """Initializes the callback, pre-computes real data statistics, and sets up checkpoint managers.

        Args:
            gan_model: Compiled GAN model exposing ``generator``, ``discriminator``,
                ``g_optimizer``, ``d_optimizer``, and ``latent_dim``.
            preprocessor: Fitted ``TabularPreprocessor`` used to inverse-transform generated samples.
            real_data_sample: Scaled holdout ``tf.Tensor`` of real rows used as the evaluation reference.
            path_manager: ``ExperimentPathManager`` providing checkpoint and log directory paths.
            save_frequency: Checkpoint is saved every this many epochs.
            eval_frequency: Full evaluation suite runs every this many epochs.
        """
        super().__init__()
        self.gan_model = gan_model
        self.preprocessor = preprocessor

        self.real_data = real_data_sample

        real_data_raw = self.real_data.numpy()
        self.real_data_df = self.preprocessor.inverse_transform(real_data_raw)
        self.real_data_df = self.preprocessor.clean_re_encode(self.real_data_df)
        self.real_data_np = self.real_data_df.values

        self.path_manager = path_manager
        self.save_frequency = save_frequency
        self.eval_frequency = eval_frequency

        self.best_score = float("inf")
        self.best_epoch = 0
        self.best_metrics = {}

        self.checkpoint = tf.train.Checkpoint(
            g_optimizer=gan_model.g_optimizer,
            d_optimizer=gan_model.d_optimizer,
            generator=gan_model.generator,
            discriminator=gan_model.discriminator,
        )

        self.latest_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.path_manager.latest_checkpoints_dir,
            max_to_keep=1,
        )

        self.best_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.path_manager.best_checkpoints_dir,
            max_to_keep=1,
        )

        # Pre-compute real statistics once to avoid redundant work on every eval cycle
        self.real_corr = pd.DataFrame(self.real_data_np).corr().fillna(0).values

        # Pre-fit PCA on real data
        self.n_comps = min(5, self.real_data_np.shape[1])
        self.pca = PCA(n_components=self.n_comps)
        self.scaler = StandardScaler()

        real_data_scaled = self.scaler.fit_transform(self.real_data_np)
        self.real_pca = self.pca.fit_transform(real_data_scaled)

    def on_epoch_end(self, epoch, logs=None):
        """Triggers checkpointing and/or evaluation based on configured frequencies.

        Args:
            epoch: Integer index of the completed epoch (0-based).
            logs: Unused. Passed by the Keras training loop.
        """
        current_epoch = epoch + 1

        if current_epoch % self.save_frequency == 0:
            print("[Checkpoint] Saving checkpoint...")
            self.latest_manager.save(checkpoint_number=current_epoch)

        if current_epoch % self.eval_frequency == 0:
            print("[Eval] Running Evaluation Suite...")
            self.run_evaluation_suite(epoch=epoch)

    def on_train_end(self, logs=None):
        """Prints the best model summary to stdout at the conclusion of training.

        Args:
            logs: Unused. Passed by the Keras training loop.
        """
        print("\n" + "=" * 50)
        print(" TRAINING COMPLETE: BEST MODEL SUMMARY ")
        print("=" * 50)

        if self.best_epoch > 0:
            print(f"Best Model found at Epoch: {self.best_epoch}")
            print(f"Total Error Score:         {self.best_metrics.get('total_error', 0):.4f}")
            print(f"Adversarial AUC:           {self.best_metrics.get('adv_auc', 0):.4f}")
            print(f"Adversarial Error:         {self.best_metrics.get('adv_error', 0):.4f}")
            print(f"Correlation Error:         {self.best_metrics.get('corr_error', 0):.4f}")
            print(f"PCA Error:                 {self.best_metrics.get('pca_error', 0):.4f}")
            log_location = self.path_manager.best_checkpoints_dir / "best_eval.json"
            print(f"\nMetrics saved to: {log_location}")
        else:
            print("No evaluation was successfully recorded during training.")

        print("=" * 50 + "\n")

    def run_evaluation_suite(self, epoch: int):
        """Generates a synthetic batch, runs all fidelity sub-evaluations, logs to W&B, and saves the best model.

        Args:
            epoch: Current 0-based epoch number used explicitly as the target W&B step.
        """
        batch_size = self.real_data.shape[0]
        noise = tf.random.normal([batch_size, self.gan_model.latent_dim])

        fake_data_tf = self.gan_model.generator(noise, training=False)
        fake_data_df = self.preprocessor.inverse_transform(fake_data_tf.numpy())
        fake_data_df = self.preprocessor.clean_re_encode(fake_data_df)
        fake_data_np = fake_data_df.values

        # Wasserstein distance in the pre-fitted PCA latent space
        pca_error = self._evaluate_pca_distance(fake_data_np)

        # Mean absolute error between real and fake Pearson correlation matrices (logged but excluded from total score)
        corr_error, fake_corr = self._evaluate_correlations(fake_data_np)

        # AUC-based penalty from a held-out classifier trained to separate real from fake
        adv_error, adv_auc = self._evaluate_adversarial(fake_data_np)

        # root mean squared error calculated without correlation error
        raw_total_error = (pca_error ** 2) + (adv_error ** 2)
        total_error = np.sqrt(raw_total_error / 2.0)

        current_metrics = {
            "total_error": float(total_error),
            "adv_auc": float(adv_auc),
            "adv_error": float(adv_error),
            "corr_error": float(corr_error),
            "pca_error": float(pca_error),
        }

        corr_fig = self._plot_correlation_heatmap(self.real_corr, fake_corr)
        pca_fig = self._plot_pca_scatter(fake_data_np)

        if wandb.run is not None:
            wandb.log(
                {
                    "Eval/Total_Error": total_error,
                    "Eval/PCA_Error": pca_error,
                    "Eval/Correlation_Error": corr_error,
                    "Eval/Adversarial_AUC": adv_auc,
                    "Eval/Adversarial_Error": adv_error,
                    "Visuals/Correlation_Matrices": wandb.Image(corr_fig),
                    "Visuals/PCA_Overlap": wandb.Image(pca_fig),
                },
                step=epoch
            )

        plt.close(corr_fig)
        plt.close(pca_fig)

        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in current_metrics.items()])
        print(f"[Eval] Epoch {epoch + 1} Metrics | {metric_str}")

        if total_error < self.best_score:
            print(
                f"[Eval] New Best Score! Quality improved "
                f"({self.best_score:.4f} → {total_error:.4f}). Saving best model..."
            )
            self.best_score = total_error
            self.best_epoch = epoch + 1
            self.best_metrics = current_metrics
            self.best_manager.save(checkpoint_number=epoch + 1)
            self._save_best_log(epoch + 1, current_metrics)

    def _evaluate_correlations(self, fake_data_np: np.ndarray) -> tuple[float, np.ndarray]:
        """Computes mean absolute error of the off-diagonal Pearson correlations.

        Args:
            fake_data_np: Numpy array of inverse-transformed generated samples.

        Returns:
            Tuple of bounded correlation error and the fake correlation matrix.
        """
        fake_corr = pd.DataFrame(fake_data_np).corr().fillna(0).values

        # Extract the indices for the upper triangle excluding the diagonal
        upper_triangle_indices = np.triu_indices_from(self.real_corr, k=1)

        real_off_diag = self.real_corr[upper_triangle_indices]
        fake_off_diag = fake_corr[upper_triangle_indices]

        # Calculate the mean absolute error strictly on the meaningful correlations
        raw_corr_error = np.mean(np.abs(real_off_diag - fake_off_diag))

        # Squash it to keep the scale bounded between zero and one
        bounded_corr_error = self._squash_to_unit(float(raw_corr_error))

        return bounded_corr_error, fake_corr

    def _evaluate_adversarial(self, fake_data_np: np.ndarray) -> tuple[float, float]:
        """Trains a held-out classifier to distinguish real from fake samples and returns an AUC-based penalty.

        Args:
            fake_data_np: Numpy array of inverse-transformed generated samples.

        Returns:
            Tuple of ``(penalty, auc)`` where penalty is ``|auc - 0.5| * 2`` mapped to ``[0, 1]``.
        """
        X = np.vstack([self.real_data_np, fake_data_np])
        y = np.concatenate([np.ones(len(self.real_data_np)), np.zeros(len(fake_data_np))])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1130)

        # Shallow depth cap prevents overfitting on small eval sets while remaining a strong non-linear detector
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=1130, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        # AUC of 0.5 means the classifier cannot distinguish distributions; deviation in either direction is penalized
        penalty = abs(auc - 0.5) * 2.0
        return float(penalty), float(auc)

    def _evaluate_pca_distance(self, fake_data_np: np.ndarray) -> float:
        """Measures Wasserstein distance between real and fake projections on the pre-fitted PCA basis.

        Args:
            fake_data_np: Numpy array of inverse-transformed generated samples.

        Returns:
            Squashed scalar PCA distance error in ``[0, 1]``.
        """
        fake_scaled = self.scaler.transform(fake_data_np)
        fake_pca = self.pca.transform(fake_scaled)

        pca_errors = [
            wasserstein_distance(self.real_pca[:, i], fake_pca[:, i])
            for i in range(self.n_comps)
        ]
        return self._squash_to_unit(float(np.mean(pca_errors)))

    def _squash_to_unit(self, value: float) -> float:
        """Maps any non-negative scalar into a bounded range via an exponential function.

        Args:
            value: Non-negative raw error magnitude.

        Returns:
            Squashed float appropriately bounded.
        """
        return float(1.0 - np.exp(-value))

    def _plot_correlation_heatmap(self, real_corr: np.ndarray, fake_corr: np.ndarray) -> plt.Figure:
        """Renders a side-by-side heatmap of real, fake, and absolute-difference correlation matrices.

        Args:
            real_corr: Real data Pearson correlation matrix.
            fake_corr: Generated data Pearson correlation matrix.

        Returns:
            Matplotlib ``Figure`` with three heatmap panels.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
        axes[0].set_title("Real Data Correlation")

        sns.heatmap(fake_corr, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
        axes[1].set_title("Synthetic Data Correlation")

        # Plot the absolute difference
        diff_corr = np.abs(real_corr - fake_corr)
        sns.heatmap(diff_corr, ax=axes[2], cmap="Reds", vmin=0, vmax=2)
        axes[2].set_title("Absolute Difference")

        plt.tight_layout()
        return fig

    def _plot_pca_scatter(self, fake_data_np: np.ndarray) -> plt.Figure:
        """Renders a 2-D PCA scatter plot overlaying real and generated sample projections.

        Args:
            fake_data_np: Numpy array of inverse-transformed generated samples.

        Returns:
            Matplotlib ``Figure`` with a single scatter axes.
        """
        fake_scaled = self.scaler.transform(fake_data_np)
        fake_pca = self.pca.transform(fake_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.real_pca[:, 0], self.real_pca[:, 1], c="blue", alpha=0.3, label="Real", s=10)
        ax.scatter(fake_pca[:, 0], fake_pca[:, 1], c="red", alpha=0.3, label="Synthetic", s=10)
        ax.set_title("PCA Overlap: PC1 vs. PC2")
        ax.legend()
        plt.tight_layout()
        return fig

    def _save_best_log(self, epoch: int, metrics_dict: dict):
        """Serializes the best-epoch metrics to a JSON file in the best checkpoints directory.

        Args:
            epoch: Epoch number at which the best score was achieved.
            metrics_dict: Dictionary of metric names to float values.
        """
        log_path = self.path_manager.best_checkpoints_dir / "best_eval.json"
        log_data = {"best_epoch": epoch, "metrics": metrics_dict}

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=4)