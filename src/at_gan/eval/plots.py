"""Advanced plotting mechanism for synthetic data evaluation."""

import math
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class GANEvaluationPlotter:
    """Generates high-resolution evaluation plots comparing real and synthetic data."""

    def __init__(self, real_df: pd.DataFrame, synth_df: pd.DataFrame, dpi: int = 300) -> None:
        """Initializes the plotter and sets global high-resolution styling."""
        self.real_df = real_df
        self.synth_df = synth_df
        self.dpi = dpi

        sns.set_theme(style="whitegrid", palette="Dark2")
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi

    def plot_pca(self, save_path: Optional[str] = None) -> None:
        """Renders a 2D PCA scatter plot overlaying real and synthetic data."""
        print("\nGenerating PCA Overlap Plot...", end=" ", flush=True)

        numeric_cols = self.real_df.select_dtypes(include=[np.number]).columns
        real_numeric = self.real_df[numeric_cols].dropna()
        synth_numeric = self.synth_df[numeric_cols].dropna()

        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_numeric)
        synth_scaled = scaler.transform(synth_numeric)

        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(real_pca[:, 0], real_pca[:, 1], c="#1f77b4", alpha=0.3, label="Real", s=15)
        ax.scatter(synth_pca[:, 0], synth_pca[:, 1], c="#d62728", alpha=0.3, label="Synthetic", s=15)

        ax.set_title("PCA Overlap: PC1 vs. PC2", fontsize=14, pad=10)
        ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(frameon=True, shadow=True)
        sns.despine()

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_correlation_matrices(self, save_path: Optional[str] = None) -> None:
        """Generates side-by-side heatmaps of real corr, synthetic corr, and their difference."""
        print("\nGenerating Correlation Matrices...", end=" ", flush=True)

        numeric_cols = self.real_df.select_dtypes(include=[np.number]).columns
        real_corr = self.real_df[numeric_cols].corr()
        synth_corr = self.synth_df[numeric_cols].corr()
        diff_corr = real_corr - synth_corr

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(real_corr, ax=axes[0], cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
        axes[0].set_title("Real Data Correlation", fontsize=12)

        sns.heatmap(synth_corr, ax=axes[1], cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
        axes[1].set_title("Synthetic Data Correlation", fontsize=12)

        sns.heatmap(diff_corr, ax=axes[2], cmap="coolwarm", vmin=-0.5, vmax=0.5, square=True, cbar_kws={"shrink": .8})
        axes[2].set_title("Difference (Real - Synthetic)", fontsize=12)

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_feature_distributions(self, features: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
        """Plots distributions for specified features in a multi-plot grid."""
        print("\nGenerating Feature Distributions...", end=" ", flush=True)

        if features is None:
            features = self.real_df.columns.tolist()

        num_features = len(features)
        cols = 3
        rows = math.ceil(num_features / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            if pd.api.types.is_numeric_dtype(self.real_df[feature]):
                # Added warn_singular=False to suppress 0-variance warnings
                sns.kdeplot(data=self.real_df, x=feature, ax=ax, color="#1f77b4", label="Real", fill=True, alpha=0.3, warn_singular=False)
                sns.kdeplot(data=self.synth_df, x=feature, ax=ax, color="#d62728", label="Synthetic", fill=True, alpha=0.3, warn_singular=False)
            else:
                combined_df = pd.concat([
                    pd.DataFrame({feature: self.real_df[feature], 'Dataset': 'Real'}),
                    pd.DataFrame({feature: self.synth_df[feature], 'Dataset': 'Synthetic'})
                ])
                sns.countplot(data=combined_df, x=feature, hue='Dataset', ax=ax, palette=["#1f77b4", "#d62728"], alpha=0.8)

            ax.set_title(f"Distribution of {feature}")
            if i == 0:
                ax.legend()
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_2d_correlations(self, feature_pairs: List[tuple], save_path: Optional[str] = None) -> None:
        """Plots 2D scatter/density plots for specific pairs of continuous features."""
        print("\nGenerating 2D Feature Correlations...", end=" ", flush=True)

        num_pairs = len(feature_pairs)
        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, num_pairs * 4))

        if num_pairs == 1:
            axes = [axes]

        for i, (f1, f2) in enumerate(feature_pairs):
            sns.kdeplot(data=self.real_df, x=f1, y=f2, ax=axes[i][0], cmap="Blues", fill=True, alpha=0.8, warn_singular=False)
            axes[i][0].set_title(f"Real: {f1} vs {f2}")

            sns.kdeplot(data=self.synth_df, x=f1, y=f2, ax=axes[i][1], cmap="Reds", fill=True, alpha=0.8, warn_singular=False)
            axes[i][1].set_title(f"Synthetic: {f1} vs {f2}")

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_categorical_vs_continuous(self, cat_feature: str, cont_feature: str, save_path: Optional[str] = None) -> None:
        """Plots a grouped boxplot comparing continuous value distributions across categories."""
        print(f"\nGenerating Boxplot: {cont_feature} by {cat_feature}...", end=" ", flush=True)

        combined_df = pd.concat([
            pd.DataFrame({cat_feature: self.real_df[cat_feature], cont_feature: self.real_df[cont_feature], 'Dataset': 'Real'}),
            pd.DataFrame({cat_feature: self.synth_df[cat_feature], cont_feature: self.synth_df[cont_feature], 'Dataset': 'Synthetic'})
        ])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=combined_df, x=cat_feature, y=cont_feature, hue="Dataset", ax=ax, palette=["#1f77b4", "#d62728"])

        ax.set_title(f"Comparison of {cont_feature} across {cat_feature}", fontsize=14, pad=10)
        sns.despine()

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def generate_all_plots(self, save_dir: Optional[str] = None) -> None:
        """Master function to execute the full evaluation plotting suite."""
        self.plot_pca(save_path=f"{save_dir}/pca_overlap.png" if save_dir else None)
        self.plot_correlation_matrices(save_path=f"{save_dir}/correlation_matrices.png" if save_dir else None)
        self.plot_feature_distributions(save_path=f"{save_dir}/feature_distributions.png" if save_dir else None)
        print("\nAll plots generated successfully.")

    def _handle_output(self, fig: plt.Figure, save_path: Optional[str]) -> None:
        """Internal helper to force-render plots regardless of the active Matplotlib backend."""
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            try:
                import io
                from IPython.display import Image, display
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
                buf.seek(0)
                plt.close(fig)
                display(Image(data=buf.getvalue(), format='png'))
            except Exception as e:
                print(f"Rendering failed: {e}")
                plt.close(fig)