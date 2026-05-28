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
        ax.scatter(real_pca[:, 0], real_pca[:, 1], c="#1f77b4", alpha=0.5, label="Real", s=15)
        ax.scatter(synth_pca[:, 0], synth_pca[:, 1], c="#d62728", alpha=0.5, label="Synthetic", s=15)

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

    def plot_feature_distributions(
            self,
            features: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ) -> None:
        """Plots high-res distributions for features in a smart multi-plot grid.

        Args:
            features: List of feature names to plot. Defaults to all columns.
            save_path: Path to save the figure file.
        """
        print("\nGenerating Feature Distributions...", end=" ", flush=True)

        if features is None:
            features = self.real_df.columns.tolist()
        else:
            features = [f.lower() for f in features]

        # Categorize and reorder features so that the same types are grouped sequentially
        continuous_features = []
        binary_features = []
        categorical_features = []

        for feature in features:
            unique_vals = self.real_df[feature].dropna().nunique()
            if unique_vals == 2:
                binary_features.append(feature)
            elif pd.api.types.is_numeric_dtype(self.real_df[feature]) and unique_vals > 10:
                continuous_features.append(feature)
            else:
                categorical_features.append(feature)

        ordered_features = continuous_features + binary_features + categorical_features
        num_features = len(ordered_features)
        cols = 3
        rows = math.ceil(num_features / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()

        for i, feature in enumerate(ordered_features):
            ax = axes[i]
            unique_vals = self.real_df[feature].dropna().nunique()
            if unique_vals == 2:
                f_type = 'binary'
            elif pd.api.types.is_numeric_dtype(self.real_df[feature]) and unique_vals > 10:
                f_type = 'continuous'
            else:
                f_type = 'categorical'

            # Render the plot based on the determined type
            if f_type == 'continuous':
                # Smooth continuous density curves
                sns.kdeplot(data=self.real_df, x=feature, ax=ax, color="#1f77b4", label="Real", fill=True, alpha=0.5,
                            warn_singular=False)
                sns.kdeplot(data=self.synth_df, x=feature, ax=ax, color="#d62728", label="Synthetic", fill=True,
                            alpha=0.5, warn_singular=False)
                ax.set_ylabel("Density")
                ax.set_xlabel(feature)

            else:
                # Clean up float values to flat integers for display names where matching
                def clean_val(val):
                    try:
                        if isinstance(val, (int, float, np.number)) and val == int(val):
                            return int(val)
                    except (ValueError, TypeError):
                        pass
                    return val

                real_series = self.real_df[feature].dropna().map(clean_val)
                synth_series = self.synth_df[feature].dropna().map(clean_val)

                # Filter out values that represent less than 1% of total counts across both datasets
                if f_type == 'categorical':
                    combined_counts = pd.concat([real_series, synth_series]).value_counts(normalize=True)
                    valid_categories = combined_counts[combined_counts >= 0.01].index

                    real_series = real_series[real_series.isin(valid_categories)]
                    synth_series = synth_series[synth_series.isin(valid_categories)]

                # Prepare data based on absolute sample frequency counts
                real_counts = real_series.value_counts(normalize=False).rename("Frequency")
                synth_counts = synth_series.value_counts(normalize=False).rename("Frequency")

                real_inst = real_counts.to_frame().reset_index()
                real_inst["Dataset"] = "Real"

                synth_inst = synth_counts.to_frame().reset_index()
                synth_inst["Dataset"] = "Synthetic"

                combined_dist = pd.concat([real_inst, synth_inst], ignore_index=True)

                # Anonymize categorical string assignments into clear sequence numbers
                if f_type == 'categorical':
                    combined_codes, _ = pd.factorize(combined_dist[feature])
                    plot_x_var = "Numeric Code"
                    combined_dist[plot_x_var] = combined_codes
                else:
                    plot_x_var = feature

                # Side-by-side grouped categorical bar chart comparison
                sns.barplot(
                    data=combined_dist,
                    x=plot_x_var,
                    y="Frequency",
                    hue="Dataset",
                    ax=ax,
                    palette=["#1f77b4", "#d62728"],
                    alpha=0.5
                )
                ax.set_ylabel("Frequency")
                ax.set_xlabel(feature)

                # If the type is binary, dynamically calculate heights and overlay percentage annotations above the bars
                if f_type == 'binary':
                    # Calculate real percentages for the bar annotations
                    real_pcts = real_series.value_counts(normalize=True) * 100
                    synth_pcts = synth_series.value_counts(normalize=True) * 100

                    max_height = combined_dist["Frequency"].max()
                    ax.set_ylim(0, max_height * 1.18)

                    # Determine ordered categorical alignment mapped along the active x-axis
                    categories = [t.get_text() for t in ax.get_xticklabels()]

                    # Use axis containers to parse groups separately
                    for c_idx, container in enumerate(ax.containers):
                        is_real_dataset = (c_idx == 0)

                        for bar_idx, p in enumerate(container):
                            height = p.get_height()
                            if not np.isnan(height) and height > 0:
                                if bar_idx < len(categories):
                                    category_label = categories[bar_idx]

                                    # Safe variant casting sequence for accurate dictionary indexing
                                    try:
                                        if category_label.endswith('.0'):
                                            category_label = category_label[:-2]
                                        category_key = int(category_label) if category_label.isdigit() else category_label
                                    except ValueError:
                                        category_key = category_label

                                    if is_real_dataset:
                                        pct_val = real_pcts.get(category_key, 0.0)
                                    else:
                                        pct_val = synth_pcts.get(category_key, 0.0)

                                    ax.annotate(
                                        f"{pct_val:.1f}%",
                                        (p.get_x() + p.get_width() / 2.0, height),
                                        ha='center',
                                        va='bottom',
                                        fontsize=9,
                                        fontweight='bold',
                                        xytext=(0, 3),
                                        textcoords='offset points'
                                    )

            ax.set_title(f"Distribution of {feature} ({f_type})", fontsize=11, weight='bold')

            # Clean legend management across subplot boundaries
            if i == 0:
                ax.legend(frameon=True)
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        # Wipe remaining empty subplot quadrants
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_2d_correlations(self, feature_pairs: List[tuple], save_path: Optional[str] = None) -> None:
        """Plots 2D scatter/density plots for specific pairs of continuous features."""
        print("\nGenerating 2D Feature Correlations...", end=" ", flush=True)

        num_pairs = len(feature_pairs)
        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, num_pairs * 4), squeeze=False)

        for i, (f1, f2) in enumerate(feature_pairs):
            f1, f2 = f1.lower(), f2.lower()

            sns.kdeplot(data=self.real_df, x=f1, y=f2, ax=axes[i][0], cmap="Blues", fill=True, alpha=0.5, warn_singular=False)
            axes[i][0].set_title(f"Real: {f1} vs {f2}")

            sns.kdeplot(data=self.synth_df, x=f1, y=f2, ax=axes[i][1], cmap="Reds", fill=True, alpha=0.5, warn_singular=False)
            axes[i][1].set_title(f"Synthetic: {f1} vs {f2}")

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_categorical_vs_continuous(self, cat_feature: str, cont_feature: str, save_path: Optional[str] = None) -> None:
        """Plots a grouped boxplot comparing continuous value distributions across categories."""
        cat_feature, cont_feature = cat_feature.lower(), cont_feature.lower()
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
                import sys
                from IPython.display import Image, display

                sys.stdout.flush()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
                buf.seek(0)
                plt.close(fig)
                display(Image(data=buf.getvalue(), format='png'))
            except Exception as e:
                print(f"Rendering failed: {e}")
                plt.close(fig)