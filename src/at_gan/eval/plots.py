"""Advanced plotting mechanism for synthetic data evaluation."""

import math
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

    def plot_pca(self, save_path: Optional[str] = None, max_samples: Optional[int] = 2500) -> None:
        """Renders a 2D PCA scatter plot overlaying real and synthetic data."""
        print("\nGenerating PCA Overlap Plot...", end=" ", flush=True)

        if max_samples is not None:
            real_df_pca = self.real_df.sample(min(max_samples, len(self.real_df)), random_state=1130)
            synth_df_pca = self.synth_df.sample(min(max_samples, len(self.synth_df)), random_state=1130)
        else:
            real_df_pca = self.real_df.copy()
            synth_df_pca = self.synth_df.copy()

        valid_cols = []
        for col in real_df_pca.columns:
            if not pd.api.types.is_numeric_dtype(real_df_pca[col]):
                unique_ratio = real_df_pca[col].dropna().nunique() / len(real_df_pca)
                if unique_ratio > 0.3:
                    continue
            valid_cols.append(col)

        real_subset = real_df_pca[valid_cols]
        synth_subset = synth_df_pca[valid_cols]

        real_encoded = pd.get_dummies(real_subset, drop_first=False)
        synth_encoded = pd.get_dummies(synth_subset, drop_first=False)

        synth_encoded = synth_encoded.reindex(columns=real_encoded.columns, fill_value=0)

        real_numeric = real_encoded.dropna()
        synth_numeric = synth_encoded.dropna()

        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_numeric)
        synth_scaled = scaler.transform(synth_numeric)

        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(real_pca[:, 0], real_pca[:, 1], c="#1f77b4", alpha=0.35, label="Real", s=15)
        ax.scatter(synth_pca[:, 0], synth_pca[:, 1], c="#d62728", alpha=0.35, label="Synthetic", s=15)

        ax.set_title("PCA Overlap: PC1 vs. PC2", fontsize=14, pad=10, weight='bold')
        ax.set_xlabel(f"Principal Component 1")
        ax.set_ylabel(f"Principal Component 2")
        ax.legend(frameon=True, shadow=True)
        sns.despine()

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_correlation_matrices(self, save_path: Optional[str] = None) -> None:
        """Generates side-by-side heatmaps of real corr, synthetic corr, and their difference."""
        print("\nGenerating Correlation Matrices...", end=" ", flush=True)

        real_encoded = self.real_df.copy()
        synth_encoded = self.synth_df.copy()

        for col in real_encoded.columns:
            if not pd.api.types.is_numeric_dtype(real_encoded[col]):
                labels, uniques = pd.factorize(real_encoded[col])
                real_encoded[col] = labels
                mapping = {val: idx for idx, val in enumerate(uniques)}
                synth_encoded[col] = synth_encoded[col].map(mapping).fillna(-1)

        real_corr = real_encoded.corr().fillna(0)
        synth_corr = synth_encoded.corr().fillna(0)
        diff_corr = np.abs(real_corr - synth_corr)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        base_cmap = plt.get_cmap('RdBu_r')
        truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
            'lighter_rdbu', base_cmap(np.linspace(0.15, 0.85, 256))
        )
        base_cmpa_2 = plt.get_cmap('Reds')
        truncated_cmap_2 = mcolors.LinearSegmentedColormap.from_list(
            'lighter_reds', base_cmpa_2(np.linspace(0.0, 0.75, 256))
        )

        sns.heatmap(real_corr, ax=axes[0], cmap=truncated_cmap, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
        axes[0].set_title("Real Data Correlation", fontsize=12, weight='bold')

        sns.heatmap(synth_corr, ax=axes[1], cmap=truncated_cmap, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
        axes[1].set_title("Synthetic Data Correlation", fontsize=12, weight='bold')

        sns.heatmap(diff_corr, ax=axes[2], cmap=truncated_cmap_2, vmin=0, vmax=1, square=True, cbar_kws={"shrink": .8})
        axes[2].set_title("Absolute Difference in Correlation", fontsize=12, weight='bold')


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

            if f_type == 'continuous':
                sns.kdeplot(data=self.real_df, x=feature, ax=ax, color="#1f77b4", label="Real", fill=True, alpha=0.4,
                            warn_singular=False)
                sns.kdeplot(data=self.synth_df, x=feature, ax=ax, color="#d62728", label="Synthetic", fill=True,
                            alpha=0.4, warn_singular=False)
                ax.set_ylabel("Density")
                ax.set_xlabel(feature)

            else:
                def clean_val(val):
                    try:
                        if isinstance(val, (int, float, np.number)) and val == int(val):
                            return int(val)
                    except (ValueError, TypeError):
                        pass
                    return val

                real_series = self.real_df[feature].dropna().map(clean_val)
                synth_series = self.synth_df[feature].dropna().map(clean_val)

                if f_type == 'categorical':
                    combined_counts = pd.concat([real_series, synth_series]).value_counts(normalize=True)
                    valid_categories = combined_counts[combined_counts >= 0.01].index

                    real_series = real_series[real_series.isin(valid_categories)]
                    synth_series = synth_series[synth_series.isin(valid_categories)]

                real_counts = real_series.value_counts(normalize=False).rename("Frequency")
                synth_counts = synth_series.value_counts(normalize=False).rename("Frequency")

                real_inst = real_counts.to_frame().reset_index()
                real_inst["Dataset"] = "Real"

                synth_inst = synth_counts.to_frame().reset_index()
                synth_inst["Dataset"] = "Synthetic"

                combined_dist = pd.concat([real_inst, synth_inst], ignore_index=True)

                if f_type == 'categorical':
                    combined_codes, _ = pd.factorize(combined_dist[feature])
                    plot_x_var = "Numeric Code"
                    combined_dist[plot_x_var] = combined_codes
                else:
                    plot_x_var = feature

                sns.barplot(
                    data=combined_dist,
                    x=plot_x_var,
                    y="Frequency",
                    hue="Dataset",
                    ax=ax,
                    palette=["#1f77b4", "#d62728"],
                    alpha=0.65
                )
                ax.set_ylabel("Frequency")
                ax.set_xlabel(feature)

                if f_type == 'binary':
                    real_pcts = real_series.value_counts(normalize=True) * 100
                    synth_pcts = synth_series.value_counts(normalize=True) * 100

                    max_height = combined_dist["Frequency"].max()
                    ax.set_ylim(0, max_height * 1.18)

                    categories = [t.get_text() for t in ax.get_xticklabels()]

                    for c_idx, container in enumerate(ax.containers):
                        is_real_dataset = (c_idx == 0)

                        for bar_idx, p in enumerate(container):
                            height = p.get_height()
                            if not np.isnan(height) and height > 0:
                                if bar_idx < len(categories):
                                    category_label = categories[bar_idx]

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

            ax.set_title(f"Distribution of '{feature}'", fontsize=11, weight='bold')

            if i == 0:
                ax.legend(frameon=True)
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_2d_correlations(self, feature_pairs: List[tuple], save_path: Optional[str] = None) -> None:
        """Plots 2D scatter/density plots with percentile-based outlier-robust scale alignment."""
        print("\nGenerating 2D Feature Correlations...", end=" ", flush=True)

        num_pairs = len(feature_pairs)
        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, num_pairs * 4), squeeze=False)

        for i, (f1, f2) in enumerate(feature_pairs):
            f1, f2 = f1.lower(), f2.lower()

            combined_x = pd.concat([self.real_df[f1], self.synth_df[f1]]).dropna()
            combined_y = pd.concat([self.real_df[f2], self.synth_df[f2]]).dropna()
            x_min, x_max = np.percentile(combined_x, [0, 99.9])
            y_min, y_max = np.percentile(combined_y, [0, 99.9])

            sns.kdeplot(data=self.real_df, x=f1, y=f2, ax=axes[i][0], cmap="Blues", fill=True, alpha=0.8, warn_singular=False)
            axes[i][0].set_title(f"Real: '{f1}' vs. '{f2}'", weight='bold')
            axes[i][0].set_xlim(x_min, x_max)
            axes[i][0].set_ylim(y_min, y_max)

            sns.kdeplot(data=self.synth_df, x=f1, y=f2, ax=axes[i][1], cmap="Reds", fill=True, alpha=0.8, warn_singular=False)
            axes[i][1].set_title(f"Synthetic: '{f1}' vs. '{f2}'", weight='bold')
            axes[i][1].set_xlim(x_min, x_max)
            axes[i][1].set_ylim(y_min, y_max)

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_categorical_vs_continuous(self, cat_feature: str, cont_feature: str,
                                       save_path: Optional[str] = None) -> None:
        """Plots a single horizontal grouped boxplot comparing continuous distributions nested within categoricals."""
        cat_feature, cont_feature = cat_feature.lower(), cont_feature.lower()
        print(f"\nGenerating Grouped Nested Boxplot: {cont_feature} by {cat_feature}...", end=" ", flush=True)

        def clean_val(val):
            try:
                if isinstance(val, (int, float, np.number)) and val == int(val):
                    return str(int(val))
            except (ValueError, TypeError):
                pass
            return str(val)

        real_cat = self.real_df[cat_feature].map(clean_val)
        synth_cat = self.synth_df[cat_feature].map(clean_val)

        combined_df = pd.concat([
            pd.DataFrame({cat_feature: real_cat, cont_feature: self.real_df[cont_feature], 'Dataset': 'Real'}),
            pd.DataFrame({cat_feature: synth_cat, cont_feature: self.synth_df[cont_feature], 'Dataset': 'Synthetic'})
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(
            data=combined_df,
            x=cont_feature,
            y=cat_feature,
            hue="Dataset",
            ax=ax,
            palette=["#1f77b4", "#d62728"],
            orient="h",
            linewidth=1.5,
            gap=0.1,
        )

        for patch in ax.patches:
            patch.set_alpha(0.75)

        ax.set_title(f"Comparison of '{cont_feature}' across '{cat_feature}'", fontsize=14, pad=10, weight='bold')
        ax.set_xlabel(cont_feature)
        ax.set_ylabel(cat_feature)
        ax.legend(frameon=True, shadow=True)
        sns.despine()

        plt.tight_layout()
        self._handle_output(fig, save_path)

    def plot_conditional_error_matrix(
            self,
            binary_feature: str,
            continuous_feature: str,
            threshold: float = 0.0,
            save_path: Optional[str] = None
    ) -> dict:
        """Evaluates conditional constraint validation errors between a flag and boundary limit."""
        b_feat, c_feat = binary_feature.lower(), continuous_feature.lower()
        print(f"\nEvaluating Logical Conditional Matrix: '{b_feat}' vs. '{c_feat}'...", end=" ", flush=True)

        total_samples = len(self.synth_df)

        flag_true_above = ((self.synth_df[c_feat] > threshold) & (self.synth_df[b_feat] == 1)).sum()
        flag_false_above = ((self.synth_df[c_feat] > threshold) & (self.synth_df[b_feat] == 0)).sum()
        flag_true_at_below = ((self.synth_df[c_feat] <= threshold) & (self.synth_df[b_feat] == 1)).sum()
        flag_false_at_below = ((self.synth_df[c_feat] <= threshold) & (self.synth_df[b_feat] == 0)).sum()

        percentages = np.array([
            [flag_true_above / total_samples, flag_false_above / total_samples],
            [flag_true_at_below / total_samples, flag_false_at_below / total_samples]
        ])

        fig, ax = plt.subplots(figsize=(6, 5))

        base_cmap = plt.get_cmap('Reds')
        truncated_reds = mcolors.LinearSegmentedColormap.from_list(
            'matrix_reds', base_cmap(np.linspace(0.0, 0.7, 256))
        )

        ax.grid(False, axis='both')

        sns.heatmap(
            percentages,
            annot=True,
            fmt=".2%",
            cmap=truncated_reds,
            alpha=0.8,
            xticklabels=['1', '0'],
            yticklabels=[f'> {int(threshold)}', f'{int(threshold)}'],
            ax=ax,
            square=True,
            cbar=True,
            cbar_kws={"shrink": .75},
            linewidths=0,
            linecolor='white',
        )

        ax.set_title(f"Conditional Logic Check\n('{b_feat}' vs. '{c_feat}')", fontsize=11, weight='bold', pad=15)
        ax.set_xlabel(b_feat)
        ax.set_ylabel(c_feat)

        plt.tight_layout()

        false_positive_error = flag_true_at_below / total_samples
        false_negative_error = flag_false_above / total_samples
        combined_constraint_error = false_positive_error + false_negative_error

        matrix_results = {
            f"{b_feat}_conditional_false_positive_error": float(false_positive_error),
            f"{b_feat}_conditional_false_negative_error": float(false_negative_error),
            f"{b_feat}_combined_logical_error": float(combined_constraint_error)
        }

        self._handle_output(fig, save_path)
        return matrix_results

    def generate_all_plots(self, save_dir: Optional[str] = None) -> None:
        """Master function to execute the full evaluation plotting suite."""
        self.plot_pca(save_path=f"{save_dir}/pca_overlap.png" if save_dir else None)
        self.plot_correlation_matrices(save_path=f"{save_dir}/correlation_matrices.png" if save_dir else None)
        self.plot_feature_distributions(save_path=f"{save_dir}/feature_distributions.png" if save_dir else None)
        print("\nAll plots generated successfully.")

    def plot_model_comparison(self, metrics_list: List[dict], save_dir: Optional[str] = None) -> None:
        """Generates a multi-plot suite comparing performance across multiple generative models.

        Args:
            metrics_list: List of metric dictionaries, each containing 'dcr', 'sdv',
                         'tstr', and 'name' keys.
            save_dir: Optional directory path where generated plots will be saved.
        """
        print(f"\nGenerating Multi-Model Comparison Plots for {len(metrics_list)} models...", flush=True)

        # ----------------------------------------------------
        # Extract Data into DataFrames
        # ----------------------------------------------------
        overall_data = []
        dcr_dist_data = []
        copies_data = []
        tstr_data = []
        trtr_values = []

        for m in metrics_list:
            name = m['name']

            # 1. Macro Overview Metric parsing
            privacy_score = 100.0 - m['dcr'].get('pct_exact_copies', 0.0)
            overall_data.append({'Model': name, 'Metric': 'Privacy Score\n(100 - % Copies)', 'Value': privacy_score})
            overall_data.append(
                {'Model': name, 'Metric': 'Statistical Fidelity\n(SDV)', 'Value': m['sdv']['sdv_overall_score'] * 100})
            overall_data.append(
                {'Model': name, 'Metric': 'Utility Retention\n(F1 Score)', 'Value': m['tstr']['utility_retention']})

            # 2. DCR Distances
            dcr_dist_data.append({'Model': name, 'Metric': 'DCR Min', 'Value': m['dcr']['dcr_min']})
            dcr_dist_data.append(
                {'Model': name, 'Metric': 'DCR 5th Percentile', 'Value': m['dcr']['dcr_5th_percentile']})
            dcr_dist_data.append({'Model': name, 'Metric': 'DCR Mean', 'Value': m['dcr']['dcr_mean']})

            # 3. Exact Copies Percentages
            copies_data.append({'Model': name, 'Pct Exact Copies': m['dcr']['pct_exact_copies']})

            # 4. TSTR Machine Learning Utility Performance
            tstr_data.append(
                {'Model': name, 'Metric': 'TSTR (Synth Train / Real Test)', 'Value': m['tstr']['tstr_mean_f1'] * 100})
            if 'trtr_mean_f1' in m['tstr']:
                trtr_values.append(m['tstr']['trtr_mean_f1'] * 100)

        df_overall = pd.DataFrame(overall_data)
        df_dcr = pd.DataFrame(dcr_dist_data)
        df_copies = pd.DataFrame(copies_data)
        df_tstr = pd.DataFrame(tstr_data)

        sns.set_theme(style="whitegrid", palette="Dark2")

        # ----------------------------------------------------
        # Plot A: Framework Macro Breakdown (0 - 100%)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_overall, x='Metric', y='Value', hue='Model', ax=ax, alpha=0.85)
        ax.set_title("Synthesis Quality & Privacy Matrix Comparison", fontsize=14, weight='bold', pad=15)
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, 110)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%', padding=3, fontweight='bold', fontsize=9)

        ax.legend(frameon=True, shadow=True, loc='lower right')
        sns.despine()
        plt.tight_layout()
        self._handle_output(fig, f"{save_dir}/comparison_macro_metrics.png" if save_dir else None)

        # ----------------------------------------------------
        # Plot B: DCR Boundary Distance Statistics
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_dcr, x='Metric', y='Value', hue='Model', ax=ax, alpha=0.85)
        ax.set_title("Distance to Closest Record (DCR) Statistics", fontsize=14, weight='bold', pad=15)
        ax.set_ylabel("Euclidean Distance Scale (Higher = More Private)")
        ax.set_xlabel("")

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontweight='bold', fontsize=9)

        ax.legend(frameon=True, shadow=True, loc='lower right')
        sns.despine()
        plt.tight_layout()
        self._handle_output(fig, f"{save_dir}/comparison_dcr_distances.png" if save_dir else None)

        # ----------------------------------------------------
        # Plot C: Memory Leakage (Percentage Exact Copies)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_copies, x='Model', y='Pct Exact Copies', ax=ax, color='#d62728', alpha=0.75)
        ax.set_title("Data Leakage: Rate of Exact Matches Detected", fontsize=14, weight='bold', pad=15)
        ax.set_xlabel("Generative Pipeline Framework")
        ax.set_ylabel("Exact Copies (%)")

        # Balance scale if values are tiny, while preserving room for labels
        max_val = df_copies['Pct Exact Copies'].max()
        ax.set_ylim(0, max(max_val * 1.15, 5.0))

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%', padding=3, fontweight='bold', fontsize=9)

        sns.despine()
        plt.tight_layout()
        self._handle_output(fig, f"{save_dir}/comparison_exact_copies.png" if save_dir else None)

        # ----------------------------------------------------
        # Plot D: ML Utility Analysis (TSTR vs. Shared TRTR Baseline)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_tstr, x='Metric', y='Value', hue='Model', ax=ax, alpha=0.85)
        ax.set_title("Downstream Predictor Performance: TSTR vs. Real Baseline", fontsize=14, weight='bold', pad=15)
        ax.set_ylabel("Mean Cross-Model Classifier F1 Score (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, 110)

        # Draw a single unified reference line for the shared TRTR value if present
        if trtr_values:
            mean_trtr = np.mean(trtr_values)
            ax.axhline(mean_trtr, color='#1f77b4', linestyle='--', linewidth=2,
                       label=f'TRTR Baseline ({mean_trtr:.2f}%)')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%', padding=3, fontweight='bold', fontsize=9)

        ax.legend(frameon=True, shadow=True, loc='lower right')
        sns.despine()
        plt.tight_layout()
        self._handle_output(fig, f"{save_dir}/comparison_utility_train_test.png" if save_dir else None)

        if save_dir:
            print(f"All comparison plots successfully exported to: {save_dir}")

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