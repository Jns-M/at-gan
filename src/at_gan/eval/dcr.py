"""Distance to Closest Record (DCR) evaluator for privacy assessment."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from at_gan.utils.logger import get_logger

logger = get_logger(__name__)


class DCREvaluator:
    """Evaluates synthetic data privacy by measuring distances to the closest real records."""

    def __init__(
            self,
            real_df: pd.DataFrame,
            synth_df: pd.DataFrame,
            target_column: str,
    ) -> None:
        """Initializes the DCR Evaluator with pre-loaded DataFrames."""
        self.real_df = real_df.drop(columns=[target_column], errors="ignore")
        self.synth_df = synth_df.drop(columns=[target_column], errors="ignore")

    def run_evaluation(self) -> dict:
        """Computes the DCR metrics across the dataset.

        Returns:
            Dictionary containing min_dcr, mean_dcr, 5th_percentile_dcr,
            exact_copies, and pct_exact_copies.
        """
        print("\nCalculating Distance to Closest Record (Privacy)...", end=" ", flush=True)

        real_sample = self.real_df
        synth_sample = self.synth_df

        preprocessor = self._build_preprocessor(real_sample)

        X_real_scaled = preprocessor.fit_transform(real_sample)
        X_synth_scaled = preprocessor.transform(synth_sample)

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
        nn.fit(X_real_scaled)

        distances, _ = nn.kneighbors(X_synth_scaled)
        distances = distances.flatten()

        min_dcr = float(np.min(distances))
        mean_dcr = float(np.mean(distances))
        p05_dcr = float(np.percentile(distances, 5))

        exact_copies = int(np.sum(distances < 5e-5))
        pct_exact_copies = float(exact_copies / len(distances) * 100)

        print(f"Min. DCR: {min_dcr:.4f}")
        print(f"\tMean DCR: {mean_dcr:.4f}")
        print(f"\t5th Percentile DCR: {p05_dcr:.4f}")
        print(f"\tExact Copies: {exact_copies} ({pct_exact_copies:.2f}%)")

        return {
            "dcr_min": min_dcr,
            "dcr_mean": mean_dcr,
            "dcr_5th_percentile": p05_dcr,
            "exact_copies": exact_copies,
            "pct_exact_copies": pct_exact_copies,
        }

    def _build_preprocessor(self, X_reference: pd.DataFrame) -> ColumnTransformer:
        """Creates a scaler/encoder pipeline to ensure distance math is uniform."""
        categorical_cols = X_reference.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X_reference.select_dtypes(exclude=["object", "category"]).columns.tolist()

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ],
            remainder="drop",
        )