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
            Dictionary containing min_dcr, mean_dcr, and 5th_percentile_dcr.
        """
        print("\nCalculating Distance to Closest Record (Privacy)...", end=" ", flush=True)

        # Ensure equal sizes for fair distance metrics
        n = min(len(self.real_df), len(self.synth_df))
        n = min(n, 10000)

        real_sample = self.real_df.sample(n=n, random_state=1130)
        synth_sample = self.synth_df.sample(n=n, random_state=1130)

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

        print(f"Min. DCR: {min_dcr:.4f}")

        print(f"\tMean DCR: {mean_dcr:.4f}")
        print(f"\t5th Percentile DCR: {p05_dcr:.4f}")

        return {
            "dcr_min": min_dcr,
            "dcr_mean": mean_dcr,
            "dcr_5th_percentile": p05_dcr,
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