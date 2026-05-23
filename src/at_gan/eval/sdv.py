"""Synthetic Data Vault (SDV) evaluator for statistical fidelity assessment."""

import pandas as pd
from sdmetrics.reports.single_table import QualityReport

from at_gan.utils.logger import get_logger

logger = get_logger(__name__)


class SDVEvaluator:
    """Evaluates synthetic data fidelity using SDMetrics Quality Reports."""

    def __init__(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
        """Initializes the SDV Evaluator."""
        self.real_df = real_df
        self.synth_df = synth_df

    def run_evaluation(self) -> dict:
        """Generates the SDV Quality Report.

        Returns:
            Dictionary containing the overall quality score, column shapes score, and pair trends score.
        """
        print("\nRunning SDV Quality Report (Statistical Fidelity)...", end=" ", flush=True)

        metadata = self._infer_metadata(self.real_df)
        report = QualityReport()

        report.generate(self.real_df, self.synth_df, metadata, verbose=False)

        properties = report.get_properties()
        col_shapes = properties.loc[properties['Property'] == 'Column Shapes', 'Score'].values[0]
        col_pairs = properties.loc[properties['Property'] == 'Column Pair Trends', 'Score'].values[0]

        overall_score = report.get_score()

        print(f"SDV Overall Score: {overall_score * 100:.2f}%")

        # print detailed scores
        for prop, score in properties[['Property', 'Score']].values:
            print(f"\t{prop}: {score * 100:.2f}%")

        return {
            "sdv_overall_score": float(overall_score),
            "sdv_column_shapes": float(col_shapes),
            "sdv_column_pairs": float(col_pairs),
        }

    def _infer_metadata(self, df: pd.DataFrame) -> dict:
        """Automatically infers SDV-compatible metadata from pandas dtypes."""
        columns = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                columns[col] = {"sdtype": "numerical"}
            else:
                columns[col] = {"sdtype": "categorical"}

        return {
            "primary_key": None,
            "columns": columns,
        }