"""Train-on-Synthetic, Test-on-Real (TSTR) evaluator for post-hoc synthetic data quality assessment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from at_gan.utils.logger import get_logger

logger = get_logger(__name__)

_CLASSIFIERS: dict[str, object] = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=1130),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=1130),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=1130),
}


class TSTREvaluator:
    """Evaluates synthetic data quality via Train-on-Synthetic, Test-on-Real (TSTR).

    Runs complementary assessments:
    * TRTR baseline: Classifiers trained on real data, tested on a held-out real split.
    * TSTR: Classifiers trained on synthetic data, tested on the same real split.

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: GAN-generated DataFrame.
        target_column: Name of the classification target column.
        test_size: Fraction of real data held out as the evaluation test split.
        random_state: seed for all splits and classifiers.
    """

    def __init__(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 1130,
    ) -> None:
        self.real_df = real_df.copy()
        self.synth_df = synth_df.copy()
        self.target_column = target_column.lower()
        self.test_size = test_size
        self.random_state = random_state

    def run_evaluation(self) -> dict:
        """Runs the TSTR evaluation suite and returns a metrics dictionary."""
        print("\nRunning TSTR Evaluation (Utility) ...", end=" ", flush=True)

        X_real_train, X_real_test, y_real_train, y_real_test, X_synth, y_synth = self._prepare_splits()
        preprocessor = self._build_preprocessor(X_real_train)

        # TRTR baseline
        trtr_results = self._evaluate_classifiers(
            label="TRTR",
            X_train=X_real_train,
            y_train=y_real_train,
            X_test=X_real_test,
            y_test=y_real_test,
            preprocessor=preprocessor,
        )

        # TSTR evaluation
        tstr_results = self._evaluate_classifiers(
            label="TSTR",
            X_train=X_synth,
            y_train=y_synth,
            X_test=X_real_test,
            y_test=y_real_test,
            preprocessor=preprocessor,
        )

        # Aggregate means
        trtr_mean_f1 = float(np.mean([v for k, v in trtr_results.items() if k.endswith("_f1")]))
        tstr_mean_f1 = float(np.mean([v for k, v in tstr_results.items() if k.endswith("_f1")]))
        utility_retention = (tstr_mean_f1 / trtr_mean_f1 * 100) if trtr_mean_f1 > 0 else 0.0

        print(f"F1-Score Retention: {utility_retention:.2f}%")
        # print detailed f1 scores
        for name in _CLASSIFIERS.keys():
            trtr_f1 = trtr_results.get(f"{name}_f1", 0.0)
            tstr_f1 = tstr_results.get(f"{name}_f1", 0.0)
            print(f"\t{name}: TRTR F1={trtr_f1:.4f}, TSTR F1={tstr_f1:.4f}")

        metrics: dict = {}
        metrics.update({f"trtr_{k}": v for k, v in trtr_results.items()})
        metrics.update({f"tstr_{k}": v for k, v in tstr_results.items()})
        metrics["trtr_mean_f1"] = trtr_mean_f1
        metrics["tstr_mean_f1"] = tstr_mean_f1
        metrics["utility_retention"] = utility_retention

        return metrics

    def _prepare_splits(self) -> tuple:
        """Splits real data into train/test and extracts the synthetic set."""
        X_real = self.real_df.drop(columns=[self.target_column])
        y_real = self.real_df[self.target_column]

        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_real,
        )

        X_synth = self.synth_df.drop(columns=[self.target_column])
        y_synth = self.synth_df[self.target_column]

        return X_real_train, X_real_test, y_real_train, y_real_test, X_synth, y_synth

    def _build_preprocessor(self, X_reference: pd.DataFrame) -> ColumnTransformer:
        """Returns an unfitted ColumnTransformer inferred from the reference schema."""
        categorical_cols = X_reference.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X_reference.select_dtypes(exclude=["object", "category"]).columns.tolist()

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ],
            remainder="drop",
        )

    def _build_pipeline(self, classifier, preprocessor: ColumnTransformer) -> Pipeline:
        """Clones the preprocessor and classifier into a fresh Scikit-Learn Pipeline."""
        return Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", clone(classifier)),
        ])

    def _evaluate_classifiers(
        self,
        label: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        preprocessor: ColumnTransformer,
    ) -> dict[str, float]:
        """Fits and scores every classifier for one evaluation scenario."""
        results: dict[str, float] = {}

        for name, clf in _CLASSIFIERS.items():
            pipeline = self._build_pipeline(clf, preprocessor)
            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                results[f"{name}_f1"] = float(f1_score(y_test, preds, average="weighted", zero_division=0))
                results[f"{name}_accuracy"] = float(accuracy_score(y_test, preds))
            except Exception as exc:
                logger.warning("[%s] %s failed: %s — recording zero.", label, name, exc)
                results[f"{name}_f1"] = 0.0
                results[f"{name}_accuracy"] = 0.0

        return results