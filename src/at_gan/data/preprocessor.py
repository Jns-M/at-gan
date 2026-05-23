"""Scikit-Learn-compatible preprocessor for mixed-type tabular GAN data."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from at_gan.utils.logger import get_logger

logger = get_logger(__name__)


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Handles fit/transform/inverse_transform for continuous, binary, discrete-count, and categorical columns."""

    def __init__(
            self,
            continuous_cols: list[str] | None = None,
            discrete_count_cols: list[str] | None = None,
            binary_cols: list[str] | None = None,
            categorical_cols: list[str] | None = None,
            binary_noise: bool = True,
            noise_lower_bound: float = 0.0001,
            noise_upper_bound: float = 0.25,
            treat_bin_as_cat: bool = False,
            beta_noise: bool = True,
            smooth_categorical: bool = False,
    ):
        """Configures column routing and instantiates scalers/encoders.

        Args:
            continuous_cols: Column names to scale to ``[-1, 1]`` via MinMax.
            discrete_count_cols: Non-negative integer columns to scale to ``[0, 1]``.
            binary_cols: Binary (0/1) columns to receive noise augmentation.
            categorical_cols: Nominal columns to one-hot encode.
            binary_noise: Whether to apply noise to binary columns during transform.
            noise_lower_bound: Lower bound of the uniform noise range.
            noise_upper_bound: Upper bound of the uniform noise range.
            treat_bin_as_cat: If ``True``, routes binary columns through the OHE pipeline instead.
            beta_noise: If ``True``, uses Beta-distributed noise instead of uniform for binary cols.
            smooth_categorical: If ``True``, applies label-preserving noise to OHE outputs.
        """
        self.cat_feature_names_ = None
        self.continuous_cols = [c.lower() for c in (continuous_cols or [])]
        self.discrete_count_cols = [c.lower() for c in (discrete_count_cols or [])]
        self.binary_cols = [c.lower() for c in (binary_cols or [])]
        self.categorical_cols = [c.lower() for c in (categorical_cols or [])]
        self.binary_noise = binary_noise
        self.noise_lower_bound = noise_lower_bound
        self.noise_upper_bound = noise_upper_bound
        self.treat_bin_as_cat = treat_bin_as_cat
        self.beta_noise = beta_noise
        self.smooth_categorical = smooth_categorical
        self.continuous_precisions_ = {}

        # Route binary columns through OHE if treat_bin_as_cat is set
        if self.treat_bin_as_cat:
            self.categorical_cols.extend(self.binary_cols)
            self.binary_cols = []

        self.continuous_scaler = MinMaxScaler(feature_range=(-1, 1)) if self.continuous_cols else None
        self.discrete_count_scaler = MinMaxScaler(feature_range=(0, 1)) if self.discrete_count_cols else None
        self.categorical_encoder = (
            OneHotEncoder(sparse_output=False, handle_unknown="ignore") if self.categorical_cols else None
        )

    @property
    def continuous_dim(self) -> int:
        """Number of continuous features."""
        return len(self.continuous_cols)

    @property
    def binary_dim(self) -> int:
        """Number of binary features."""
        return len(self.binary_cols)

    @property
    def discrete_count_dim(self) -> int:
        """Number of discrete-count features."""
        return len(self.discrete_count_cols)

    @property
    def categorical_dims(self) -> list[int]:
        """Per-feature cardinality list for categorical columns after fitting.

        Returns:
            List of category counts, one per categorical column. Empty if not fitted.
        """
        if self.categorical_encoder and hasattr(self.categorical_encoder, "categories_"):
            return [len(cats) for cats in self.categorical_encoder.categories_]
        return []

    @property
    def categorical_dim(self) -> int:
        """Total one-hot encoded width across all categorical columns."""
        return sum(self.categorical_dims)

    @property
    def total_dim(self) -> int:
        """Total feature dimension of the transformed output vector."""
        return self.continuous_dim + self.binary_dim + self.discrete_count_dim + self.categorical_dim

    def fit(self, X: pd.DataFrame, y=None) -> "TabularPreprocessor":
        """Fits all scalers and encoders on ``X``.

        Args:
            X: Raw input DataFrame with original column names.
            y: Unused. Present for Scikit-Learn API compatibility.

        Returns:
            The fitted ``TabularPreprocessor`` instance.
        """
        X_lower = X.copy()
        X_lower.columns = X_lower.columns.str.lower()

        if self.continuous_scaler:
            self.continuous_scaler.fit(X_lower[self.continuous_cols])
            # Capture per-column decimal precision from training data for use in inverse_transform
            for col in self.continuous_cols:
                self.continuous_precisions_[col] = self._calculate_max_decimals(X_lower[col])

        if self.discrete_count_scaler:
            self.discrete_count_scaler.fit(X_lower[self.discrete_count_cols])

        if self.categorical_encoder:
            self.categorical_encoder.fit(X_lower[self.categorical_cols])
            self.cat_feature_names_ = self.categorical_encoder.get_feature_names_out(self.categorical_cols)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scales and encodes ``X`` into the GAN's input representation.

        Args:
            X: Raw input DataFrame. Column names are lowercased internally.

        Returns:
            Transformed ``DataFrame`` in the column order defined by ``get_feature_names_out``.
        """
        data_scaled = X.copy()
        data_scaled.columns = data_scaled.columns.str.lower()

        required_cols = self.binary_cols + self.continuous_cols + self.discrete_count_cols + self.categorical_cols
        data_scaled = data_scaled.loc[:, required_cols]

        if self.continuous_scaler:
            data_scaled[self.continuous_cols] = self.continuous_scaler.transform(data_scaled[self.continuous_cols])

        if self.discrete_count_scaler:
            data_scaled[self.discrete_count_cols] = self.discrete_count_scaler.transform(
                data_scaled[self.discrete_count_cols]
            )

        if self.categorical_encoder:
            encoded_cats = self.categorical_encoder.transform(data_scaled[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded_cats, columns=self.cat_feature_names_, index=data_scaled.index)
            data_scaled = pd.concat([data_scaled.drop(columns=self.categorical_cols), encoded_df], axis=1)

            if self.smooth_categorical:
                for i, original_col in enumerate(self.categorical_cols):
                    cats = self.categorical_encoder.categories_[i]
                    group_cols = [f"{original_col}_{cat}" for cat in cats]
                    data_scaled = self._smooth_categorical(data_scaled, group_cols)

        if self.beta_noise:
            data_scaled = self._add_strict_beta_noise(data_scaled, self.binary_cols)
        else:
            data_scaled = self._add_noise(data_scaled, self.binary_cols)

        report = (
            f"\n--- Data Scaling Summary ---\n"
            f"Shape: {data_scaled.shape}\n"
            f"Columns: {list(data_scaled.columns)}\n"
            f"Missing Values: {(data_scaled.isnull().sum() / data_scaled.shape[0]).max():.2%}"
        )
        logger.info(report)

        return data_scaled

    def inverse_transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Reverses scaling and encoding to recover human-readable feature values.

        Args:
            X: Transformed data as a ``DataFrame`` or ``numpy`` array.

        Returns:
            ``DataFrame`` with original-scale values, discrete types, and decoded categories.
        """
        if isinstance(X, np.ndarray):
            data_restored = pd.DataFrame(X, columns=self.get_feature_names_out())
        else:
            data_restored = X.copy()

        if self.continuous_scaler:
            data_restored[self.continuous_cols] = self.continuous_scaler.inverse_transform(
                data_restored[self.continuous_cols]
            )
            # Round each column to its original precision
            for col in self.continuous_cols:
                precision = self.continuous_precisions_.get(col, 4)
                data_restored[col] = data_restored[col].round(precision)
                if precision == 0:
                    data_restored[col] = data_restored[col].astype(int)

        if self.discrete_count_scaler:
            data_restored[self.discrete_count_cols] = self.discrete_count_scaler.inverse_transform(
                data_restored[self.discrete_count_cols]
            )
            for col in self.discrete_count_cols:
                # Threshold at 0.9 to snap near-zero predictions to exactly 0 before rounding
                data_restored[col] = np.where(data_restored[col] < 0.9, 0.0, data_restored[col])
                data_restored[col] = data_restored[col].round().astype(int)
                # Guard against rare negative values from floating-point rounding errors
                data_restored[col] = np.maximum(data_restored[col], 0)

        if self.categorical_encoder:
            encoded_data = data_restored[self.cat_feature_names_]
            decoded_cats = self.categorical_encoder.inverse_transform(encoded_data)
            decoded_df = pd.DataFrame(decoded_cats, columns=self.categorical_cols, index=data_restored.index)
            data_restored = pd.concat(
                [data_restored.drop(columns=self.cat_feature_names_), decoded_df], axis=1
            )

        if self.binary_cols:
            for col in self.binary_cols:
                data_restored[col] = (data_restored[col] >= 0.5).astype(int)

        return data_restored

    def clean_re_encode(self, data_restored: pd.DataFrame) -> pd.DataFrame:
        """Re-applies scaling and OHE to an inverse-transformed DataFrame without adding noise.

        Used to evaluate the GAN's true discretized output without stochastic augmentation.

        Args:
            data_restored: DataFrame in original (human-readable) value space.

        Returns:
            Noise-free scaled ``DataFrame`` in the exact column order expected by the network.
        """
        data_clean = data_restored.copy()

        if self.continuous_scaler:
            data_clean[self.continuous_cols] = self.continuous_scaler.transform(data_clean[self.continuous_cols])

        if self.discrete_count_scaler:
            data_clean[self.discrete_count_cols] = self.discrete_count_scaler.transform(
                data_clean[self.discrete_count_cols]
            )

        if self.categorical_encoder:
            encoded_cats = self.categorical_encoder.transform(data_clean[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded_cats, columns=self.cat_feature_names_, index=data_clean.index)
            data_clean = pd.concat([data_clean.drop(columns=self.categorical_cols), encoded_df], axis=1)

        # Enforce exact column order to match the generator's output layout
        return data_clean[self.get_feature_names_out()]

    def _calculate_max_decimals(self, series: pd.Series) -> int:
        """Finds the maximum number of significant decimal places present in a numeric series.

        Args:
            series: A ``pd.Series`` of numeric values.

        Returns:
            Integer maximum decimal precision observed across all non-null values.
        """

        def count_decimals(val) -> int:
            if pd.isna(val):
                return 0
            val_str = str(val).lower()
            if "e-" in val_str:
                # Exponent directly encodes the number of decimal places in scientific notation
                return int(val_str.split("e-")[1])
            if "." in val_str:
                # Strip trailing zeros to avoid Python float repr artifacts (e.g. ``45.0`` → ``45.``)
                return len(val_str.split(".")[1].rstrip("0"))
            return 0

        return int(series.apply(count_decimals).max())

    def _add_noise(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Applies uniform random noise to binary columns, pushing values away from hard 0/1 boundaries.

        Args:
            df: Input DataFrame containing binary columns.
            cols: List of binary column names to augment.

        Returns:
            DataFrame with noise-augmented binary columns.
        """
        if not self.binary_noise or not cols:
            return df

        df = df.copy()
        for col in cols:
            noise = np.random.uniform(self.noise_lower_bound, self.noise_upper_bound, size=len(df))
            is_zero = df[col] == 0
            df[col] = np.where(is_zero, df[col] + noise, df[col] - noise)

        return df

    def _smooth_categorical(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Applies label-preserving Dirichlet-style noise to a one-hot encoded feature group.

        Noise is computed and distributed such that the argmax label is never flipped.

        Args:
            df: DataFrame containing the OHE feature group columns.
            feature_cols: Column names belonging to a single categorical feature's OHE group.

        Returns:
            DataFrame with smoothed OHE columns.
        """
        k = len(feature_cols)
        if k <= 1:
            return df

        # Maximum noise that preserves the argmax: active class stays above all others
        max_safe_noise = ((k - 1) / k) - 0.01

        safe_upper = min(self.noise_upper_bound, max_safe_noise)
        safe_lower = min(self.noise_lower_bound, safe_upper)

        noise = np.random.uniform(safe_lower, safe_upper, size=(len(df), 1))
        vals = df[feature_cols].values
        is_one = vals == 1.0

        vals = np.where(is_one, vals - noise, vals)
        # Distribute deducted noise evenly across all inactive categories to preserve row sum
        vals = np.where(~is_one, vals + (noise / (k - 1)), vals)

        df.loc[:, feature_cols] = vals
        return df

    def _add_strict_beta_noise(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Applies Beta-distributed noise to binary columns, keeping values strictly separated from 0.5.

        Args:
            df: Input DataFrame containing binary columns.
            cols: List of binary column names to augment.

        Returns:
            DataFrame where binary columns are replaced with Beta-sampled values in ``[0, 0.49]`` or ``[0.51, 1.0]``.
        """
        if not self.binary_noise or not cols:
            return df

        df = df.copy()
        for col in cols:
            is_zero = df[col] == 0

            # Beta(1,5) is right-skewed toward 0; Beta(5,1) is left-skewed toward 1
            raw_zeros = np.random.beta(1, 5, size=len(df))
            raw_ones = np.random.beta(5, 1, size=len(df))

            # Scale to [0.0, 0.49] and [0.51, 1.0] to enforce a hard decision boundary at 0.5
            noise_for_zeros = raw_zeros * 0.49
            noise_for_ones = (raw_ones * 0.49) + 0.51

            df[col] = np.where(is_zero, noise_for_zeros, noise_for_ones)

        return df

    def get_feature_names_out(self) -> list[str]:
        """Returns the canonical column order produced by ``transform``.

        Returns:
            Ordered list of output column names: binary, continuous, discrete-count, then OHE categoricals.
        """
        cols = self.binary_cols + self.continuous_cols + self.discrete_count_cols
        if self.cat_feature_names_ is not None:
            cols.extend(list(self.cat_feature_names_))
        return cols

    def dump_scalers(self, output_dir: str | Path) -> None:
        """Serializes all fitted scalers and encoders to disk using joblib.

        Args:
            output_dir: Target directory. Created if it does not exist. No-op if falsy.
        """
        if not output_dir:
            return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.continuous_scaler:
            dump(self.continuous_scaler, out_path / "continuous_scaler.bin", compress=True)
            dump(self.continuous_precisions_, out_path / "continuous_precisions.bin", compress=True)
        if self.discrete_count_scaler:
            dump(self.discrete_count_scaler, out_path / "discrete_count_scaler.bin", compress=True)
        if self.categorical_encoder:
            dump(self.categorical_encoder, out_path / "categorical_encoder.bin", compress=True)

    def load_scalers(self, input_dir: str | Path) -> None:
        """Deserializes and restores fitted scalers and encoders from disk.

        Args:
            input_dir: Directory previously populated by ``dump_scalers``.
        """
        in_path = Path(input_dir)

        if (in_path / "continuous_scaler.bin").exists():
            self.continuous_scaler = load(in_path / "continuous_scaler.bin")
            precisions_path = in_path / "continuous_precisions.bin"
            if precisions_path.exists():
                self.continuous_precisions_ = load(precisions_path)

        if (in_path / "discrete_count_scaler.bin").exists():
            self.discrete_count_scaler = load(in_path / "discrete_count_scaler.bin")

        if (in_path / "categorical_encoder.bin").exists():
            self.categorical_encoder = load(in_path / "categorical_encoder.bin")
            self.cat_feature_names_ = self.categorical_encoder.get_feature_names_out(self.categorical_cols)

    def summarize_data(self, df: pd.DataFrame) -> None:
        """Logs a descriptive statistics summary of the given DataFrame.

        Args:
            df: Any DataFrame to summarize, typically the post-transform training set.
        """
        logger.info(f"\nData Summary:\n{df.describe()}")