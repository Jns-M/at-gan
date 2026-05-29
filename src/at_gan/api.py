"""Public Python API for training, sweeping, generating, and evaluating the tabular GAN."""

import os

# Suppress TensorFlow logging info and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Suppress the oneDNN custom operations info dump
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import pandas as pd
import yaml

from at_gan.engine.core import GANCoreEngine
from at_gan.engine.sweeper import GANSweepManager
from at_gan.engine.synthesizer import GANSynthesizer
from at_gan.eval.dcr import DCREvaluator
from at_gan.eval.sdv import SDVEvaluator
from at_gan.eval.tstr import TSTREvaluator
from at_gan.eval.plots import GANEvaluationPlotter
from at_gan.utils.paths import ExperimentPathManager


def train(
        config: str | Path | dict,
        enable_wandb: bool = True,
        export_generator: bool = True,
        generate_samples: int = 1000,
) -> GANCoreEngine:
    """Launches a full GAN training experiment and returns the trained engine.

    Args:
        config: Path to a YAML config file or a pre-parsed configuration ``dict``.
        enable_wandb: Whether to enable W&B logging.
        export_generator: Whether to save generator ``.keras`` files to disk after training.
        generate_samples: Number of synthetic rows to auto-generate post-training. ``0`` disables.

    Returns:
        The initialized and trained ``GANCoreEngine`` instance.
    """
    engine = GANCoreEngine(
        config_data=config,
        enable_wandb=enable_wandb,
        export_generator=export_generator,
        generate_samples=generate_samples,
    )
    engine.run_experiment()
    return engine


def sweep(
        base_config: str | Path,
        sweep_config: str | Path | None = None,
        sweep_id: str | None = None,
        count: int = 30,
) -> str:
    """Launches or resumes a W&B hyperparameter sweep and returns the sweep ID.

    Args:
        base_config: Path to the baseline YAML experiment config.
        sweep_config: Path to the W&B sweep definition YAML. Required when ``sweep_id`` is ``None``.
        sweep_id: Existing W&B sweep ID to resume. If ``None``, a new sweep is created.
        count: Maximum number of sweep runs to execute.

    Returns:
        The W&B sweep ID used for this run.
    """
    manager = GANSweepManager(
        base_config_path=base_config,
        sweep_config_path=sweep_config,
        sweep_id=sweep_id,
    )
    manager.execute(count=count)
    return manager.sweep_id


def generate(
        config_path: str | Path,
        run_id: str,
        samples: int = 1000,
        output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Loads a trained generator and produces a ``DataFrame`` of synthetic tabular rows.

    Args:
        config_path: Path to the YAML configuration file used during training.
        run_id: W&B run ID (or ``"offline_run"``) identifying which artifact directory to load from.
        samples: Number of synthetic rows to generate.
        output_path: Optional path to save the result as a CSV. If ``None``, no file is written.

    Returns:
        ``DataFrame`` of inverse-transformed synthetic data in original feature space.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    base_output = config["data"]["output_path"]
    experiment_name = config.get("experiment_name", "default_experiment")

    path_manager = ExperimentPathManager(base_output, experiment_name, run_id)

    synthesizer = GANSynthesizer(
        generator_path=path_manager.best_generator_path,
        scaler_dir=path_manager.scalers_dir,
        config=config,
    )

    df = synthesizer.sample(num_samples=samples)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def evaluate(
        real_data_path: str | Path,
        synthetic_data_path: str | Path,
        target_column: str | None = None
) -> dict:
    """Runs the full benchmark suite (SDV, DCR, and conditionally TSTR).

    Args:
        real_data_path: Path to the ground-truth CSV.
        synthetic_data_path: Path to the GAN-generated CSV.
        target_column: Optional target column name. If provided, TSTR is also run.

    Returns:
        Dict containing nested results for ``sdv``, ``dcr``, and ``tstr``.
    """
    real_df = pd.read_csv(real_data_path)
    synth_df = pd.read_csv(synthetic_data_path)

    real_df.columns = real_df.columns.str.lower()
    synth_df.columns = synth_df.columns.str.lower()

    if target_column:
        target_column = target_column.lower()
        if target_column not in real_df.columns:
            raise ValueError(f"Target column '{target_column}' missing from real data.")
        if target_column not in synth_df.columns:
            raise ValueError(f"Target column '{target_column}' missing from synthetic data.")

    # Validate and align schemas globally before running any evaluators
    real_features = set(real_df.columns)
    synth_features = set(synth_df.columns)

    if target_column:
        real_features -= {target_column}
        synth_features -= {target_column}

    shared = sorted(real_features & synth_features)

    keep_cols = shared
    if target_column:
        keep_cols.append(target_column)

    real_df = real_df[keep_cols]
    synth_df = synth_df[keep_cols]

    results = {}

    dcr_eval = DCREvaluator(real_df=real_df, synth_df=synth_df, target_column=target_column)
    results["dcr"] = dcr_eval.run_evaluation()

    sdv_eval = SDVEvaluator(real_df=real_df, synth_df=synth_df)
    results["sdv"] = sdv_eval.run_evaluation()

    if target_column:
        tstr_eval = TSTREvaluator(
            real_df=real_df,
            synth_df=synth_df,
            target_column=target_column,
        )
        results["tstr"] = tstr_eval.run_evaluation()
    else:
        results["tstr"] = None

    plotter = GANEvaluationPlotter(real_df=real_df, synth_df=synth_df, dpi=300)
    results["plotter"] = plotter

    return results