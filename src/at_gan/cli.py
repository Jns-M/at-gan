"""Typer CLI exposing train, sweep, generate, and evaluate commands for the tabular GAN framework."""

import os

# Suppress TensorFlow logging info and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Suppress the oneDNN custom operations info dump
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml
import typer
from pathlib import Path

app = typer.Typer(
    help="Command Line Interface for the Tabular GAN Training Package.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main():
    """Tabular GAN Framework CLI entry point."""
    pass


@app.command()
def train(
        config_path: Path = typer.Option(
            ...,
            "--config",
            "-c",
            help="Path to the YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        enable_wandb: bool = typer.Option(
            True,
            "--wandb/--no-wandb",
            "-w/-nw",
            help="Enable Weights and Biases tracking for this run.",
        ),
        export_generator: bool = typer.Option(
            True,
            "--export/--no-export",
            "-e/-ne",
            help="Export the standalone generator as a .keras file after training.",
        ),
        generate_samples: int = typer.Option(
            1000,
            "--generate-samples",
            "-g",
            help="Auto-generate N synthetic samples after training and save to CSV. Set to 0 to disable.",
        ),
):
    """Train a GAN model using the specified configuration file and options.

    Args:
        config_path: Path to the YAML configuration file.
        enable_wandb: Whether to enable W&B logging for this run.
        export_generator: Whether to export the trained generator as a ``.keras`` file.
        generate_samples: Number of synthetic rows to auto-generate post-training. ``0`` disables.
    """
    from at_gan.engine.core import GANCoreEngine
    typer.secho(f"Loading configuration from: {config_path}", fg=typer.colors.GREEN)

    engine = GANCoreEngine(
        config_data=config_path,
        enable_wandb=enable_wandb,
        export_generator=export_generator,
        generate_samples=generate_samples,
    )

    typer.secho("[Info] Starting the GAN pipeline engine...", fg=typer.colors.GREEN)
    engine.run_experiment()
    typer.secho("[PASS] Experiment completed successfully!", fg=typer.colors.GREEN, bold=True)


@app.command()
def sweep(
        base_config_path: Path = typer.Option(..., "--base-config", "-c", exists=True),
        sweep_config_path: Path = typer.Option(None, "--sweep-config", "-s"),
        sweep_count: int = typer.Option(30, "--count", "-n", min=1),
        sweep_id: str = typer.Option(None, "--sweep-id", "-id"),
):
    """Run or resume a hyperparameter optimization sweep via Weights & Biases.

    Args:
        base_config_path: Path to the baseline YAML experiment config.
        sweep_config_path: Path to the W&B sweep definition YAML. Required when ``sweep_id`` is ``None``.
        sweep_count: Maximum number of sweep runs to execute.
        sweep_id: Existing W&B sweep ID to resume. If ``None``, a new sweep is created.
    """
    from at_gan.engine.sweeper import GANSweepManager

    typer.secho("[Info] Initializing Sweep Manager...", fg=typer.colors.BLUE)

    try:
        manager = GANSweepManager(
            base_config_path=base_config_path,
            sweep_config_path=sweep_config_path,
            sweep_id=sweep_id,
        )
        typer.secho(f"[Info] Starting WandB Agent for {sweep_count} runs...", fg=typer.colors.GREEN, bold=True)
        manager.execute(count=sweep_count)
        typer.secho("[PASS] Sweep sequence completed!", fg=typer.colors.GREEN, bold=True)

    except Exception as e:
        typer.secho(f"[FAIL] Sweep failed: {str(e)}", fg=typer.colors.RED, bold=True)


@app.command()
def generate(
        config_path: Path = typer.Option(..., "--config", "-c", exists=True),
        run_id: str = typer.Option(..., "--run-id", "-r", help="The W&B run ID (e.g. 'vc8hvxvs') or 'offline_run'."),
        samples: int = typer.Option(1000, "--samples", "-n", help="Number of rows to generate."),
        output: Path = typer.Option(None, "--output", "-o", help="Optional override path for the output CSV."),
):
    """Generate synthetic tabular data using a trained GAN and save to CSV.

    Args:
        config_path: Path to the YAML configuration file used during training.
        run_id: W&B run ID identifying the artifact directory to load the generator from.
        samples: Number of synthetic rows to generate.
        output: Optional output CSV path. Defaults to the experiment's standard synthetic data path.
    """
    from at_gan.engine.synthesizer import GANSynthesizer
    from at_gan.utils.paths import ExperimentPathManager

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

    save_path = output if output else path_manager.synthetic_data_path
    df.to_csv(save_path, index=False)
    typer.secho(f"[Info] Successfully saved {samples} rows to {save_path}", fg=typer.colors.GREEN, bold=True)


@app.command()
def evaluate(
        real_data: Path = typer.Option(..., "--real", "-r", help="Path to the original real CSV.", exists=True),
        synthetic_data: Path = typer.Option(..., "--synthetic", "-s", help="Path to the generated synthetic CSV.", exists=True),
        target: str = typer.Option(None, "--target", "-t", help="Optional target column for TSTR evaluation."),
):
    """Run the complete evaluation suite. TSTR is skipped if no target is provided.

    Args:
        real_data (Path): Path to the original real CSV.
        synthetic_data (Path): Path to the generated synthetic CSV.
        target (str, optional): Optional target column for TSTR evaluation. Defaults to None.
    """
    from at_gan.api import evaluate as api_evaluate

    typer.secho("\n====================================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho("  SYNTHETIC DATA Evaluation SUITE", fg=typer.colors.CYAN, bold=True)
    typer.secho("====================================================================\n", fg=typer.colors.CYAN, bold=True)
    if target:
        typer.secho(f"Target column '{target}' detected. Running full suite including TSTR...", fg=typer.colors.CYAN)
    else:
        typer.secho("No target column provided. Skipping TSTR evaluation...", fg=typer.colors.YELLOW)

    # Delegate all the orchestration to the API
    results = api_evaluate(
        real_data_path=real_data,
        synthetic_data_path=synthetic_data,
        target_column=target
    )

    typer.secho("\n====================================================================", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("  SYNTHETIC DATA Evaluation VERDICT", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("====================================================================", fg=typer.colors.MAGENTA, bold=True)



    # Privacy Evaluation
    min_dcr = results["dcr"]["dcr_min"]
    if round(min_dcr, 4) >= 0.0001:
        typer.secho(f"  [PASS] PRIVACY: SECURE (DCR {min_dcr:.4f} > 0, No Exact Copies)", fg=typer.colors.GREEN)
    else:
        typer.secho(f"  [FAIL] PRIVACY: COMPROMISED (DCR = 0, Exact Copies Detected!)", fg=typer.colors.RED, bold=True)

    # Fidelity Evaluation
    sdv_score = results["sdv"]["sdv_overall_score"] * 100
    if sdv_score >= 85.0:
        typer.secho(f"  [PASS] STATISTIC FIDELITY: {sdv_score:.2f}% SDV Overall Score", fg=typer.colors.GREEN)
    elif sdv_score >= 70.0:
        typer.secho(f"  [WARN] STATISTIC FIDELITY: {sdv_score:.2f}% SDV Overall Score", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"  [FAIL] STATISTIC FIDELITY: {sdv_score:.2f}% SDV Overall Score", fg=typer.colors.RED)

    # Utility Evaluation
    if results["tstr"]:
        utility = results["tstr"]["utility_retention"]
        if utility >= 85.0:
            typer.secho(f"  [PASS] UTILITY RETENTION: {utility:.2f}% F1-Score Retention", fg=typer.colors.GREEN)
        elif utility >= 70.0:
            typer.secho(f"  [WARN] UTILITY RETENTION: {utility:.2f}% F1-Score Retention", fg=typer.colors.YELLOW)
        else:
            typer.secho(f"  [FAIL] UTILITY RETENTION: {utility:.2f}% F1-Score Retention", fg=typer.colors.RED)

    typer.secho("====================================================================", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("====================================================================\n", fg=typer.colors.MAGENTA, bold=True)


if __name__ == "__main__":
    app()