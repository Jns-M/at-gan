"""W&B sweep manager with dynamic config merging, TTUR enforcement, and NAS architecture building."""

import copy
import random
import yaml
import wandb
from pathlib import Path

from at_gan.engine.core import GANCoreEngine
from at_gan.utils.logger import get_logger

logger = get_logger(__name__)

class GANSweepManager:
    """Manages W&B sweep lifecycle: creation or resumption, per-run config injection, and agent dispatch."""

    def __init__(
            self,
            base_config_path: Path | str,
            sweep_config_path: Path | str | None = None,
            sweep_id: str | None = None,
    ):
        """Loads the base config and prepares sweep identifiers.

        Args:
            base_config_path: Path to the baseline YAML experiment config.
            sweep_config_path: Path to the W&B sweep definition YAML. Required when ``sweep_id`` is ``None``.
            sweep_id: Existing W&B sweep ID to resume. If ``None``, a new sweep is created.

        Raises:
            FileNotFoundError: If ``base_config_path`` does not exist.
        """
        self.base_config_path = Path(base_config_path)
        self.sweep_config_path = Path(sweep_config_path) if sweep_config_path else None
        self.sweep_id = sweep_id

        with open(self.base_config_path, "r") as file:
            self.base_config = yaml.safe_load(file)

        self.project_name = self.base_config.get("project_name", "at-gan-framework")

    def _setup_sweep(self) -> None:
        """Creates a new W&B sweep from the sweep config or attaches to an existing one by ID.

        Raises:
            ValueError: If ``sweep_id`` is ``None`` and no ``sweep_config_path`` was provided.
        """
        if self.sweep_id is None:
            if not self.sweep_config_path:
                raise ValueError("You must provide a sweep_config_path if no sweep_id is provided.")
            with open(self.sweep_config_path, "r") as file:
                sweep_config = yaml.safe_load(file)
            self.sweep_id = wandb.sweep(sweep=sweep_config, project=self.project_name)
            logger.info(f"Initialized NEW WandB Sweep: {self.sweep_id}")
        else:
            logger.info(f"Resuming EXISTING WandB Sweep: {self.sweep_id}")

    def _sweep_train_wrapper(self) -> None:
        """Agent callback executed by W&B for each sweep run; merges sweep params and launches the engine."""
        with wandb.init() as _:
            # Deepcopy prevents sweep parameter bleed-over between consecutive runs
            current_config = copy.deepcopy(self.base_config)

            # Merge dot-namespaced W&B sweep parameters into the nested config dict
            for key, value in wandb.config.items():
                if "." in key:
                    parent, child = key.split(".", 1)
                    if parent in ["generator", "discriminator"]:
                        current_config["model"][parent][child] = value
                    elif parent == "model":
                        current_config["model"][child] = value
                    elif parent == "data":
                        current_config["data"][child] = value
                    elif parent == "training":
                        current_config["training"][child] = value
                else:
                    # Flat keys without a namespace prefix fall back to the training block
                    if key in current_config.get("training", {}):
                        current_config["training"][key] = value
                    else:
                        current_config[key] = value

            # TTUR: clamp d_lr to g_lr to prevent the discriminator from dominating the generator
            g_lr = current_config["training"]["g_lr"]
            d_lr = current_config["training"]["d_lr"]
            if d_lr > g_lr:
                current_config["training"]["d_lr"] = g_lr
                wandb.config.update({"d_lr": g_lr}, allow_val_change=True)
                logger.info(f"TTUR Enforced: Clamped d_lr down to match g_lr ({g_lr})")

            # NAS builder: translate shape + base_units into a concrete units list per network
            for net in ["generator", "discriminator"]:
                net_config = current_config["model"][net]
                if "architecture_shape" in net_config:
                    shape = net_config["architecture_shape"]
                    base_units = net_config.get("base_units", 128)
                    num_layers = net_config.get("num_hidden_layers", 2)
                    max_units = net_config.get("max_units", 512)

                    units_list = []
                    current_size = base_units

                    for _ in range(num_layers):
                        units_list.append(current_size)
                        if shape == "descending":
                            # Stochastic halving introduces architectural diversity within the sweep
                            current_size = max(8, current_size // random.choice([1, 2]))
                        elif shape == "ascending":
                            # Stochastic doubling introduces architectural diversity within the sweep
                            current_size = min(max_units, current_size * random.choice([1, 2]))

                    net_config["units"] = units_list
                    wandb.config.update({f"{net}.actual_units": units_list}, allow_val_change=True)

            logger.info(f"Launching Engine with config: {wandb.config}")

            engine = GANCoreEngine(
                config_data=current_config,
                enable_wandb=True,
                export_generator=False,
                generate_samples=2000,
            )
            engine.run_experiment()

    def execute(self, count: int = 30) -> None:
        """Sets up the sweep and launches the W&B agent for the specified number of runs.

        Args:
            count: Maximum number of sweep runs to execute.
        """
        self._setup_sweep()
        wandb.agent(self.sweep_id, project=self.project_name, function=self._sweep_train_wrapper, count=count)