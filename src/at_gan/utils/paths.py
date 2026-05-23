from pathlib import Path


class ExperimentPathManager:
    def __init__(
            self,
            base_output_path: str | Path,
            experiment_name: str,
            run_id: str,
            scalers_dir_name: str = "scalers",
            checkpoints_dir_name: str = "checkpoints",
            latest_generator_name: str = "latest_generator.keras",
            best_generator_name: str = "best_generator.keras",
            synthetic_data_name: str = "synthetic_data.csv"
    ):
        self.base_output_path = Path(base_output_path)
        self.experiment_name = experiment_name
        self.run_id = run_id

        # Store the directory and file names
        self._scalers_dir_name = scalers_dir_name
        self._checkpoints_dir_name = checkpoints_dir_name
        self._latest_generator_name = latest_generator_name
        self._best_generator_name = best_generator_name
        self._synthetic_data_name = synthetic_data_name

        self.run_dir = self.base_output_path / self.experiment_name / self.run_id

    @property
    def scalers_dir(self) -> Path:
        return self.run_dir / self._scalers_dir_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / self._checkpoints_dir_name

    @property
    def best_checkpoints_dir(self) -> Path:
        return self.checkpoints_dir / "best"

    @property
    def latest_checkpoints_dir(self) -> Path:
        return self.checkpoints_dir / "latest"

    @property
    def latest_generator_path(self) -> Path:
        return self.run_dir / self._latest_generator_name

    @property
    def best_generator_path(self) -> Path:
        return self.run_dir / self._best_generator_name

    @property
    def synthetic_data_path(self) -> Path:
        return self.run_dir / self._synthetic_data_name

    def create_dirs(self) -> None:
        """Creates the foundational directory structure for a new experiment run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scalers_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoints_dir.mkdir(parents=True, exist_ok=True)