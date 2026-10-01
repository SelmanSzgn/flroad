import os

import mlflow

from flroad.config import Config

DEFAULT_URI = "sqlite:///mlflow.db"


def setup_tracking(experiment: str) -> None:
    """Point MLflow to its database and select the experiment."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def log_config(cfg: Config) -> None:
    """Log every configuration value as a parameter of the active run."""
    mlflow.log_params(cfg.model_dump())


def log_round(row: dict[str, float | int], step: int) -> None:
    """Log the numeric metrics of one round at the given step."""
    metrics = {
        k: float(v) for k, v in row.items() if k not in ("round", "time_s")
    }
    mlflow.log_metrics(metrics, step=step)
