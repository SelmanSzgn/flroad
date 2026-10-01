import copy
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch.nn as nn

from flroad.config import Config

DEFAULT_URI = "sqlite:///mlflow.db"


def log_config(cfg: Config) -> None:
    """Log every configuration value as a parameter of the active run."""
    mlflow.log_params(cfg.model_dump())


def log_round(row: dict[str, float | int], step: int) -> None:
    """Log the numeric metrics of one round at the given step."""
    metrics = {
        k: float(v) for k, v in row.items() if k not in ("round", "time_s")
    }
    mlflow.log_metrics(metrics, step=step)


def log_run_files(run_dir: Path) -> None:
    """Upload every file of the run folder as artifacts of the active run."""
    mlflow.log_artifacts(str(run_dir))


def log_model(model: nn.Module) -> None:
    """Log the model in MLflow's PyTorch format (always on CPU)."""
    # One fake CIFAR-10 image: batch of 1, 3 channels, 32 x 32 pixels
    example = np.zeros((1, 3, 32, 32), dtype=np.float32)
    mlflow.pytorch.log_model(
        copy.deepcopy(model).cpu(),
        name="model",
        input_example=example,
        registered_model_name="flroad",
    )


def set_tracking_uri() -> None:
    """Point MLflow to its database (env variable or default)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)
    mlflow.set_tracking_uri(uri)


def setup_tracking(experiment: str) -> None:
    """Point MLflow to its database and select the experiment."""
    set_tracking_uri()
    mlflow.set_experiment(experiment)


def load_tracked_model(uri: str) -> nn.Module:
    """Reload a model from a MLflow URI (runs:/... or models:/...)."""
    set_tracking_uri()
    return mlflow.pytorch.load_model(uri)
