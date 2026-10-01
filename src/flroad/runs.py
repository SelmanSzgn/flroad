import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from flroad.config import Config

FIELDS = [
    "round",
    "time_s",
    "active",
    "dropped",
    "acc",
    "loss",
    "energy_j",
    "client_acc_min",
    "client_acc_mean",
]


def make_run_dir(cfg: Config, root: str | Path = "runs") -> Path:
    """Create and return a new folder for one run."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{stamp}-seed{cfg.seed}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_config(cfg: Config, run_dir: Path) -> None:
    """Write the exact configuration of the run in run_dir."""
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)


class MetricsWriter:
    """Append one row of metrics per round to a CSV file."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "metrics.csv"
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def write(self, row: dict[str, float | int]) -> None:
        """Append a row, a dict whose keys are listed in FIELDS."""
        row = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in row.items()
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


def save_model(model: nn.Module, run_dir: Path) -> None:
    """Save the weights of the model in run_dir/model.pt."""
    torch.save(model.state_dict(), run_dir / "model.pt")


def load_model(run_dir: Path, device: str | torch.device = "cpu") -> nn.Module:
    """Rebuild a Model and load the weights saved in run_dir."""
    from flroad.model import Model

    model = Model()
    state = torch.load(
        run_dir / "model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    return model.eval()
