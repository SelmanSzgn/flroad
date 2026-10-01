import csv
from datetime import datetime
from pathlib import Path

import yaml


def make_run_dir(cfg, root="runs"):
    """Create and return a new folder for one run."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{stamp}-seed{cfg.seed}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_config(cfg, run_dir):
    """Write the exact configuration of the run in run_dir."""
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)


FIELDS = [
    "round", "time_s", "active", "dropped", "acc", "loss", "energy_j",
    "client_acc_min", "client_acc_mean",
]


class MetricsWriter:
    """Append one row of metrics per round to a CSV file."""

    def __init__(self, run_dir):
        self.path = run_dir / "metrics.csv"
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def write(self, row):
        """Append a row, a dict whose keys are listed in FIELDS."""
        row = {k: round(v, 4) if isinstance(v, float) else v
               for k, v in row.items()}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
