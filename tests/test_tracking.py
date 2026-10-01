from pathlib import Path

import torch.nn as nn

from flroad.tracking import (
    load_tracked_model,
    log_model,
    log_round,
    log_run_files,
)


def test_log_round_drops_non_metrics_and_casts_to_float(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "flroad.tracking.mlflow.log_metrics",
        lambda metrics, step: seen.update(metrics=metrics, step=step),
    )
    log_round({"round": 3, "time_s": 120, "acc": 50, "active": 2}, step=3)
    assert seen["step"] == 3
    assert seen["metrics"] == {"acc": 50.0, "active": 2.0}
    assert all(isinstance(v, float) for v in seen["metrics"].values())


def test_log_run_files_uploads_the_run_folder(monkeypatch):
    seen = []
    monkeypatch.setattr("flroad.tracking.mlflow.log_artifacts", seen.append)
    log_run_files(Path("runs/abc"))
    assert seen == ["runs/abc"]


def test_log_model_logs_a_cpu_copy(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "flroad.tracking.mlflow.pytorch.log_model",
        lambda model, name, input_example, registered_model_name: seen.update(
            model=model,
            name=name,
            example=input_example,
            registered=registered_model_name,
        ),
    )
    original = nn.Linear(2, 1)
    log_model(original)
    assert seen["name"] == "model"
    assert seen["registered"] == "flroad"
    assert seen["model"] is not original
    assert seen["model"].weight.device.type == "cpu"
    assert seen["example"].shape == (1, 3, 32, 32)


def test_load_tracked_model_forwards_the_uri(monkeypatch):
    monkeypatch.setattr(
        "flroad.tracking.mlflow.set_tracking_uri", lambda u: None
    )
    monkeypatch.setattr(
        "flroad.tracking.mlflow.pytorch.load_model", lambda uri: ("loaded", uri)
    )
    assert load_tracked_model("models:/flroad/3") == (
        "loaded",
        "models:/flroad/3",
    )
