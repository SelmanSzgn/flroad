from pathlib import Path

from flroad.tracking import log_round, log_run_files


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
