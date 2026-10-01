from pathlib import Path

from typer.testing import CliRunner

from flroad.cli import app

CFG_PATH = str(Path(__file__).resolve().parent.parent / "cfg.yaml")
runner = CliRunner()


def test_help_lists_the_run_command():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "run" in res.output


def test_valid_config_calls_the_simulation(monkeypatch):
    calls = []
    # Replace the real simulation by a function that only records inputs
    monkeypatch.setattr(
        "flroad.main.run", lambda cfg, exp: calls.append((cfg, exp))
    )
    res = runner.invoke(
        app,
        ["run", "--config", CFG_PATH, "--set", "seed=7", "-e", "demo"],
    )
    assert res.exit_code == 0
    assert len(calls) == 1
    cfg, exp = calls[0]
    assert cfg.seed == 7
    assert exp == "demo"


def test_invalid_value_is_reported_cleanly():
    res = runner.invoke(
        app, ["run", "--config", CFG_PATH, "--set", "min_speed_kph=-5"]
    )
    assert res.exit_code == 1
    assert "min_speed_kph" in res.output


def test_missing_file_is_reported_cleanly():
    res = runner.invoke(app, ["run", "--config", "absent.yaml"])
    assert res.exit_code == 1
    assert "not found" in res.output


def test_malformed_override_is_rejected():
    res = runner.invoke(app, ["run", "--config", CFG_PATH, "--set", "seed"])
    assert res.exit_code == 1
    assert "key=value" in res.output


def write_campaign(tmp_path, batches):
    f = tmp_path / "c.yaml"
    f.write_text(
        f"experiment: demo\nbase: {CFG_PATH}\nseeds: [1, 2]\n"
        f"grid:\n  batch: {batches}\n"
    )
    return str(f)


def test_sweep_runs_every_configuration(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "flroad.main.run",
        lambda cfg, exp, name: calls.append((cfg.batch, exp, name)),
    )
    res = runner.invoke(app, ["sweep", write_campaign(tmp_path, "[8, 16]")])
    assert res.exit_code == 0
    assert [c[2] for c in calls] == [
        "batch=8,seed=1",
        "batch=8,seed=2",
        "batch=16,seed=1",
        "batch=16,seed=2",
    ]
    assert calls[0][:2] == (8, "demo")


def test_sweep_validates_everything_before_running(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "flroad.main.run", lambda cfg, exp, name: calls.append(name)
    )
    # batch=0 is invalid but only appears in the second combination
    res = runner.invoke(app, ["sweep", write_campaign(tmp_path, "[16, 0]")])
    assert res.exit_code == 1
    assert "batch" in res.output
    assert calls == []


def test_sweep_reports_a_missing_file():
    res = runner.invoke(app, ["sweep", "absent.yaml"])
    assert res.exit_code == 1
    assert "not found" in res.output
