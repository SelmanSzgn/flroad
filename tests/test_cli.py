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
