from pathlib import Path

import torch

from flroad.config import load_config
from flroad.runs import (
    MetricsWriter,
    load_model,
    make_run_dir,
    save_config,
    save_model,
)

CFG_PATH = Path(__file__).resolve().parent.parent / "cfg.yaml"


def test_run_dir_is_created_under_root(tmp_path):
    cfg = load_config(CFG_PATH)
    run_dir = make_run_dir(cfg, tmp_path)
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path


def test_run_dir_name_contains_the_seed(tmp_path):
    cfg = load_config(CFG_PATH, {"seed": 7})
    assert "seed7" in make_run_dir(cfg, tmp_path).name


def test_saved_config_can_be_loaded_back(tmp_path):
    cfg = load_config(CFG_PATH, {"seed": 7})
    run_dir = make_run_dir(cfg, tmp_path)
    save_config(cfg, run_dir)
    assert load_config(run_dir / "config.yaml") == cfg


def test_metrics_writer_writes_header_and_rows(tmp_path):
    w = MetricsWriter(tmp_path)
    w.write({"round": 0, "acc": 10.123456})
    w.write({"round": 1, "acc": 20.0, "client_acc_min": 5.0})
    lines = (tmp_path / "metrics.csv").read_text().splitlines()
    assert lines[0].startswith("round,time_s,active")
    assert len(lines) == 3
    assert lines[1].split(",")[4] == "10.1235"


def test_saved_model_can_be_reloaded(tmp_path):
    from flroad.model import Model

    m = Model()
    save_model(m, tmp_path)
    m2 = load_model(tmp_path)
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
    assert not m2.training
