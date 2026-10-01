from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flroad.config import Config, load_config

CFG_PATH = Path(__file__).resolve().parent.parent / "cfg.yaml"


def raw():
    """Load cfg.yaml as a plain dict, to build broken variants of it."""
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def test_repository_config_is_valid():
    cfg = load_config(CFG_PATH)
    assert cfg.n_sub_classes == 8
    assert cfg.min_cpu_hertz == 1e9


def test_unknown_key_is_rejected():
    d = raw()
    d["learning_rat"] = 0.1  # typo
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_negative_speed_is_rejected():
    d = raw()
    d["min_speed_kph"] = -5
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_min_above_max_is_rejected():
    d = raw()
    d["min_speed_kph"], d["max_speed_kph"] = 60, 30
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_too_many_classes_is_rejected():
    d = raw()
    d["n_sub_classes"] = 11
    with pytest.raises(ValidationError):
        Config.model_validate(d)
