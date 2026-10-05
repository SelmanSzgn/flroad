from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flroad.config import Config, load_config, parse_overrides

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


def test_parse_overrides_converts_types():
    out = parse_overrides(["seed=7", "learning_rate=0.5"])
    assert out == {"seed": 7, "learning_rate": 0.5}


def test_override_is_applied():
    cfg = load_config(CFG_PATH, {"seed": 7})
    assert cfg.seed == 7


def test_unknown_override_is_rejected():
    with pytest.raises(ValidationError):
        load_config(CFG_PATH, {"learning_rat": 0.1})


def test_speed_parameters_default_to_constant_speed():
    d = raw()
    for k in ("speed_alpha", "speed_std_kph", "speed_step_s"):
        d.pop(k)
    assert Config.model_validate(d).speed_alpha == 1.0


def test_speed_alpha_above_one_is_rejected():
    d = raw()
    d["speed_alpha"] = 1.5
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_path_loss_defaults_to_no_distance_effect():
    d = raw()
    for k in ("path_loss_exp", "bs_offset_m", "ref_distance_m"):
        d.pop(k)
    assert Config.model_validate(d).path_loss_exp == 0.0


def test_negative_path_loss_exponent_is_rejected():
    d = raw()
    d["path_loss_exp"] = -1
    with pytest.raises(ValidationError):
        Config.model_validate(d)
