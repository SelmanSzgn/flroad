import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    """All simulation parameters, validated at load time."""

    # Reject unknown keys (typos) and forbid changes after creation
    model_config = ConfigDict(extra="forbid", frozen=True)

    # simulation and traffic
    simulation_time_s: float = Field(gt=0)
    round_duration_s: float = Field(gt=0)
    poisson_rate: float = Field(gt=0)
    road_length_m: float = Field(gt=0)
    min_speed_kph: float = Field(gt=0)
    max_speed_kph: float = Field(gt=0)
    seed: int

    # local data
    min_n_data: int = Field(ge=1)
    max_n_data: int = Field(ge=1)
    n_sub_classes: int = Field(ge=1, le=10)
    data_path: str

    # client hardware and channel
    min_cpu_hertz: float = Field(gt=0)
    max_cpu_hertz: float = Field(gt=0)
    n_cpu_cycles_per_data: float = Field(gt=0)
    effective_capacitance: float = Field(gt=0)
    snr_db_min: float
    snr_db_max: float
    bandwidth_hz: float = Field(gt=0)
    tx_power_w: float = Field(gt=0)
    model_precision: int = Field(gt=0)

    # local training
    batch: int = Field(ge=1)
    n_local_epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    momentum: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)

    @model_validator(mode="after")
    def check_ranges(self):
        """Every min_* value must not exceed its max_* counterpart."""
        pairs = [
            ("min_speed_kph", "max_speed_kph"),
            ("min_n_data", "max_n_data"),
            ("min_cpu_hertz", "max_cpu_hertz"),
            ("snr_db_min", "snr_db_max"),
        ]
        for lo, hi in pairs:
            if getattr(self, lo) > getattr(self, hi):
                raise ValueError(f"{lo} must not exceed {hi}")
        return self


def parse_overrides(pairs):
    """Turn ["seed=7", "batch=8"] into {"seed": 7, "batch": 8}."""
    out = {}
    for pair in pairs:
        key, sep, val = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Override must look like key=value: {pair!r}")
        # Reuse the YAML parser so that "7" -> 7 and "0.5" -> 0.5
        out[key.strip()] = yaml.safe_load(val)
    return out


def load_config(path="cfg.yaml", overrides=None):
    """Read a YAML file, apply overrides, return a validated Config."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    raw.update(overrides or {})
    return Config.model_validate(raw)
