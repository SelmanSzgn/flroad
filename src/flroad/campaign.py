import itertools
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def expand_grid(
    grid: dict[str, list[Any]], seeds: list[int]
) -> list[dict[str, Any]]:
    """Return one override dict per (grid combination, seed) pair."""
    keys = list(grid)
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        base = dict(zip(keys, combo))
        for seed in seeds:
            out.append({**base, "seed": seed})
    return out


class Campaign(BaseModel):
    """A set of runs: a grid of parameters repeated over several seeds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment: str
    base: str = "cfg.yaml"
    seeds: list[int] = Field(min_length=1)
    grid: dict[str, list[Any]] = {}
    fixed: dict[str, Any] = {}

    @model_validator(mode="after")
    def check_keys(self) -> Self:
        """Seeds have their own field, and nothing is set twice."""
        for name, vals in self.grid.items():
            if not vals:
                raise ValueError(f"Grid entry {name!r} has no value")
        overlap = set(self.grid) & set(self.fixed)
        if overlap or "seed" in self.grid or "seed" in self.fixed:
            raise ValueError(
                "Use 'seeds' for seeds, and a key only once "
                f"(in grid or fixed): {sorted(overlap)}"
            )
        return self

    def runs(self) -> list[dict[str, Any]]:
        """Return the overrides of every run of the campaign."""
        return [{**self.fixed, **o} for o in expand_grid(self.grid, self.seeds)]


def load_campaign(path: str | Path) -> Campaign:
    """Read a YAML campaign file and return a validated Campaign."""
    with open(path, "r") as f:
        return Campaign.model_validate(yaml.safe_load(f))
