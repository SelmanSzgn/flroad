import pytest
from pydantic import ValidationError

from flroad.campaign import Campaign, expand_grid, load_campaign


def test_expand_grid_is_a_cartesian_product():
    out = expand_grid({"a": [1, 2], "b": [10, 20, 30]}, [0, 1])
    assert len(out) == 12
    assert {"a": 2, "b": 20, "seed": 1} in out


def test_expand_grid_without_grid_gives_one_run_per_seed():
    assert expand_grid({}, [5, 6]) == [{"seed": 5}, {"seed": 6}]


def test_campaign_file_is_loaded_and_expanded(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(
        "experiment: demo\n"
        "seeds: [1, 2]\n"
        "grid:\n  batch: [8, 16]\n"
        "fixed:\n  simulation_time_s: 120\n"
    )
    camp = load_campaign(f)
    assert camp.base == "cfg.yaml"
    runs = camp.runs()
    assert len(runs) == 4
    assert {"simulation_time_s": 120, "batch": 8, "seed": 1} in runs


def test_repository_campaign_is_valid():
    assert len(load_campaign("campaigns/epochs.yaml").runs()) == 18


@pytest.mark.parametrize(
    "bad",
    [
        {"experiment": "x", "seeds": []},
        {"experiment": "x", "seeds": [1], "grid": {"batch": []}},
        {"experiment": "x", "seeds": [1], "grid": {"seed": [1, 2]}},
        {"experiment": "x", "seeds": [1], "fixed": {"seed": 3}},
        {
            "experiment": "x",
            "seeds": [1],
            "grid": {"batch": [8]},
            "fixed": {"batch": 16},
        },
        {"experiment": "x", "seeds": [1], "seed": 3},
    ],
)
def test_invalid_campaigns_are_rejected(bad):
    with pytest.raises(ValidationError):
        Campaign.model_validate(bad)


def test_run_name_lists_grid_keys_and_seed():
    camp = Campaign(
        experiment="x",
        seeds=[1],
        grid={"batch": [8], "n_local_epochs": [1]},
        fixed={"simulation_time_s": 60},
    )
    name = camp.run_name(camp.runs()[0])
    assert name == "batch=8,n_local_epochs=1,seed=1"
