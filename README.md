# flroad

Federated learning simulator for vehicular networks (mobility, computation
and communication costs), with MLflow tracking and a small serving API.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

Requires Python 3.12+. PyTorch is installed in its CPU version.

## Run a simulation

```bash
flroad run                                    # uses cfg.yaml
flroad run --set seed=7 --set n_local_epochs=2 -e my-experiment
```

Each run writes `runs/<id>/` (`config.yaml`, `metrics.csv`, `model.pt`)
and is tracked in MLflow (params, per-round metrics, artifacts, model).

## Run a campaign

A campaign is a YAML file: a grid of parameters repeated over several seeds.

```bash
flroad sweep campaigns/epochs.yaml
```

All configurations are validated before the first run starts.

## Compare runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000, select runs, click **Compare**.

## Serve a model

```bash
flroad serve -m models:/flroad/1              # a registry version
flroad serve -m runs:/<run_id>/model          # a specific run
```

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@image.png" http://127.0.0.1:8000/predict
```

Interactive docs: http://127.0.0.1:8000/docs

## Docker

```bash
make up                                       # MLflow on :5000
make run ARGS="-e demo --set seed=7"          # one run
make sweep CAMPAIGN=campaigns/epochs.yaml     # a campaign
make serve MODEL=models:/flroad/1             # API on :8000
make down                                     # stop (data is kept)
```

After editing files in `campaigns/` or the code, run `make up` again to
rebuild the image.

## Development

```bash
make check    # ruff format, ruff lint, mypy, pytest
```

CI runs the same checks on every push.

## Configuration

All parameters are in `cfg.yaml` and validated at load time
(`src/flroad/config.py`). Any value can be overridden with `--set key=value`.
