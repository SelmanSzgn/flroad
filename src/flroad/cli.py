from pathlib import Path
from typing import Annotated, Optional

import typer

from flroad.campaign import load_campaign
from flroad.config import load_config, parse_overrides

app = typer.Typer()


@app.callback()
def main() -> None:
    """Federated learning simulator for vehicular networks."""


@app.command("run")
def run_cmd(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML config file.")
    ] = Path("cfg.yaml"),
    set_: Annotated[
        Optional[list[str]],
        typer.Option("--set", "-s", help="Override a value: key=value."),
    ] = None,
    experiment: Annotated[
        str, typer.Option("--experiment", "-e", help="MLflow experiment.")
    ] = "flroad",
) -> None:
    """Run one simulation."""
    try:
        cfg = load_config(config, parse_overrides(set_ or []))
    except FileNotFoundError:
        typer.echo(f"Config file not found: {config}", err=True)
        raise typer.Exit(code=1)
    except ValueError as err:
        typer.echo(f"Invalid configuration:\n{err}", err=True)
        raise typer.Exit(code=1)

    # Imported here so that --help and config errors stay instant
    from flroad.main import run

    run(cfg, experiment)


@app.command("sweep")
def sweep_cmd(
    campaign: Annotated[Path, typer.Argument(help="YAML campaign file.")],
) -> None:
    """Run every configuration of a campaign, one after the other."""
    try:
        camp = load_campaign(campaign)
        runs = camp.runs()
        # Build and validate ALL configs before running anything
        cfgs = [load_config(camp.base, o) for o in runs]
    except FileNotFoundError as err:
        typer.echo(f"File not found: {err.filename}", err=True)
        raise typer.Exit(code=1)
    except ValueError as err:
        typer.echo(f"Invalid campaign or configuration:\n{err}", err=True)
        raise typer.Exit(code=1)

    from flroad.main import run

    for i, (over, cfg) in enumerate(zip(runs, cfgs), start=1):
        name = camp.run_name(over)
        print(f"=== run {i}/{len(cfgs)}: {name} ===")
        run(cfg, camp.experiment, name)


@app.command("serve")
def serve_cmd(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="MLflow model URI, e.g. models:/flroad/1 or runs:/<id>/model",
        ),
    ],
    host: Annotated[str, typer.Option(help="Address to listen on.")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Serve a tracked model through an HTTP API."""
    # Imported here so that --help stays instant
    import uvicorn

    from flroad.api import create_app
    from flroad.tracking import load_tracked_model

    try:
        net = load_tracked_model(model)
    except Exception as err:  # MLflow raises several kinds of errors
        typer.echo(f"Cannot load model {model!r}:\n{err}", err=True)
        raise typer.Exit(code=1)

    uvicorn.run(create_app(net, model), host=host, port=port)


if __name__ == "__main__":
    app()
