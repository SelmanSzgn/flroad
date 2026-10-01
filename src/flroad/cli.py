from pathlib import Path
from typing import Annotated, Optional

import typer

from flroad.config import load_config, parse_overrides

app = typer.Typer()


@app.callback()
def main():
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
):
    """Load the configuration and print it."""
    cfg = load_config(config, parse_overrides(set_ or []))
    print(cfg)


if __name__ == "__main__":
    app()
