import torch.nn as nn
from fastapi import FastAPI


def create_app(model: nn.Module, model_uri: str) -> FastAPI:
    """Build the API around an already loaded model."""
    app = FastAPI(title="flroad")
    app.state.model = model

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_uri}

    return app
