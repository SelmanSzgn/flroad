import io
from collections.abc import Callable
from typing import Any

import torch
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from flroad.data import CLASSES, get_transform

RESIZE = T.Resize((32, 32))
TO_TENSOR = get_transform(augment=False)


def preprocess(data: bytes) -> torch.Tensor:
    """Turn raw image bytes into a normalized (1, 3, 32, 32) tensor."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return TO_TENSOR(RESIZE(img)).unsqueeze(0)


def create_app(
    model: Callable[[torch.Tensor], torch.Tensor], model_uri: str
) -> FastAPI:
    """Build the API around an already loaded model."""
    app = FastAPI(title="flroad")
    app.state.model = model

    @app.get("/health")
    def health() -> dict[str, str]:
        """Tell that the service is alive and which model it serves."""
        return {"status": "ok", "model": model_uri}

    @app.post("/predict")
    def predict(file: UploadFile) -> dict[str, Any]:
        """Classify an uploaded image into a CIFAR-10 class."""
        try:
            x = preprocess(file.file.read())
        except OSError:
            raise HTTPException(400, "The file is not a readable image")
        with torch.no_grad():
            probs = model(x).softmax(dim=1)[0]
        best = int(probs.argmax())
        return {
            "class": CLASSES[best],
            "index": best,
            "probabilities": {
                c: round(float(p), 4) for c, p in zip(CLASSES, probs)
            },
        }

    return app
