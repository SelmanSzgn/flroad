import io

import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from flroad.api import create_app, preprocess
from flroad.data import CLASSES
from flroad.model import Model


def png_bytes(color, size=(100, 50)):
    """Build an in-memory PNG file of one color."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_health_reports_status_and_model():
    client = TestClient(create_app(nn.Linear(2, 1), "models:/flroad/1"))
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model": "models:/flroad/1"}


def test_preprocess_resizes_and_normalizes():
    x = preprocess(png_bytes((255, 255, 255)))
    assert x.shape == (1, 3, 32, 32)
    # white pixel: (255 / 255 - 0.5) / 0.5 = 1.0, like during training
    assert torch.allclose(x, torch.ones_like(x))


def test_predict_returns_a_class_and_probabilities():
    client = TestClient(create_app(Model(), "test"))
    files = {"file": ("x.png", png_bytes((10, 200, 30)), "image/png")}
    res = client.post("/predict", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["class"] == CLASSES[body["index"]]
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-2


def test_predict_rejects_a_file_that_is_not_an_image():
    client = TestClient(create_app(Model(), "test"))
    files = {"file": ("x.txt", b"hello", "text/plain")}
    assert client.post("/predict", files=files).status_code == 400


def test_predict_works_with_a_model_that_has_no_eval():
    net = Model()

    class Frozen:
        """Callable without .eval(), like an exported graph."""

        def __call__(self, x):
            return net(x)

    client = TestClient(create_app(Frozen(), "test"))
    files = {"file": ("x.png", png_bytes((10, 200, 30)), "image/png")}
    assert client.post("/predict", files=files).status_code == 200
