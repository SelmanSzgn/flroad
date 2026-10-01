import torch.nn as nn
from fastapi.testclient import TestClient

from flroad.api import create_app


def test_health_reports_status_and_model():
    client = TestClient(create_app(nn.Linear(2, 1), "models:/flroad/1"))
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model": "models:/flroad/1"}
