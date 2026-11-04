import json

from .fa_branch_latency import MODEL, predict as _predict


def predict(inputs, model=None):
    if model is None:
        model = json.loads(MODEL.read_text())
    return _predict(inputs, model)

__all__ = ["MODEL", "predict"]
