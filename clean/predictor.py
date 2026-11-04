import json
from pathlib import Path

if __package__:
    from .runtime import fa_branch_latency as engine
else:
    from runtime import fa_branch_latency as engine


ROOT = Path(__file__).resolve().parent


def _load_entry(entry):
    entry["rates"] = [float(entry["pipes"][pipe]["coefficients"].get(name, 0))
                      for pipe, name in engine.parameter_names()]
    refinement = entry.get("source_refinement")
    if refinement:
        for child in refinement["children"].values():
            _load_entry(child)


def _load_model():
    model = json.loads((ROOT / "model.json").read_text())
    for name in ("baseline", "refined"):
        bank = model[name]
        bank["hardware"] = model["hardware"]
        for entry in bank["classes"].values():
            _load_entry(entry)
    return model


def predict(inputs):
    return engine.predict(inputs, _load_model())
