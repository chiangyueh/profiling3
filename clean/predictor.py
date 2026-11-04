import json
from functools import lru_cache
from pathlib import Path

if __package__:
    from .runtime import fa_branch_latency as engine
else:
    from runtime import fa_branch_latency as engine


ROOT = Path(__file__).resolve().parent


def _load_pipe(pipe, record, parameters):
    names = parameters[pipe]
    values = record["coefficients"]
    unsupported = set(record.get("unsupported", []))
    if set(values) - set(names) or unsupported - set(names):
        raise ValueError("invalid cost parameters: " + pipe)
    return {
        "coefficients": {name: values.get(name, 0.0) for name in names},
        "observed_features": [name for name in names if name not in unsupported],
    }


def _load_entry(entry, parameters):
    entry["pipes"] = {
        pipe: _load_pipe(pipe, record, parameters)
        for pipe, record in entry.get("pipes", {}).items()
    }
    entry["rates"] = [
        float(entry["pipes"].get(pipe, {}).get("coefficients", {}).get(name, 0))
        for pipe, name in engine.parameter_names()
    ]
    refinement = entry.get("source_refinement")
    if refinement:
        for child in refinement["children"].values():
            _load_entry(child, parameters)


@lru_cache(maxsize=1)
def _load_model():
    model = json.loads((ROOT / "model.json").read_text())
    parameters = model.pop("parameters")
    if parameters != {pipe: list(names) for pipe, (_, names) in engine.PIPES.items()}:
        raise ValueError("cost parameter names differ from execution model")
    for name in ("baseline", "refined"):
        bank = model[name]
        bank["hardware"] = model["hardware"]
        for entry in bank["classes"].values():
            _load_entry(entry, parameters)
        priors = {}
        for key, record in bank["primitive_priors"].items():
            pipe = key.split("|")[0]
            profile = (
                bank["classes"][record["use_class"]]["pipes"][pipe]
                if "use_class" in record
                else _load_pipe(pipe, record, parameters)
            )
            priors[json.dumps(key.split("|"))] = {
                "names": parameters[pipe],
                "rates": [profile["coefficients"][name] for name in parameters[pipe]],
                "observed_features": profile["observed_features"],
            }
        bank["primitive_priors"] = priors
    return model


def predict(inputs):
    return engine.predict(inputs, _load_model())
