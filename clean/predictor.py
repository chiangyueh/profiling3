import json
from pathlib import Path

if __package__:
    from .runtime import fa_branch_latency as engine
else:
    from runtime import fa_branch_latency as engine


ROOT = Path(__file__).resolve().parent


def _load_entry(entry, diagnostics):
    allowed = {"local_identifiability", "micro_context_ranges", "source_condition_support", "children"}
    if set(diagnostics) - allowed:
        raise ValueError("unknown diagnostic fields")
    for name, value in diagnostics.items():
        if name == "children":
            continue
        if name in entry:
            raise ValueError("diagnostics cannot replace model parameters: " + name)
        entry[name] = value
    entry["rates"] = [float(entry["pipes"][pipe]["coefficients"].get(name, 0))
                      for pipe, name in engine.parameter_names()]
    refinement = entry.get("source_refinement")
    if refinement:
        for name, child in refinement["children"].items():
            _load_entry(child, diagnostics.get("children", {}).get(name, {}))


def _load_model():
    model = json.loads((ROOT / "model.json").read_text())
    diagnostics = json.loads((ROOT / "diagnostics.json").read_text())
    choices = {}
    for name, approved in model["selection"].items():
        evidence = diagnostics["selection"][name]
        if "approved" in evidence:
            raise ValueError("diagnostics cannot replace model selection")
        choices[name] = {"approved": approved, **evidence}
    model["selection"] = choices
    for name in ("baseline", "refined"):
        bank, metadata = model[name], diagnostics[name]
        bank["hardware"] = model["hardware"]
        bank["sdk_api_sha256"] = metadata["sdk_api_sha256"]
        if "refinement_candidates" in metadata:
            bank["refinement_candidates"] = metadata["refinement_candidates"]
        for key, entry in bank["classes"].items():
            _load_entry(entry, metadata["classes"].get(key, {}))
    return model


def predict(inputs):
    return engine.predict(inputs, _load_model())
