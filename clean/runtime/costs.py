import json
import math
from functools import lru_cache
from pathlib import Path

from .features import PARAMETERS, TRANSFER_FEATURES, TRANSFER_PARAMETERS
from .schedules import cost_class

ROOT = Path(__file__).resolve().parents[1]


def _binding(cost_class, pipe, feature):
    implementation, dtype, layout, path, output_format, policy = cost_class.split("|")
    vector = (
        "batch" if implementation == "b" else
        "s1" if implementation == "s1" else
        "same_ab" if implementation in ("same_ab", "varlen_same_ab") else "flash"
    )
    controller = (
        "batch" if implementation == "b" else
        "manual_l1" if implementation == "varlen_l1" or implementation == "varlen_same_ab" and path == "l1_carry" else
        "direct" if implementation in ("same_ab", "varlen_same_ab") else "kfc"
    )
    if pipe == "cube" or (pipe == "vector" and feature == "cast_k"):
        context = dtype
    elif pipe == "vector" and feature in ("softmax_k", "reduction_k", "online_k", "chunks"):
        context = vector
    elif pipe == "cscalar" and feature in ("workers", "commands"):
        context = controller
    elif pipe == "vscalar" and feature == "workers":
        context = controller
    elif pipe == "vscalar" and feature == "chunks":
        context = vector
    elif pipe == "mte2":
        context = "manual_nd2nz" if feature in TRANSFER_FEATURES else controller + ":" + layout
    elif pipe in ("vmte2", "vmte3"):
        context = vector
    else:
        context = "shared"
    return "/".join((pipe, context, feature))


def _aliases(keys, specification):
    labels = []
    removed = set(specification["removed"])
    for key in keys:
        if key in removed:
            labels.append(None)
            continue
        pipe, context, feature = key.split("/")
        for selected_pipe, selected_feature, axis in specification["merges"]:
            if selected_pipe not in ("*", pipe) or selected_feature not in ("*", feature):
                continue
            if axis == "api" and ":" in context:
                context = "shared:" + context.split(":")[1]
            elif axis == "layout" and ":" in context:
                context = context.split(":")[0] + ":shared"
            elif axis not in ("api", "layout"):
                context = "shared"
        labels.append("/".join((pipe, context, feature)))
    return labels


@lru_cache(maxsize=1)
def load_model():
    model = json.loads((ROOT / "model.json").read_text())
    if model["version"] != "fa_validated_shared_costs_v1":
        raise ValueError("shared cost model and predictor versions differ")
    if any(not math.isfinite(value) or value < 0 for value in model["coefficients"].values()):
        raise ValueError("cost coefficients must be finite and nonnegative")
    return model


def rates(work, graph, model):
    names = TRANSFER_PARAMETERS if work.manual_transfer else PARAMETERS
    key = cost_class(work.p, work.s)
    keys = [_binding(key, *name) for name in names]
    aliases = _aliases(keys, model)
    calibrated = set(model["coefficients"]) | set(model["zero_costs"])
    missing = [key for i, (key, alias) in enumerate(zip(keys, aliases))
               if alias is not None and alias not in calibrated and graph["templates"][:, i].any()]
    if missing:
        raise ValueError("uncalibrated source operations: " + ",".join(missing))
    return [model["coefficients"].get(alias, 0.0) for alias in aliases]
