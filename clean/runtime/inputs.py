from __future__ import annotations
import hashlib
import json
import math

INPUT_FIELDS = (
    "B",
    "N1",
    "N2",
    "S1",
    "S2",
    "D",
    "DV",
    "dtype",
    "layout",
    "has_drop_mask",
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align(a: int, b: int = 16) -> int:
    return ceil_div(a, b) * b


def normalize(inputs: dict) -> dict:
    unexpected = set(inputs) - set(INPUT_FIELDS)
    if unexpected:
        raise ValueError(
            f"prediction accepts operator inputs only: unexpected {sorted(unexpected)}"
        )
    missing = set(INPUT_FIELDS) - set(inputs)
    if missing:
        raise ValueError(f"missing inputs: {sorted(missing)}")
    p = dict(inputs)
    for name in INPUT_FIELDS[:7]:
        number = float(p[name])
        if not math.isfinite(number) or not number.is_integer() or number <= 0:
            raise ValueError(f"invalid {name}")
        p[name] = int(number)
    p["dtype"] = str(p["dtype"]).upper()
    p["layout"] = str(p["layout"]).upper()
    p["has_drop_mask"] = int(p["has_drop_mask"])
    if p["N1"] % p["N2"] or p["has_drop_mask"] not in (0, 1):
        raise ValueError("invalid heads or dropout")
    return p


def sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def api_kernel_inputs(inputs, policy=None):
    p = normalize(inputs)
    if policy is None:
        return p
    if p["layout"] != "TND":
        raise ValueError("the API normalization policy covers uniform TND inputs")
    if p["DV"] > p["D"]:
        raise ValueError("official FlashAttentionScore API requires Dqk >= Dv")
    constants = policy["constants"]
    alignment = (
        constants["PAD_BASIC_BLOCK"]
        if p["D"] < constants["PAD_LOWER_BOUND_196"]
        or p["D"] == constants["PAD_ALIGN_SPL_SHAPE"]
        else constants["PAD_ALIGN_128"]
    )
    qpad, vpad = (-p["D"]) % alignment, (-p["DV"]) % alignment
    padded = bool(qpad or vpad)
    if p["D"] != p["DV"] and p["D"] in tuple(
        constants[name] for name in ("HEAD_DIM_64", "HEAD_DIM_80", "HEAD_DIM_128")
    ):
        padded, vpad = True, qpad + p["D"] - p["DV"]
    if (
        p["D"] < constants["TND_UNPAD_MAX_DDIM"]
        and p["S2"] <= constants["TND_UNPAD_MAX_S2"]
        and p["B"] * p["S1"] < constants["TND_UNPAD_MAX_S1_SUM"]
    ):
        padded, qpad, vpad = False, 0, 0
    return {
        **p,
        "D": p["D"] + qpad if padded else p["D"],
        "DV": p["DV"] + vpad if padded else p["DV"],
    }
