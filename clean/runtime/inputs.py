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
        raise ValueError(f"unexpected inputs: {sorted(unexpected)}")
    p = dict(inputs)
    for name in INPUT_FIELDS[:7]:
        number = float(p[name])
        if not math.isfinite(number) or not number.is_integer() or number <= 0:
            raise ValueError(f"invalid {name}")
        p[name] = int(number)
    p["dtype"] = str(p["dtype"]).upper()
    p["layout"] = str(p["layout"]).upper()
    if p["dtype"] not in ("FP16", "BF16") or p["layout"] not in ("BNSD", "SBH", "BSND", "TND"):
        raise ValueError("unsupported dtype or layout")
    drop_mask = float(p["has_drop_mask"])
    if drop_mask not in (0, 1):
        raise ValueError("has_drop_mask must be 0 or 1")
    p["has_drop_mask"] = int(drop_mask)
    if p["N1"] % p["N2"]:
        raise ValueError("N1 must be a multiple of N2")
    return p


def sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def api_kernel_inputs(inputs, policy=None):
    p = inputs
    if policy is None:
        return p
    if p["DV"] > p["D"]:
        raise ValueError("D must be >= DV")
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
