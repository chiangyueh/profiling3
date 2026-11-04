from __future__ import annotations
import hashlib
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
VERSION = "fa_input_latency_v1"
INPUT_FIELDS = ("B", "N1", "N2", "S1", "S2", "D", "DV", "dtype", "layout", "has_drop_mask")
FEATURE_NAMES = ("fixed", "critical_work_m", "kv_read_mb", "global_io_mb", "serial_steps", "adapter_melements", "cache_reload_mb")
ACTIVE_FEATURES = {
    "same_ab": ("fixed", "critical_work_m", "global_io_mb"),
    "same_ab/SBH": ("fixed", "critical_work_m", "cache_reload_mb"),
    "s1": FEATURE_NAMES[:5],
    "b": FEATURE_NAMES[:5],
    "b_drop": FEATURE_NAMES[:6],
}
RELATIVE_ERROR_LIMIT = 0.10
FIT_METHOD = "nonnegative_address_minimax_v4"
ADDRESS_MODEL = {
    "line_bytes": 512, "groups": 24,
    "source": "recovered Toolkit behavioral simulator address mapping",
    "qualification": "effective address-cost feature; not verified physical silicon cache geometry",
}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align(a: int, b: int = 16) -> int:
    return ceil_div(a, b) * b


@lru_cache(maxsize=256)
def sbh_address_cost(b: int, n: int, nk: int, s1: int, s2: int, d: int, dv: int,
                     base: int, s2_base: int, ratio: int, aic: int, l2: int) -> dict:
    line, groups = ADDRESS_MODEL["line_bytes"], ADDRESS_MODEL["groups"]
    if b * n * d * 2 % line or b * nk * d * 2 % line or b * nk * dv * 2 % line:
        raise ValueError("SBH address model requires cache-line-aligned sequence strides")
    outer = ceil_div(s1, base)
    total = b * n * outer
    split = ceil_div(total, min(total, aic))
    mm1 = align(base * s2_base * ratio, 128) * 4
    mm2 = align(base * align(dv), 128) * 4
    workspace_per_core = 3 * mm1 + 4 * mm2

    @lru_cache(maxsize=256)
    def row_lines(heads: tuple, width: int, stride: int, rows: int, first_row: int = 0) -> np.ndarray:
        intervals = []
        for lo, hi in sorted((head * width // line, (head * width + width - 1) // line) for head in heads):
            if intervals and lo <= intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], max(hi, intervals[-1][1]))
            else:
                intervals.append((lo, hi))
        counts = np.zeros(groups)
        stride_lines = stride // line
        period = groups // math.gcd(stride_lines, groups)
        for step in range(min(rows, period)):
            repeats = ceil_div(rows - step, period)
            shift = (first_row + step) * stride_lines
            for lo, hi in intervals:
                for position in range(lo, hi + 1):
                    counts[(position + shift) % groups] += repeats
        return counts

    pressures = []
    for step in range(split):
        tasks = [core * split + step for core in range(aic) if core * split + step < total]
        heads = {task // outer for task in tasks}
        kv_heads = tuple(sorted({head // n * nk + head % n // (n // nk) for head in heads}))
        kv = row_lines(kv_heads, d * 2, b * nk * d * 2, s2) + row_lines(kv_heads, dv * 2, b * nk * dv * 2, s2)
        queries = {}
        for task in tasks:
            queries.setdefault(task % outer * base, []).append(task // outer)
        query_lines = np.zeros(groups)
        for first, query_heads in queries.items():
            query_lines += row_lines(tuple(sorted(query_heads)), d * 2, b * n * d * 2, min(base, s1 - first), first)
        workspace_lines = ceil_div(len(tasks) * workspace_per_core, line)
        workspace = np.full(groups, workspace_lines // groups, dtype=float)
        workspace[:workspace_lines % groups] += 1
        excess = np.maximum(0.0, kv + query_lines + workspace - l2 / line / groups)
        pressures.append(float(excess.sum() / kv.sum()))
    pressure = sum(pressures) / len(pressures)
    return {
        "reload_fraction": pressure,
        "critical_reload_mb": split * s2 * (d + dv) * 2 / 1e6 * pressure,
        "active_workspace_bytes": min(aic, ceil_div(total, split)) * workspace_per_core,
    }


def normalize(inputs: dict) -> dict:
    unexpected = set(inputs) - set(INPUT_FIELDS)
    if unexpected:
        raise ValueError(f"prediction accepts operator inputs only: unexpected {sorted(unexpected)}")
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


def derive(inputs: dict, hardware: dict) -> dict:
    p = normalize(inputs)
    b, n, nk, s1, s2, d, dv = (p[k] for k in INPUT_FIELDS[:7])
    aic = int(hardware["aic_cores"])
    aiv = int(hardware["aiv_cores"])
    l1 = int(hardware["l1_bytes"])
    if min(aic, aiv, l1) <= 0:
        raise ValueError("hardware profile is incomplete")
    as1, as2 = align(s1), align(s2)
    if p["dtype"] == "BF16" and d == 192 and dv == 128 and not p["has_drop_mask"] and s2 > 1024 and p["layout"] in ("BNSD", "SBH"):
        family, priority = "same_ab", 95
        base = min(256, as1)
        s2_base = min(128, as2)
        ratio = min(8, ceil_div(s2, s2_base))
        outer = ceil_div(s1, base)
        total = b * n * outer
        split = ceil_div(total, min(total, aic))
        windows = ceil_div(s2, s2_base * ratio)
    elif p["dtype"] == "FP16" and d == dv == 128 and p["layout"] == "BNSD" and not p["has_drop_mask"] and 128 < s2 <= 1024 and b * n <= 64:
        family, priority = "s1", 97
        initial_base = min(128, as1)
        ratio = 1
        total = b * n * ceil_div(s1, initial_base)
        while ratio < 4 and total * ratio > (ratio - 1) * aiv and initial_base * ratio < as1 and ratio * initial_base * as2 <= 131072:
            ratio += 1
            total = b * n * ceil_div(ceil_div(s1, initial_base), ratio)
        while total * ratio <= (ratio - 1) * aiv or initial_base * ratio > as1 or ratio * initial_base * as2 > 131072:
            ratio -= 1
            if ratio <= 0:
                raise ValueError("unsupported S1 scheduling")
        base = initial_base * ratio
        outer = ceil_div(s1, base)
        total = b * n * outer
        split = ceil_div(total, min(total, aiv))
        s2_base, windows = min(128, as2), 1
    elif p["dtype"] == "FP16" and d == dv == 64 and p["layout"] == "BNSD" and s2 <= 128:
        if n * as1 * as2 > 65536 or n * ((as1 + as2) * d + as2) * 2 >= l1 or n * (as1 + d) * as2 * 2 >= l1:
            raise ValueError("this input leaves the supported B route")
        family, priority = ("b_drop" if p["has_drop_mask"] else "b"), 98
        base = min(256, as1, (8192 // as2 // 16) * 16)
        outer = ceil_div(s1, base)
        s2_base, ratio, total, windows = as2, 0, b, 1
        split = ceil_div(total, min(total, aiv))
    else:
        raise ValueError("outside the implemented dense SameAB/S1/B/DropMask input domain")
    workers = ceil_div(total, split)
    if family in ("b", "b_drop"):
        critical_rows = min(split, total) * n * as1
        reads = min(split, total) * n
        steps = min(split, total) * n * outer
    else:
        def rows_before(count: int) -> int:
            groups, remainder = divmod(count, outer)
            return groups * align(s1) + min(remainder * base, align(s1))

        critical_rows = max(rows_before(min(start + split, total)) - rows_before(start) for start in range(0, total, split))
        reads = min(split, total)
        steps = min(split, total) * windows
    adapter = bool(p["has_drop_mask"] and s2 % 8)
    io = 2 * (b * n * s1 * (d + dv) + b * nk * s2 * (d + dv))
    features = {
        "fixed": 1.0,
        "critical_work_m": 2 * critical_rows * as2 * (align(d) + align(dv)) / 1e6,
        "kv_read_mb": reads * as2 * (d + dv) * 2 / 1e6,
        "global_io_mb": io / 1e6,
        "serial_steps": float(steps),
        "adapter_melements": b * n * s1 * s2 / aiv / 1e6 if adapter else 0.0,
        "cache_reload_mb": 0.0,
    }
    address = None
    if family == "same_ab" and p["layout"] == "SBH":
        l2 = int(hardware.get("l2_bytes", 0))
        if l2 <= 0:
            raise ValueError("SBH prediction requires the saved hardware L2 capacity")
        address = dict(sbh_address_cost(b, n, nk, s1, s2, d, dv, base, s2_base, ratio, aic, l2))
        features["cache_reload_mb"] = address["critical_reload_mb"]
    return {
        "family": family, "priority": priority, "s1_base": base, "s2_base": s2_base,
        "n_ratio": ratio, "s1_outer": outer, "total_tasks": total, "split_factor": split,
        "workers": workers, "s2_windows": windows, "features": features,
        "address_cost": address,
    }


def verify_schedule(inputs: dict, hardware: dict, priority: int, effective: dict) -> dict:
    derived = derive(inputs, hardware)
    if priority != derived["priority"]:
        raise ValueError(f"input-only route {derived['priority']} differs from official route {priority}")
    pairs = {
        "s1_base": "FA_S1_BASE", "s2_base": "FA_S2_BASE", "n_ratio": "FA_N_RATIO",
        "s1_outer": "FA_S1_OUTER", "total_tasks": "FA_TOTAL_TASKS", "split_factor": "FA_SPLIT_FACTOR",
    }
    for predicted, actual in pairs.items():
        if derived[predicted] != effective[actual]:
            raise ValueError(f"input-only {predicted}={derived[predicted]} differs from official {actual}={effective[actual]}")
    return derived


def group_key(inputs: dict, hardware: dict) -> str:
    p = normalize(inputs)
    return f"{derive(p, hardware)['family']}/{p['dtype']}/{p['layout']}"


def active_features(inputs: dict, hardware: dict) -> tuple:
    p = normalize(inputs)
    family = derive(p, hardware)["family"]
    return ACTIVE_FEATURES.get(f"{family}/{p['layout']}", ACTIVE_FEATURES[family])


def validate_design(plan: list[dict], hardware: dict) -> dict:
    groups = {}
    for case in plan:
        key = group_key(case["inputs"], hardware)
        groups.setdefault(key, []).append(case)
    report = {}
    for key, cases in groups.items():
        calibration = [case for case in cases if case["split"] == "calibration"]
        holdout = [case for case in cases if case["split"] == "holdout"]
        pairs = lambda items: {(case["inputs"]["S1"], case["inputs"]["S2"]) for case in items}
        if len(calibration) < 8 or not holdout or pairs(calibration) & pairs(holdout):
            raise ValueError(f"invalid independent shape split for {key}")
        active = active_features(calibration[0]["inputs"], hardware)
        matrix = np.asarray([[derive(case["inputs"], hardware)["features"][f] for f in active] for case in calibration])
        rank = int(np.linalg.matrix_rank(matrix / np.maximum(matrix.max(axis=0), 1e-12)))
        if rank != len(active):
            raise ValueError(f"calibration inputs cannot identify the cost terms for {key}")
        report[key] = {"calibration": len(calibration), "holdout": len(holdout), "rank": rank, "cost_terms": len(active)}
    return report


def predict(inputs: dict, model: dict) -> dict:
    if model.get("version") != VERSION or model.get("fit_method") != FIT_METHOD:
        raise ValueError("a frozen input-only FA latency model is required")
    if model.get("address_model") != ADDRESS_MODEL:
        raise ValueError("the frozen model uses a different address-cost mapping")
    p = normalize(inputs)
    hardware = model["hardware"]
    derived = derive(p, hardware)
    key = group_key(p, hardware)
    if key not in model["groups"]:
        raise ValueError(f"no frozen calibration for {key}")
    coefficients = model["groups"][key]["coefficients"]
    components = {f: coefficients[f] * derived["features"][f] for f in FEATURE_NAMES}
    duration = sum(components.values())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid prediction")
    ranges = model["groups"][key]["feature_ranges"]
    inputs_ranges = model["groups"][key]["input_ranges"]
    return {
        "predicted_us": duration, "group": key, "derived_schedule": derived,
        "cost_terms_us": components,
        "extrapolated_features": [f for f in ranges if not ranges[f][0] <= derived["features"][f] <= ranges[f][1]],
        "extrapolated_inputs": [f for f in inputs_ranges if not inputs_ranges[f][0] <= p[f] <= inputs_ranges[f][1]],
    }


def design(hardware: dict) -> list[dict]:
    cases = []
    seen = set()

    def add(split, b, n, nk, s1, s2, d, dv, dtype, layout, drop=0):
        shape = (b, n, nk, s1, s2, d, dv, dtype, layout, drop)
        if shape in seen:
            return
        inputs = dict(zip(INPUT_FIELDS, shape))
        try:
            derived = derive(inputs, hardware)
        except ValueError:
            return
        name = f"{derived['family']}_{layout.lower()}_b{b}_n{n}_nk{nk}_s{s1}_s{s2}_d{d}_v{dv}_drop{drop}"
        cases.append({"name": name, "split": split, "family": derived["family"], "expected": derived["priority"], "inputs": inputs, "shape": shape})
        seen.add(shape)

    for layout, s, n in itertools.product(("BNSD", "SBH"), (2048, 4096, 8192), (8, 32, 96)):
        add("calibration", 1, n, n, s, s, 192, 128, "BF16", layout)
    for b, n, nk in ((1, 4, 1), (1, 16, 2), (2, 16, 4)):
        for s1, s2 in ((64, 256), (128, 256), (256, 256), (512, 256), (64, 512), (256, 512), (128, 1024), (512, 1024)):
            add("calibration", b, n, nk, s1, s2, 128, 128, "FP16", "BNSD")
    small_shapes = ((16, 32, 1), (32, 64, 2), (64, 64, 4), (128, 32, 4), (64, 128, 4), (128, 64, 2))
    for drop, b, (s1, s2, n) in itertools.product((0, 1), (1, 4, 16), small_shapes):
        add("calibration", b, n, 1, s1, s2, 64, 64, "FP16", "BNSD", drop)
    for b, s1, s2 in itertools.product((1, 4, 16), (31, 63), (33, 65)):
        add("calibration", b, 2, 1, s1, s2, 64, 64, "FP16", "BNSD", 1)
    for layout, s, n in itertools.product(("BNSD", "SBH"), (2816, 5888), (12, 48)):
        add("holdout", 1, n, n, s, s, 192, 128, "BF16", layout)
    for b, n, nk in ((1, 12, 3), (3, 8, 2)):
        for s1, s2 in ((160, 384), (320, 384), (160, 896), (640, 896)):
            add("holdout", b, n, nk, s1, s2, 128, 128, "FP16", "BNSD")
    for drop, b, (s1, s2, n) in itertools.product((0, 1), (2, 8), ((48, 96, 3), (96, 48, 3), (80, 64, 5))):
        add("holdout", b, n, 1, s1, s2, 64, 64, "FP16", "BNSD", drop)
    for b in (2, 8):
        add("holdout", b, 3, 1, 47, 49, 64, 64, "FP16", "BNSD", 1)
    return cases


def sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
