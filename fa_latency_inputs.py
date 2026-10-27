from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


VERSION = "fa_input_latency_v1"
INPUT_FIELDS = ("B", "N1", "N2", "S1", "S2", "D", "DV", "dtype", "layout", "has_drop_mask")
FEATURE_NAMES = ("fixed", "critical_work_m", "kv_read_mb", "global_io_mb", "serial_steps", "adapter_melements", "cache_pressure_work_m")
ACTIVE_FEATURES = {
    "same_ab": ("fixed", "critical_work_m", "global_io_mb"),
    "same_ab/SBH": ("fixed", "critical_work_m", "cache_pressure_work_m"),
    "s1": FEATURE_NAMES[:5],
    "b": FEATURE_NAMES[:5],
    "b_drop": FEATURE_NAMES[:6],
}
RELATIVE_ERROR_LIMIT = 0.10
FIT_METHOD = "nonnegative_capacity_model_v3"


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align(a: int, b: int = 16) -> int:
    return ceil_div(a, b) * b


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
        "cache_pressure_work_m": 0.0,
    }
    workspace, footprint, pressure = 0, 0, 0.0
    if family == "same_ab" and p["layout"] == "SBH":
        l2 = int(hardware.get("l2_bytes", 0))
        if l2 <= 0:
            raise ValueError("SBH prediction requires the saved hardware L2 capacity")
        mm1_bytes = align(base * s2_base * ratio, 128) * 4
        mm2_bytes = align(base * align(dv), 128) * 4
        workspace = workers * (3 * mm1_bytes + 4 * mm2_bytes)
        footprint = io + 2 * b * n * s1 * 8 * 4 + workspace
        pressure = max(0.0, 1.0 - l2 / footprint)
        features["cache_pressure_work_m"] = features["critical_work_m"] * pressure
    return {
        "family": family, "priority": priority, "s1_base": base, "s2_base": s2_base,
        "n_ratio": ratio, "s1_outer": outer, "total_tasks": total, "split_factor": split,
        "workers": workers, "s2_windows": windows, "features": features,
        "active_workspace_bytes": workspace, "cache_footprint_bytes": footprint, "l2_capacity_pressure": pressure,
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


def fit_group(rows: list[dict], hardware: dict) -> dict:
    active = active_features(rows[0]["inputs"], hardware)
    x = np.asarray([[derive(row["inputs"], hardware)["features"][f] for f in active] for row in rows])
    y = np.asarray([float(row["duration_us"]) for row in rows])
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("calibration durations must be finite and positive")
    relative = x / y[:, None]
    scales = np.maximum(np.linalg.norm(relative, axis=0), np.finfo(float).tiny)
    matrix = relative / scales
    best = None
    for flags in itertools.product((False, True), repeat=len(active)):
        selected = np.flatnonzero(flags)
        weights = np.zeros(len(active))
        if len(selected):
            weights[selected] = np.linalg.lstsq(matrix[:, selected], np.ones(len(rows)), rcond=None)[0]
        if np.any(weights < -1e-10):
            continue
        weights = np.maximum(weights, 0)
        loss = float(np.sum((matrix @ weights - 1) ** 2))
        if best is None or loss < best[0]:
            best = (loss, weights / scales)
    return {
        "coefficients": {**dict.fromkeys(FEATURE_NAMES, 0.0), **dict(zip(active, best[1].tolist()))},
        "calibration_cases": [row["case"] for row in rows],
        "feature_rank": int(np.linalg.matrix_rank(x / np.maximum(np.max(x, axis=0), 1e-12))),
        "feature_count": len(active),
        "active_features": list(active),
        "feature_ranges": {name: [float(min(x[:, i])), float(max(x[:, i]))] for i, name in enumerate(active)},
        "input_ranges": {name: [min(row["inputs"][name] for row in rows), max(row["inputs"][name] for row in rows)] for name in INPUT_FIELDS[:7]},
    }


def fit(rows: list[dict], hardware: dict, plan_sha256: str) -> dict:
    calibration = [
        row for row in rows
        if row["split"] == "calibration" and row["status"] == "ok" and str(row.get("stable")).lower() == "true"
    ]
    groups = sorted({group_key(row["inputs"], hardware) for row in calibration})
    models = {}
    for key in groups:
        items = [row for row in calibration if group_key(row["inputs"], hardware) == key]
        if len(items) < 8:
            continue
        models[key] = fit_group(items, hardware)
    artifact = {
        "version": VERSION, "plan_sha256": plan_sha256, "hardware": hardware, "groups": models,
        "input_fields": list(INPUT_FIELDS), "features": list(FEATURE_NAMES), "fit_method": FIT_METHOD,
        "target": "median msprof FlashAttentionScore Task Duration(us), warm device, fixed compiled operator",
        "scope": "dense official FA; no PSE/attention mask/sparse mode; scale=1/sqrt(D); dropout keep_prob=0.9 or 1",
        "method": "nonnegative relative-error least squares on algorithm-derived costs; no coefficient penalty or per-shape corrections",
        "cache_model": "SBH capacity-pressure proxy from Q/K/V/output bytes, two FP32 softmax statistics and the active three-stage workspace; it is not an exact cache-hit simulator",
        "identifiability": "fixed D/DV makes Cube and Softmax work correlated; coefficients represent combined costs, not independently measured primitive latency",
        "acceptance": "every blind holdout must execute correctly, match the derived schedule, have stable measurements, and have absolute relative error <= 0.10",
    }
    cv = {}
    for key in groups:
        items = [row for row in calibration if group_key(row["inputs"], hardware) == key]
        axes = {}
        for axis, fields in (("sequence_pair", ("S1", "S2")), ("heads", ("N1",)), ("batch", ("B",))):
            identity = lambda row: tuple(row["inputs"][field] for field in fields)
            folds = sorted({identity(row) for row in items})
            if len(folds) < 2:
                axes[axis] = {"tested": False, "reason": "calibration contains only one value", "count": 0}
                continue
            results = []
            for fold in folds:
                train = [row for row in items if identity(row) != fold]
                if len(train) < 3:
                    continue
                held_group = fit_group(train, hardware)
                held_model = {**artifact, "groups": {key: held_group}}
                for row in items:
                    if identity(row) != fold:
                        continue
                    if held_group["feature_rank"] != held_group["feature_count"]:
                        results.append({"case": row["case"], "status": "unidentifiable_cost_terms", "feature_rank": held_group["feature_rank"], "feature_count": held_group["feature_count"]})
                        continue
                    predicted = predict(row["inputs"], held_model)["predicted_us"]
                    error = abs(predicted / float(row["duration_us"]) - 1)
                    results.append({"case": row["case"], "status": "predicted", "predicted_us": predicted, "actual_us": float(row["duration_us"]), "ape_percent": error * 100})
            axes[axis] = {
                "tested": True, "count": len(results),
                "pass": len(results) == len(items) and all(row["status"] == "predicted" and row["ape_percent"] <= 100 * RELATIVE_ERROR_LIMIT + 1e-10 for row in results),
                "max_ape_percent": max((row["ape_percent"] for row in results if row["status"] == "predicted"), default=None),
                "cases": results,
            }
        tested = [value for value in axes.values() if value["tested"]]
        cv[key] = {
            "rule": "withhold each complete sequence pair, query-head count and batch size in separate calibration-only evaluations",
            "count": sum(value["count"] for value in tested),
            "pass": key in models and models[key]["feature_rank"] == models[key]["feature_count"] and axes["sequence_pair"]["tested"] and axes["heads"]["tested"] and all(value["pass"] for value in tested),
            "max_ape_percent": max((value["max_ape_percent"] for value in tested if value["max_ape_percent"] is not None), default=None),
            "axes": axes,
        }
    artifact["calibration_grouped_cv"] = cv
    return artifact


def predict(inputs: dict, model: dict) -> dict:
    if model.get("version") != VERSION or model.get("fit_method") != FIT_METHOD:
        raise ValueError("a frozen input-only FA latency model is required")
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
    for n, s in itertools.product((16, 64), (3072, 6144)):
        add("calibration", 1, n, n, s, s, 192, 128, "BF16", "SBH")
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
