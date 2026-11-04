from __future__ import annotations
import hashlib
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from prediction.fa_latency_inputs import (
    ceil_div,
    align,
    sbh_address_cost,
    normalize,
    derive,
    verify_schedule,
    group_key,
    active_features,
    validate_design,
    predict,
    design,
    sha256,
    save_json,
)
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


def minimax_weights(matrix: np.ndarray) -> np.ndarray:
    count, width = matrix.shape
    constraints = np.vstack((
        np.column_stack((matrix, -np.ones(count))),
        np.column_stack((-matrix, -np.ones(count))),
        -np.eye(width + 1),
    ))
    bounds = np.concatenate((np.ones(count), -np.ones(count), np.zeros(width + 1)))
    best_error, best = math.inf, []
    for active in itertools.combinations(range(len(bounds)), width + 1):
        indices = list(active)
        try:
            solution = np.linalg.solve(constraints[indices], bounds[indices])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(solution)) or np.any(constraints @ solution - bounds > 1e-8):
            continue
        weights = np.maximum(solution[:-1], 0.0)
        error = float(np.max(np.abs(matrix @ weights - 1)))
        if error < best_error - 1e-9:
            best_error, best = error, [weights]
        elif abs(error - best_error) <= 1e-9 and not any(np.allclose(weights, item, rtol=1e-8, atol=1e-9) for item in best):
            best.append(weights)
    if not best:
        raise ValueError("no feasible nonnegative worst-error cost fit")
    return np.mean(best, axis=0)


def fit_group(rows: list[dict], hardware: dict) -> dict:
    active = active_features(rows[0]["inputs"], hardware)
    x = np.asarray([[derive(row["inputs"], hardware)["features"][f] for f in active] for row in rows])
    y = np.asarray([float(row["duration_us"]) for row in rows])
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("calibration durations must be finite and positive")
    relative = x / y[:, None]
    scales = np.maximum(np.linalg.norm(relative, axis=0), np.finfo(float).tiny)
    matrix = relative / scales
    if group_key(rows[0]["inputs"], hardware) == "same_ab/BF16/SBH":
        coefficients = minimax_weights(matrix) / scales
        objective = "minimize maximum absolute relative calibration error; average optimal vertices when nonunique"
    else:
        coefficients = None
        objective = "minimize sum of squared relative calibration errors"
    best = None
    for flags in itertools.product((False, True), repeat=len(active)) if coefficients is None else ():
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
    if coefficients is None:
        coefficients = best[1]
    return {
        "coefficients": {**dict.fromkeys(FEATURE_NAMES, 0.0), **dict(zip(active, coefficients.tolist()))},
        "objective": objective,
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
        "target": "median msprof FlashAttentionScore Task Duration(us), profiler kernel replay, frequency warmup, fixed compiled operator",
        "scope": "dense official FA; no PSE/attention mask/sparse mode; scale=1/sqrt(D); dropout keep_prob=0.9 or 1",
        "method": "nonnegative source-work costs; SBH uses worst-relative-error fitting, other groups use relative-error least squares; no per-shape corrections",
        "cache_model": "SBH active-head and query-block address distribution with three-stage workspace; finite-cache reload cost",
        "address_model": dict(ADDRESS_MODEL),
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
