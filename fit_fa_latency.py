from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import inspect
import itertools
import json
import math
import statistics
from collections import Counter, deque
from pathlib import Path

import numpy as np

import fa_latency_inputs as input_cost


MODEL_VERSION = "fa_latency_model_v1"
INPUT_FAMILY_VERSION = "fa_input_family_v1"
PIPELINE_VERSION = "fa_input_pipeline_v1"
PIPELINE_PIPES = {
    "cube": ("aic_cube_active_sum_us", ("flops_m", "mma_instructions"), (0, 2)),
    "mte1": ("aic_mte1_active_sum_us", ("l0_kb", "calls"), (0, 2)),
    "mte2": ("aic_mte2_active_sum_us", ("read_kb", "calls"), (0, 2)),
    "fixpipe": ("aic_fixpipe_active_sum_us", ("write_kb", "calls"), (0, 2)),
    "vector": ("aiv_vec_active_sum_us", ("elements_k", "reduction_rows_k", "generic_rows_k", "calls"), (1, 3)),
    "vmte2": ("aiv_mte2_active_sum_us", ("read_kb", "calls"), (1, 3)),
    "vmte3": ("aiv_mte3_active_sum_us", ("write_kb", "calls"), (1, 3)),
}
PIPELINE_TERMS = ("initialization_us", "pipeline_us", "active_cube_pairs", "adapter_passes")
SHAPE_FIELDS = ("B", "N1", "N2", "S1", "S2", "D", "DV", "dtype", "layout", "has_drop_mask")
FAMILIES = {"same_ab": 95, "s1": 97, "b": 98, "b_drop": 98}
FEATURES = {
    "critical_work": "critical_flops_m",
    "total_work": "total_flops_m",
    "constant": None,
}
SOURCE_ROOT = "attention/flash_attention_score/"
SOURCE_EVIDENCE = {
    "same_ab": {
        "file": SOURCE_ROOT + "op_kernel/arch32/flash_attention_score_s1s2_bn2gs1_sab.h",
        "line": 810,
        "rule": "cubeBlockIdx * splitFactorSize; sequential Q tasks and S2 windows; two drain iterations",
    },
    "s1": {
        "file": SOURCE_ROOT + "op_kernel/arch32/flash_attention_score_s1_bn2gs1.h",
        "line": 880,
        "rule": "blockIdx * splitFactorSize; Q-task pipeline; BMM1, Vec1, BMM2, Vec2",
    },
    "b": {
        "file": SOURCE_ROOT + "op_kernel/arch32/flash_attention_score_bn2gs1s2_b.h",
        "line": 544,
        "rule": "batch tasks distributed between workers; heads and S1 blocks loop inside each task",
        "head_loop_line": 794,
    },
    "b_drop": {
        "file": SOURCE_ROOT + "op_host/arch32/flash_attention_score_tiling_general.cpp",
        "line": 1690,
        "rule": "B schedule plus dropout; S2 % 8 != 0 requires bit-to-byte mask adaptation",
    },
}
PREDICTION_FIELDS = (
    "family", "case", "split", "evaluation", "model_kind", "tiling_key",
    *SHAPE_FIELDS, "actual_us", "predicted_us", "abs_error_us", "ape_percent",
    "signed_error_percent", "feature_extrapolation", "tiling_extrapolation", "sample_cv_percent",
    "median_ci95_low_us", "median_ci95_high_us", "total_flops_m", "critical_flops_m",
    "parallel_workers", "critical_tasks", "critical_q_rows", "s2_windows",
    "critical_tile_visits", "drop_adapter", "correctness_status",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def integer(value, name: str) -> int:
    parsed = positive(value, name)
    if not parsed.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(parsed)


def shape_identity(row: dict) -> tuple:
    return tuple(str(row[name]) for name in SHAPE_FIELDS)


def repetition_features(row: dict) -> dict:
    family = row["family"]
    if family not in FAMILIES or int(row["priority"]) != FAMILIES[family]:
        raise ValueError("unsupported family or Host route")
    b, n1, n2, s1, s2, d, dv = (integer(row[name], name) for name in SHAPE_FIELDS[:7])
    if n1 % n2 or row["layout"] not in ("BNSD", "SBH"):
        raise ValueError("unsupported heads or layout")
    e = row["effective_tiling"]
    if isinstance(e, str):
        e = json.loads(e)
    if int(e["FA_BMM_PRESET"]) != 0:
        raise ValueError("the family models are calibrated for official BMM tiling only")
    total = integer(e["FA_TOTAL_TASKS"], "FA_TOTAL_TASKS")
    split = integer(e["FA_SPLIT_FACTOR"], "FA_SPLIT_FACTOR")
    base = integer(e["FA_S1_BASE"], "FA_S1_BASE")
    outer = integer(e["FA_S1_OUTER"], "FA_S1_OUTER")
    if outer != math.ceil(s1 / base):
        raise ValueError("S1 outer count disagrees with the input and tiling")
    critical_tasks = min(total, split)
    workers = math.ceil(total / split)
    cores = integer(e["FA_CORE_NUM"], "FA_CORE_NUM")
    if cores != workers * (2 if family == "same_ab" else 1):
        raise ValueError("worker/core mapping differs from the supported dense schedule")
    if family in ("b", "b_drop"):
        if total != b:
            raise ValueError("B model requires one batch per task")
        critical_q_rows = critical_tasks * n1 * s1
        visits = critical_tasks * n1 * outer
        s2_windows = 1
    else:
        if total != b * n1 * outer:
            raise ValueError("Q-task count disagrees with the input and tiling")
        tail = s1 - (outer - 1) * base

        def prefix_rows(tasks: int) -> int:
            groups, remainder = divmod(tasks, outer)
            return groups * s1 + remainder * base

        critical_q_rows = max(
            prefix_rows(min(start + split, total)) - prefix_rows(start)
            for start in range(0, total, split)
        )
        if tail <= 0 or tail > base:
            raise ValueError("invalid S1 tail")
        s2_windows = (
            math.ceil(s2 / (integer(e["FA_S2_BASE"], "FA_S2_BASE") * integer(e["FA_N_RATIO"], "FA_N_RATIO")))
            if family == "same_ab" else 1
        )
        visits = critical_tasks * s2_windows
    total_flops = 2 * b * n1 * s1 * s2 * (d + dv)
    for field, expected in (
        ("bmm1_flops", 2 * b * n1 * s1 * s2 * d),
        ("bmm2_flops", 2 * b * n1 * s1 * s2 * dv),
    ):
        if field in row and row[field] != "" and int(row[field]) != expected:
            raise ValueError(f"{field} disagrees with the input")
    return {
        "total_flops_m": total_flops / 1e6,
        "critical_flops_m": 2 * critical_q_rows * s2 * (d + dv) / 1e6,
        "parallel_workers": workers,
        "critical_tasks": critical_tasks,
        "critical_q_rows": critical_q_rows,
        "s2_windows": s2_windows,
        "critical_tile_visits": visits,
        "drop_adapter": int(int(row["has_drop_mask"]) != 0 and s2 % 8 != 0),
    }


def sample_summary(samples: list[float], seed: int) -> dict:
    values = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(2000, len(values))), axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return {
        "sample_cv_percent": float(np.std(values) / np.mean(values) * 100),
        "median_ci95_low_us": float(low),
        "median_ci95_high_us": float(high),
    }


def load_rows(paths: list[Path]) -> tuple[list[dict], dict]:
    records = {}
    skipped = Counter()
    rejected = []
    successful_records = 0
    for path in paths:
        with path.open(newline="") as source:
            for index, raw in enumerate(csv.DictReader(source), 2):
                if raw.get("status") != "ok":
                    skipped[raw.get("status", "missing")] += 1
                    continue
                try:
                    row = dict(raw)
                    if None in row or row["split"] not in ("calibration", "holdout", "external"):
                        raise ValueError("invalid CSV record or split")
                    samples = [positive(x, "sample") for x in json.loads(row["samples_us"])]
                    if len(samples) < 2 or len(samples) != integer(row["sample_count"], "sample_count"):
                        raise ValueError("sample count mismatch")
                    duration = positive(row["duration_us"], "duration_us")
                    if not math.isclose(duration, statistics.median(samples), rel_tol=1e-8, abs_tol=1e-5):
                        raise ValueError("duration is not the median of the recorded samples")
                    row.update(repetition_features(row))
                    row["actual_us"] = duration
                    identity = shape_identity(row)
                    previous = records.get(identity)
                    if previous is not None and (previous["split"], previous["family"]) != (row["split"], row["family"]):
                        raise ValueError("the same complete shape appears in different splits or families")
                    row.update(sample_summary(samples, int(hashlib.sha256(repr(identity).encode()).hexdigest()[:8], 16)))
                    row["correctness_status"] = "not_checked_by_collector"
                    records[identity] = row
                    successful_records += 1
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    rejected.append({"file": str(path), "line": index, "error": str(error)})
    return list(records.values()), {
        "input_files": [{"path": str(path), "sha256": digest(path)} for path in paths],
        "unique_shapes": len(records),
        "replaced_successful_retries": successful_records - len(records),
        "skipped_status_counts": dict(skipped),
        "rejected_rows": rejected,
    }


def fit_model(rows: list[dict], kind: str = "critical_work") -> dict:
    feature = FEATURES[kind]
    y = np.asarray([row["actual_us"] for row in rows])
    if feature is None:
        intercept, coefficient = float(np.mean(y)), 0.0
    else:
        x = np.asarray([row[feature] for row in rows])
        centered = x - np.mean(x)
        denominator = float(centered @ centered)
        coefficient = max(0.0, float(centered @ (y - np.mean(y))) / denominator) if denominator > 0 else 0.0
        intercept = float(np.mean(y) - coefficient * np.mean(x))
        if intercept < 0:
            intercept = 0.0
            coefficient = max(0.0, float(x @ y) / float(x @ x))
    return {
        "kind": kind,
        "feature": feature,
        "intercept_us": intercept,
        "coefficient_us_per_mflop": coefficient,
        "feature_range": [min(row[feature] for row in rows), max(row[feature] for row in rows)] if feature else [],
        "training_cases": sorted(row["case"] for row in rows),
        "training_shapes": [list(shape_identity(row)) for row in rows],
        "training_tilings": sorted({tiling_identity(row) for row in rows}),
        "scope": {
            field: sorted({row[field] for row in rows})
            for field in ("family", "priority", "tiling_key", "dtype", "layout", "D", "DV", "has_drop_mask", "soc_unit", "npu_id")
        },
    }


def tiling_identity(row: dict) -> str:
    effective = row["effective_tiling"]
    if isinstance(effective, str):
        effective = json.loads(effective)
    return ",".join(str(effective[name]) for name in ("FA_S1_BASE", "FA_S2_BASE", "FA_N_RATIO", "FA_BMM_PRESET"))


def predict_us(row: dict, model: dict) -> float:
    for field, allowed in model["scope"].items():
        if str(row[field]) not in {str(value) for value in allowed}:
            raise ValueError(f"{field}={row[field]} is outside the trained scope")
    feature = model["feature"]
    return model["intercept_us"] + model["coefficient_us_per_mflop"] * (row[feature] if feature else 0)


def predict_latency(inputs: dict, artifact: dict) -> float:
    from fa_latency_inputs import VERSION, predict

    if artifact.get("version") == PIPELINE_VERSION:
        return predict_pipeline(inputs, artifact)["predicted_us"]
    if artifact.get("version") == INPUT_FAMILY_VERSION:
        return predict_input_family(inputs, artifact)["predicted_us"]
    if artifact.get("version") != VERSION:
        raise ValueError("legacy effective-tiling models do not support input-only prediction")
    return predict(inputs, artifact)["predicted_us"]


def input_family_source(family: str) -> str:
    sources = [inspect.getsource(input_family_features), inspect.getsource(predict_input_family)]
    if family == "b_drop":
        sources.append(inspect.getsource(input_adapter_passes))
    return input_cost.sha256({"family": family, "version": INPUT_FAMILY_VERSION, "prediction": sources, "schedule": Path(input_cost.__file__).read_text()})


def input_adapter_passes(p: dict, hardware: dict) -> int:
    if p["S2"] % 8 == 0:
        return 0
    bits = p["B"] * p["N1"] * p["S1"] * p["S2"]
    single = input_cost.ceil_div(input_cost.ceil_div(bits, 2048), int(hardware["aiv_cores"])) * 2048
    available = input_cost.ceil_div(int(hardware["ub_bytes"]), 6) // 2048 * 2048
    if available <= 0:
        raise ValueError("hardware UB is too small for the official DropMask adapter")
    return input_cost.ceil_div(single, min(available, single))


def input_family_features(inputs: dict, hardware: dict) -> dict:
    p = input_cost.normalize(inputs)
    schedule = input_cost.derive(p, hardware)
    if schedule["family"] not in ("s1", "b", "b_drop"):
        raise ValueError("this model is restricted to S1, B and B DropMask")
    names = input_cost.ACTIVE_FEATURES[schedule["family"]]
    features = {name: schedule["features"][name] for name in names if name != "adapter_melements"}
    if schedule["family"] == "b_drop":
        features["adapter_passes"] = float(input_adapter_passes(p, hardware))
    return features


def fit_input_family_group(rows: list[dict], hardware: dict) -> dict:
    names = list(input_family_features(rows[0]["inputs"], hardware))
    x = np.asarray([[input_family_features(row["inputs"], hardware)[name] for name in names] for row in rows])
    y = np.asarray([float(row["duration_us"]) for row in rows])
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("invalid calibration durations")
    relative = x / y[:, None]
    scales = np.maximum(np.linalg.norm(relative, axis=0), np.finfo(float).tiny)
    matrix = relative / scales
    best = None
    for flags in itertools.product((False, True), repeat=len(names)):
        selected = np.flatnonzero(flags)
        weights = np.zeros(len(names))
        if len(selected):
            weights[selected] = np.linalg.lstsq(matrix[:, selected], np.ones(len(rows)), rcond=None)[0]
        if np.any(weights < -1e-10):
            continue
        weights = np.maximum(weights, 0)
        loss = float(np.sum((matrix @ weights - 1) ** 2))
        if best is None or loss < best[0]:
            best = loss, weights / scales
    return {
        "coefficients": dict(zip(names, best[1].tolist())),
        "feature_rank": int(np.linalg.matrix_rank(x / np.maximum(x.max(axis=0), 1e-12))),
        "feature_count": len(names),
        "calibration_cases": [row["case"] for row in rows],
        "input_ranges": {name: [min(row["inputs"][name] for row in rows), max(row["inputs"][name] for row in rows)] for name in input_cost.INPUT_FIELDS[:7]},
    }


def predict_input_family(inputs: dict, model: dict) -> dict:
    if model.get("version") != INPUT_FAMILY_VERSION:
        raise ValueError("an independent frozen family model is required")
    p = input_cost.normalize(inputs)
    if input_cost.derive(p, model["hardware"])["family"] != model["family"]:
        raise ValueError("operator inputs select a different family")
    group = model["groups"][model["group"]]
    features = input_family_features(p, model["hardware"])
    components = {name: value * features[name] for name, value in group["coefficients"].items()}
    duration = sum(components.values())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid family prediction")
    cv = model.get("calibration_grouped_cv", {})
    return {
        "predicted_us": duration, "group": model["group"], "cost_terms_us": components,
        "prediction_quality": "calibration_cv_pass" if cv and all(value["pass"] for value in cv.values()) else "provisional",
        "extrapolated_inputs": [name for name, limits in group["input_ranges"].items() if not limits[0] <= p[name] <= limits[1]],
    }


def fit_input_family(rows: list[dict], hardware: dict, family: str, signature: str) -> dict:
    calibration = [row for row in rows if row["family"] == family and row["split"] == "calibration" and row["status"] == "ok" and str(row.get("stable")).lower() == "true"]
    if len(calibration) < 8:
        raise ValueError(f"{family} has fewer than 8 stable calibration shapes")
    group = input_cost.group_key(calibration[0]["inputs"], hardware)
    if any(input_cost.group_key(row["inputs"], hardware) != group for row in calibration):
        raise ValueError("family calibration mixes dtype or layout")
    fitted = fit_input_family_group(calibration, hardware)
    model = {
        "version": INPUT_FAMILY_VERSION, "family": family, "group": group,
        "plan_sha256": signature, "source_sha256": input_family_source(family),
        "hardware": hardware, "groups": {group: fitted},
        "source_evidence": SOURCE_EVIDENCE[family],
        "target": "median msprof FlashAttentionScore Task Duration(us); official tiling; independent family calibration",
        "calibration_grouped_cv": {},
    }
    axes = {}
    for axis, fields in (("sequence_pair", ("S1", "S2")), ("heads", ("N1",)), ("batch", ("B",))):
        identity = lambda row: tuple(row["inputs"][name] for name in fields)
        folds = sorted({identity(row) for row in calibration})
        results = []
        for fold in folds if len(folds) > 1 else ():
            training = [row for row in calibration if identity(row) != fold]
            if len(training) < 3:
                continue
            fitted_fold = fit_input_family_group(training, hardware)
            candidate = {**model, "groups": {group: fitted_fold}}
            for row in calibration:
                if identity(row) != fold:
                    continue
                if fitted_fold["feature_rank"] < fitted_fold["feature_count"]:
                    results.append({"case": row["case"], "status": "unidentifiable_cost_terms"})
                    continue
                predicted = predict_input_family(row["inputs"], candidate)["predicted_us"]
                results.append({"case": row["case"], "status": "predicted", "predicted_us": predicted, "actual_us": float(row["duration_us"]), "ape_percent": abs(predicted / float(row["duration_us"]) - 1) * 100})
        axes[axis] = {
            "tested": len(folds) > 1, "count": len(results), "cases": results,
            "pass": len(results) == len(calibration) and all(row["status"] == "predicted" and row["ape_percent"] <= 10 for row in results),
            "max_ape_percent": max((row["ape_percent"] for row in results if row["status"] == "predicted"), default=None),
        }
    model["calibration_grouped_cv"][group] = {
        "axes": axes, "pass": fitted["feature_rank"] == fitted["feature_count"] and all(value["pass"] for value in axes.values()),
        "max_ape_percent": max((value["max_ape_percent"] for value in axes.values() if value["max_ape_percent"] is not None), default=None),
    }
    return model


def pipeline_nnls(matrix: list, values: list) -> dict:
    x, y = np.asarray(matrix, dtype=float), np.asarray(values, dtype=float)
    if x.ndim != 2 or not len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("pipeline calibration contains missing or invalid counters")
    relative = x / y[:, None]
    scales = np.maximum(np.linalg.norm(relative, axis=0), np.finfo(float).tiny)
    a = relative / scales
    weights, passive = np.zeros(x.shape[1]), np.zeros(x.shape[1], dtype=bool)
    for iteration in range(20 * x.shape[1] + 100):
        gradient = a.T @ (np.ones(len(y)) - a @ weights)
        gradient[passive] = -np.inf
        if np.max(gradient) <= 1e-10:
            break
        passive[int(np.argmax(gradient))] = True
        for inner in range(20 * x.shape[1] + 100):
            candidate = np.zeros(x.shape[1])
            candidate[passive] = np.linalg.lstsq(a[:, passive], np.ones(len(y)), rcond=None)[0]
            if np.all(candidate[passive] > 0):
                weights = candidate
                break
            removed = passive & (candidate <= 0)
            denominator = weights[removed] - candidate[removed]
            alpha = min((weights[removed][denominator > 0] / denominator[denominator > 0]).tolist(), default=0.0)
            weights += alpha * (candidate - weights)
            passive[passive & (weights <= 1e-12)] = False
        else:
            raise RuntimeError("pipeline nonnegative calibration did not converge")
    else:
        raise RuntimeError("pipeline nonnegative calibration did not converge")
    active = np.any(x != 0, axis=0)
    return {
        "weights": (np.maximum(weights, 0) / scales).tolist(),
        "rank": int(np.linalg.matrix_rank(a[:, active])) if np.any(active) else 0,
        "terms": int(np.count_nonzero(active)),
        "observed": active.tolist(),
    }


def pipeline_job(p: dict, rows: int, heads: int, kv_heads: int, s1_base: int, family: str) -> list[dict]:
    s2, d, dv = input_cost.align(p["S2"]), p["D"], p["DV"]
    aligned_rows = input_cost.align(rows)
    vector_s2 = input_cost.align(p["S2"], 64) if family == "s1" else s2
    vec1_base = min(s1_base, 1024 // vector_s2 * 8) if family == "s1" else s1_base
    vec2_base = min(input_cost.align(p["S1"]), 8192 // input_cost.align(dv) // 16 * 16)
    if min(vec1_base, vec2_base) <= 0:
        raise ValueError("unsupported official Vector block geometry")
    generic_rows = sum(min(vec1_base, rows - start) for start in range(0, rows, vec1_base)
                       if family != "s1" and (min(vec1_base, rows - start) % 8 or p["S2"] % 64))
    score_kb = 4 * heads * rows * s2 / 1024
    return [
        {"flops_m": 2 * heads * aligned_rows * s2 * d / 1e6, "query_rows": heads * rows,
         "mma_instructions": float(heads * input_cost.ceil_div(aligned_rows, 128) * input_cost.ceil_div(s2, 128) * input_cost.ceil_div(d, 128)),
         "read_kb": 2 * (heads * rows * d + kv_heads * p["S2"] * d) / 1024,
         "l0_kb": 2 * heads * (aligned_rows * d * input_cost.ceil_div(s2, 128) + s2 * d * input_cost.ceil_div(aligned_rows, 128)) / 1024,
         "write_kb": score_kb, "calls": 1.0},
        {"elements_k": heads * rows * vector_s2 / 1000, "reduction_rows_k": heads * rows / 1000,
         "generic_rows_k": heads * generic_rows / 1000,
         "read_kb": score_kb, "write_kb": heads * rows * (2 * s2 + 64) / 1024,
         "calls": float(heads * input_cost.ceil_div(rows, vec1_base))},
        {"flops_m": 2 * heads * aligned_rows * s2 * dv / 1e6,
         "mma_instructions": float(heads * input_cost.ceil_div(aligned_rows, 128) * input_cost.ceil_div(dv, 128) * input_cost.ceil_div(s2, 128)),
         "read_kb": 2 * (heads * rows * s2 + kv_heads * p["S2"] * dv) / 1024,
         "l0_kb": 2 * heads * (aligned_rows * s2 * input_cost.ceil_div(dv, 128) + s2 * dv * input_cost.ceil_div(aligned_rows, 128)) / 1024,
         "write_kb": 4 * heads * rows * dv / 1024, "calls": 1.0},
        {"elements_k": heads * rows * dv / 1000, "reduction_rows_k": 0.0, "generic_rows_k": 0.0,
         "read_kb": heads * rows * (4 * dv + 32) / 1024,
         "write_kb": 2 * heads * rows * dv / 1024,
         "calls": float(heads * input_cost.ceil_div(rows, vec2_base))},
    ]


def pipeline_design(inputs: dict, hardware: dict) -> dict:
    p = input_cost.normalize(inputs)
    schedule = input_cost.derive(p, hardware)
    family = schedule["family"]
    if family not in ("s1", "b", "b_drop") or int(hardware["aiv_cores"]) != 2 * int(hardware["aic_cores"]):
        raise ValueError("pipeline model requires the supported dense S1/B kernels and two Vector cores per Cube")
    workers = []
    for start in range(0, schedule["total_tasks"], schedule["split_factor"]):
        tasks = []
        for task in range(start, min(start + schedule["split_factor"], schedule["total_tasks"])):
            rows = min(schedule["s1_base"], p["S1"] - task % schedule["s1_outer"] * schedule["s1_base"]) if family == "s1" else p["S1"]
            tasks.append(pipeline_job(p, rows, 1 if family == "s1" else p["N1"], 1 if family == "s1" else p["N2"], schedule["s1_base"], family))
        workers.append(tasks)
    return {"inputs": p, "schedule": schedule, "workers": workers,
            "active_cube_pairs": input_cost.ceil_div(len(workers), 2),
            "critical_cube_rows": max(sum(job[0]["query_rows"] for tasks in workers[first:first + 2] for job in tasks) for first in range(0, len(workers), 2)),
            "mma_instructions": sum(job[stage]["mma_instructions"] for tasks in workers for job in tasks for stage in (0, 2)),
            "adapter_passes": input_adapter_passes(p, hardware) if family == "b_drop" else 0}


def pipeline_service(job: dict, pipe: str, model: dict) -> float:
    return sum(value * job.get(name, 0.0) for name, value in model["pipes"][pipe]["coefficients"].items())


def pipeline_schedule(design: dict, model: dict) -> dict:
    pair_results = []
    for first in range(0, len(design["workers"]), 2):
        workers = design["workers"][first:first + 2]
        programs = []
        for tasks in workers:
            program = []
            for step in range(len(tasks) + 2):
                if 0 <= step - 1 < len(tasks):
                    program.append(("wait", step - 1, 0))
                if step < len(tasks):
                    program.append(("launch", step, 0))
                if 0 <= step - 1 < len(tasks):
                    program.append(("vector", step - 1, 1))
                if 0 <= step - 2 < len(tasks):
                    program.append(("wait", step - 2, 2))
                if 0 <= step - 1 < len(tasks):
                    program.append(("launch", step - 1, 2))
                if 0 <= step - 2 < len(tasks):
                    program.append(("vector", step - 2, 3))
            programs.append(program)
        positions, blocked, finished = [0] * len(workers), {}, {}
        completion, queue, events = {}, deque(), []
        stage_totals = dict.fromkeys(("bmm1", "vec1", "bmm2", "vec2"), 0.0)
        serial, cube_busy = 0, False

        def event(time, kind, worker, task=-1, stage=-1):
            nonlocal serial
            serial += 1
            heapq.heappush(events, (time, serial, kind, worker, task, stage))

        def start_cube(time):
            nonlocal cube_busy
            if cube_busy or not queue:
                return
            worker, task, stage = queue.popleft()
            job = workers[worker][task][stage]
            duration = max(pipeline_service(job, pipe, model) for pipe in ("cube", "mte1", "mte2")) + pipeline_service(job, "fixpipe", model)
            stage_totals[("bmm1", "vec1", "bmm2", "vec2")[stage]] += duration
            cube_busy = True
            event(time + duration, "cube", worker, task, stage)

        def advance(worker, time):
            while positions[worker] < len(programs[worker]):
                kind, task, stage = programs[worker][positions[worker]]
                if kind == "wait" and (worker, task, stage) not in completion:
                    blocked[worker] = (worker, task, stage)
                    return
                positions[worker] += 1
                if kind == "launch":
                    queue.append((worker, task, stage))
                    start_cube(time)
                elif kind == "vector":
                    job = workers[worker][task][stage]
                    duration = max(pipeline_service(job, "vector", model), pipeline_service(job, "vmte2", model) + pipeline_service(job, "vmte3", model))
                    stage_totals[("bmm1", "vec1", "bmm2", "vec2")[stage]] += duration
                    event(time + duration, "vector", worker)
                    return
            finished[worker] = time

        for worker in range(len(workers)):
            advance(worker, 0.0)
        while events:
            time, _, kind, worker, task, stage = heapq.heappop(events)
            if kind == "cube":
                completion[worker, task, stage] = time
                cube_busy = False
                start_cube(time)
                if blocked.get(worker) == (worker, task, stage):
                    blocked.pop(worker)
                    advance(worker, time)
            else:
                advance(worker, time)
        if len(finished) != len(workers) or queue or blocked:
            raise RuntimeError("pipeline dependency schedule did not drain")
        pair_results.append({"pipeline_us": max(finished.values()), "critical_pair": first // 2,
                             "stage_service_us": stage_totals})
    return max(pair_results, key=lambda result: result["pipeline_us"])


def fit_pipeline_group(rows: list[dict], hardware: dict, family: str) -> dict:
    if len(rows) < 3 or any(input_cost.derive(row["inputs"], hardware)["family"] != family for row in rows):
        raise ValueError("pipeline fitting needs at least three measurements from one supported family")
    design = [pipeline_design(row["inputs"], hardware) for row in rows]
    model = {"version": PIPELINE_VERSION, "family": family, "hardware": hardware, "pipes": {}}
    for pipe, (target, names, stages) in PIPELINE_PIPES.items():
        selected = [index for index, item in enumerate(design) if not item["adapter_passes"]]
        if len(selected) < 3:
            raise ValueError("pipeline calibration needs aligned-mask examples with measured pipe counters")
        matrix = [[sum(job[stage].get(name, 0.0) for tasks in design[index]["workers"] for job in tasks for stage in stages) for name in names] for index in selected]
        fitted = pipeline_nnls(matrix, [float(rows[index][target]) for index in selected])
        model["pipes"][pipe] = {"coefficients": dict(zip(names, fitted.pop("weights"))), **fitted}
    matrix = [[1.0, pipeline_schedule(item, model)["pipeline_us"], item["active_cube_pairs"], item["adapter_passes"]] for item in design]
    fitted = pipeline_nnls(matrix, [float(row["duration_us"]) for row in rows])
    model.update(coefficients=dict(zip(PIPELINE_TERMS, fitted.pop("weights"))), fit=fitted,
                 calibration_cases=[row["case"] for row in rows],
                 input_ranges={name: [min(row["inputs"][name] for row in rows), max(row["inputs"][name] for row in rows)] for name in input_cost.INPUT_FIELDS[:7]},
                 adapter_pass_range=[min(item["adapter_passes"] for item in design), max(item["adapter_passes"] for item in design)],
                 target="median msprof OpBasicInfo Task Duration(us); official dense FP16 BNSD tiling",
                 rules={"mapping": "two adjacent Vector clients share one Cube; contiguous Host task partitions with actual S1 tails",
                        "dependencies": "BMM1 -> Vec1 -> BMM2 -> Vec2; asynchronous Matmul FIFO, ping-pong waits and two drain iterations",
                        "bmm_service": "max(Cube, MTE1, MTE2) + Fixpipe, calibrated from aggregate per-pipe active times",
                        "vector_service": "max(Vector, MTE2 + MTE3); S1 64-column alignment; B Softmax generic/basic-block branch",
                        "latency": "initialization + fitted intra-call overlap scale * scheduled elapsed + active-pair setup + adapter passes",
                        "stage_totals": "aggregate service in the critical Cube pair, before overlap scaling; do not add to the final elapsed time",
                        "adapter": "first-pass synchronization and unpacking are jointly identified; extra-pass behavior is not independently calibrated",
                        "memory": "GQA K/V counts, FP16 operands and probabilities, FP32 BMM intermediates and statistics; L0 tile model uses 128-sized tiles",
                        "limits": "no same_ab, backward, TND, sparse, L1 reuse or arbitrary candidate tiling coverage"},
                 source_evidence={**SOURCE_EVIDENCE[family],
                                  "softmax": SOURCE_ROOT + "op_kernel/arch32/flash_attention_score_common.h:143",
                                  "intermediate_dtype": SOURCE_ROOT + f"op_kernel/flash_attention_score.cpp:{557 if family == 's1' else 642}",
                                  "adapter": SOURCE_ROOT + "op_kernel/arch32/flash_attention_score_drop_mask_adapter.h:90"},
                 qualification="development_model_requires_fresh_blind_validation")
    return model


def predict_pipeline(inputs: dict, model: dict) -> dict:
    if model.get("version") != PIPELINE_VERSION:
        raise ValueError("a calibrated pipeline model is required")
    if "families" in model:
        families = model["families"]
        if not families:
            raise ValueError("the artifact contains no calibrated pipeline family")
        family = input_cost.derive(inputs, next(iter(families.values()))["hardware"])["family"]
        if family not in families:
            raise ValueError(f"the artifact has no calibrated {family} pipeline model")
        model = families[family]
    design = pipeline_design(inputs, model["hardware"])
    if design["schedule"]["family"] != model["family"]:
        raise ValueError("operator inputs select another family")
    if design["adapter_passes"] > model["adapter_pass_range"][1]:
        raise ValueError("additional DropMask passes have not been independently calibrated")
    schedule = pipeline_schedule(design, model)
    raw = dict(zip(PIPELINE_TERMS, (1.0, schedule["pipeline_us"], design["active_cube_pairs"], design["adapter_passes"])))
    components = {name: value * raw[name] for name, value in model["coefficients"].items()}
    if any(raw[name] and not observed for name, observed in zip(PIPELINE_TERMS, model["fit"]["observed"])):
        raise ValueError("prediction needs a cost term absent from this calibration fold")
    duration = sum(components.values())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid pipeline prediction")
    return {"predicted_us": duration, "cost_terms_us": components, **schedule,
            "qualification": model["qualification"],
            "active_vector_workers": len(design["workers"]), "active_cube_pairs": design["active_cube_pairs"],
            "critical_cube_rows": design["critical_cube_rows"], "scheduled_tasks": design["schedule"]["total_tasks"],
            "mma_instructions": design["mma_instructions"],
            "extrapolated_inputs": [name for name, bounds in model["input_ranges"].items() if not bounds[0] <= design["inputs"][name] <= bounds[1]],
            "adapter_extrapolated": not model["adapter_pass_range"][0] <= design["adapter_passes"] <= model["adapter_pass_range"][1]}


def pipeline_counters_complete(row: dict) -> bool:
    try:
        return row.get("pmu_status") == "ok" and all(math.isfinite(float(row[target])) and float(row[target]) > 0 for target, _, _ in PIPELINE_PIPES.values())
    except (KeyError, TypeError, ValueError):
        return False


def pipeline_report(path: Path) -> dict:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        fields, flat = list(reader.fieldnames or []), list(reader)
    if not flat:
        raise ValueError("pipeline evaluation needs the existing diagnostic measurements")
    hardware_names = ("soc", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes", "ub_bytes")
    hardware = {name: flat[0][name] if name == "soc" else int(flat[0][name]) for name in hardware_names}
    hardware["calibrated_frequency_mhz"] = sorted({float(row[name]) for row in flat if row["split"] == "calibration" for name in ("freq_min_mhz", "freq_max_mhz") if row.get(name)})
    rows = []
    for index, row in enumerate(flat):
        if any(str(row[name]) != str(hardware[name]) for name in hardware_names):
            raise ValueError("pipeline calibration mixes hardware profiles")
        inputs = input_cost.normalize({name: row[name] for name in input_cost.INPUT_FIELDS})
        rows.append({**row, "inputs": inputs, "case": input_cost.sha256(inputs), "index": index})
    report = {"version": PIPELINE_VERSION, "training_rule": "stable original calibration measurements and PMU only; no holdout timing used for fitting",
              "evaluation_rule": "development replay of already exposed shapes plus grouped calibration cross-validation; not a new blind validation",
              "fresh_blind_pass": False, "families": {}}
    models, predictions = {}, {}
    for family in ("s1", "b", "b_drop"):
        candidates = [row for row in rows if row["family"] == family and row["split"] == "calibration" and row["status"] == "ok" and str(row["stable"]).lower() == "true"]
        calibration = [row for row in candidates if pipeline_counters_complete(row)]
        if len(calibration) < 8:
            report["families"][family] = {"error": "fewer than eight stable original calibration cases"}
            continue
        try:
            model = fit_pipeline_group(calibration, hardware, family)
            models[family] = model
            axes = {}
            for axis, keys in (("sequence_pair", ("S1", "S2")), ("heads", ("N1",)), ("batch", ("B",))):
                outcomes = []
                identity = lambda row: tuple(row["inputs"][key] for key in keys)
                for fold in sorted({identity(row) for row in calibration}):
                    training = [row for row in calibration if identity(row) != fold]
                    testing = [row for row in calibration if identity(row) == fold]
                    try:
                        fitted = fit_pipeline_group(training, hardware, family)
                        if fitted["fit"]["rank"] != fitted["fit"]["terms"]:
                            raise ValueError("held-out input group leaves unidentifiable latency terms")
                        for row in testing:
                            prediction = predict_pipeline(row["inputs"], fitted)
                            error = abs(prediction["predicted_us"] / float(row["duration_us"]) - 1) * 100
                            outcomes.append({"case": row["case"], "ape_percent": error})
                    except (ValueError, RuntimeError, KeyError) as error:
                        outcomes.extend({"case": row["case"], "error": str(error)} for row in testing)
                errors = [row["ape_percent"] for row in outcomes if "ape_percent" in row]
                axes[axis] = {"pass": len(errors) == len(calibration) and max(errors, default=math.inf) <= 10,
                              "max_ape_percent": max(errors, default=None), "cases": outcomes}
            replay, failed_predictions = [], {}
            for row in rows:
                if row["family"] != family or row["status"] != "ok":
                    continue
                try:
                    result = predict_pipeline(row["inputs"], model)
                except (ValueError, RuntimeError) as error:
                    failed_predictions[str(row["index"])] = str(error)
                    continue
                result["ape_percent"] = abs(result["predicted_us"] / float(row["duration_us"]) - 1) * 100
                predictions[row["index"]] = result
                if row["split"] == "holdout":
                    replay.append(result["ape_percent"])
            report["families"][family] = {"calibration_cases": len(calibration), "calibration_missing_counters": len(candidates) - len(calibration), "calibration_grouped_cv": axes,
                "calibration_cv_pass": all(item["pass"] for item in axes.values()),
                "unidentifiable_pipe_terms": [pipe for pipe, fit in model["pipes"].items() if fit["rank"] != fit["terms"]],
                "prediction_errors": failed_predictions,
                "replay_count": len(replay), "replay_point_error_pass": sum(value <= 10 for value in replay),
                "replay_mape_percent": statistics.mean(replay) if replay else None, "replay_max_ape_percent": max(replay, default=None)}
            print(f"PIPELINE {family}: existing_holdout_replay={sum(value <= 10 for value in replay)}/{len(replay)} max_ape_percent={max(replay, default=None)}; fresh_blind_pass=False", flush=True)
        except (ValueError, RuntimeError, KeyError) as error:
            report["families"][family] = {"error": str(error)}
            print(f"PIPELINE {family}: unavailable={error}; existing measurements retained", flush=True)
    output = path.parent
    input_cost.save_json(output / "pipeline_models.json", {"version": PIPELINE_VERSION, "families": models})
    input_cost.save_json(output / "pipeline_report.json", report)
    extra = ("pipeline_version", "pipeline_predicted_us", "pipeline_ape_percent", "pipeline_pass_10_percent", "pipeline_qualification", "pipeline_cv_pass",
             "pipeline_initialization_us", "pipeline_service_us", "pipeline_cube_pair_setup_us", "pipeline_adapter_us",
             "pipeline_bmm1_service_us", "pipeline_vec1_service_us", "pipeline_bmm2_service_us", "pipeline_vec2_service_us", "pipeline_raw_elapsed_us",
             "pipeline_active_vector_workers", "pipeline_active_cube_pairs", "pipeline_critical_cube_rows", "pipeline_scheduled_tasks", "pipeline_mma_instructions",
             "pipeline_extrapolated_inputs", "pipeline_adapter_extrapolated", "pipeline_critical_pair", "pipeline_error")
    fields.extend(name for name in extra if name not in fields)
    temporary = path.with_suffix(".pipeline.tmp")
    with temporary.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(flat):
            row.update({name: "" for name in extra})
            row["pipeline_version"] = PIPELINE_VERSION
            family = report["families"].get(row["family"], {})
            if index in predictions:
                result = predictions[index]
                row.update(pipeline_predicted_us=result["predicted_us"], pipeline_ape_percent=result["ape_percent"], pipeline_pass_10_percent=result["ape_percent"] <= 10,
                           pipeline_qualification=result["qualification"], pipeline_cv_pass=family["calibration_cv_pass"], pipeline_critical_pair=result["critical_pair"],
                           pipeline_raw_elapsed_us=result["pipeline_us"], pipeline_extrapolated_inputs="|".join(result["extrapolated_inputs"]),
                           pipeline_adapter_extrapolated=result["adapter_extrapolated"])
                for name in ("active_vector_workers", "active_cube_pairs", "critical_cube_rows", "scheduled_tasks", "mma_instructions"):
                    row[f"pipeline_{name}"] = result[name]
                for name, term in (("initialization", "initialization_us"), ("service", "pipeline_us"), ("cube_pair_setup", "active_cube_pairs"), ("adapter", "adapter_passes")):
                    row[f"pipeline_{name}_us"] = result["cost_terms_us"][term]
                for name, value in result["stage_service_us"].items():
                    row[f"pipeline_{name}_service_us"] = value
            else:
                row["pipeline_error"] = family.get("prediction_errors", {}).get(str(index), family.get("error", row.get("error", "")))
            for name in extra:
                if isinstance(row.get(name), float):
                    row[name] = format(row[name], ".6g")
            writer.writerow(row)
    temporary.replace(path)
    return report


def prediction(row: dict, model: dict, evaluation: str) -> dict:
    predicted = predict_us(row, model)
    actual = row["actual_us"]
    feature = model["feature"]
    extrapolation = bool(feature and not model["feature_range"][0] <= row[feature] <= model["feature_range"][1])
    return {
        **row,
        "evaluation": evaluation,
        "model_kind": model["kind"],
        "predicted_us": predicted,
        "abs_error_us": abs(predicted - actual),
        "ape_percent": abs(predicted - actual) / actual * 100,
        "signed_error_percent": (predicted - actual) / actual * 100,
        "feature_extrapolation": extrapolation,
        "tiling_extrapolation": tiling_identity(row) not in model["training_tilings"],
    }


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    error = [row["ape_percent"] for row in rows]
    return {
        "count": len(rows),
        "mape_percent": statistics.mean(error),
        "median_ape_percent": statistics.median(error),
        "max_ape_percent": max(error),
        "mae_us": statistics.mean(row["abs_error_us"] for row in rows),
        "rmse_us": math.sqrt(statistics.mean(row["abs_error_us"] ** 2 for row in rows)),
        "mean_signed_error_percent": statistics.mean(row["signed_error_percent"] for row in rows),
        "extrapolated_shapes": sum(row["feature_extrapolation"] for row in rows),
        "unseen_tiling_shapes": sum(row.get("tiling_extrapolation", False) for row in rows),
    }


def cross_validate(rows: list[dict], kind: str) -> list[dict]:
    predictions = []
    for held_out in rows:
        train = [row for row in rows if shape_identity(row) != shape_identity(held_out)]
        model = fit_model(train, kind)
        predictions.append(prediction(held_out, model, "calibration_leave_one_shape_out"))
    return predictions


def frozen_sameab(rows: list[dict], path: Path) -> tuple[list[dict], dict]:
    if not path.is_file():
        return [], {"status": "frozen_model_missing", "path": str(path)}
    source = json.loads(path.read_text())
    if source.get("model_version") != "sameab_latency_model_v1":
        raise ValueError("unsupported frozen SameAB model")
    predictions = []
    for row in rows:
        scope = source["scope"]
        if row["dtype"] != scope["dtype"] or int(row["D"]) != scope["D"] or int(row["DV"]) != scope["DV"] or row["layout"] not in scope["layouts"]:
            raise ValueError("SameAB observation is outside the frozen model scope")
        predicted = source["baseline_model"]["coefficient_us_per_flop"][row["layout"]] * row["total_flops_m"] * 1e6
        actual = row["actual_us"]
        predictions.append({
            **row,
            "evaluation": "external_frozen_model",
            "model_kind": "frozen_sameab_official_baseline",
            "predicted_us": predicted,
            "abs_error_us": abs(predicted - actual),
            "ape_percent": abs(predicted - actual) / actual * 100,
            "signed_error_percent": (predicted - actual) / actual * 100,
            "feature_extrapolation": "unknown_training_range",
        })
    report = metrics([{**row, "feature_extrapolation": False} for row in predictions])
    report.pop("extrapolated_shapes", None)
    return predictions, {"status": "frozen_external_evaluation", "path": str(path), "sha256": digest(path), "external": report}


def build_models(paths: list[Path], output_dir: Path, sameab_model: Path) -> dict:
    rows, audit = load_rows(paths)
    if not rows:
        raise RuntimeError("no valid latency records")
    print(f"DATA_AUDIT shapes={len(rows)} rejected={len(audit['rejected_rows'])} duplicate_retries={audit['replaced_successful_retries']}", flush=True)
    models = {}
    reports = {}
    predictions = []
    for family in ("s1", "b", "b_drop"):
        items = [row for row in rows if row["family"] == family]
        train = [row for row in items if row["split"] == "calibration"]
        test = [row for row in items if row["split"] == "holdout"]
        if len(train) < 3 or not test:
            reports[family] = {"status": "insufficient_calibration_or_holdout", "calibration_shapes": len(train), "holdout_shapes": len(test)}
            continue
        model = fit_model(train)
        holdout = [prediction(row, model, "holdout") for row in test]
        cv = cross_validate(train, "critical_work")
        calibration = [prediction(row, model, "calibration_in_sample") for row in train]
        models[family] = model
        predictions.extend(calibration + cv + holdout)
        ablations = {}
        for kind in ("constant", "total_work"):
            alternative = fit_model(train, kind)
            ablations[kind] = {
                "calibration_loso": metrics(cross_validate(train, kind)),
                "holdout": metrics([prediction(row, alternative, "holdout") for row in test]),
            }
        reports[family] = {
            "status": "experimental_requires_new_shape_validation",
            "calibration_shapes": len(train),
            "holdout_shapes": len(test),
            "calibration_in_sample": metrics(calibration),
            "calibration_loso": metrics(cv),
            "holdout": metrics(holdout),
            "ablations": ablations,
            "median_sample_cv_percent": statistics.median(row["sample_cv_percent"] for row in items),
            "holdout_used_for_training": False,
            "model_selected_using_holdout": False,
        }
        print(f"{family.upper()} train={len(train)} holdout={len(test)} LOSO_MAPE={metrics(cv)['mape_percent']:.3f}% HOLDOUT_MAPE={metrics(holdout)['mape_percent']:.3f}% MAX_APE={metrics(holdout)['max_ape_percent']:.3f}%", flush=True)
    sameab_rows = [row for row in rows if row["family"] == "same_ab"]
    if sameab_rows:
        external_predictions, reports["same_ab"] = frozen_sameab(sameab_rows, sameab_model)
        predictions.extend(external_predictions)
        if external_predictions:
            print(f"SAME_AB external={len(external_predictions)} FROZEN_MAPE={reports['same_ab']['external']['mape_percent']:.3f}%", flush=True)
    common = {
        "model_version": MODEL_VERSION,
        "ops_transformer_version": "v8.5.0",
        "target": "median stream Event interval around the ACLNN call, in microseconds",
        "formula": "predicted_us = intercept_us + coefficient_us_per_mflop * critical_flops_m",
        "feature_rule": "sum logical BMM1+BMM2 FLOPs in each actual contiguous task partition; take the maximum partition",
        "fit_rule": "nonnegative least squares with two coefficients; calibration shapes only",
        "validation_rule": "leave one complete calibration shape out; evaluate the predefined holdout separately",
        "source_evidence": SOURCE_EVIDENCE,
        "limitations": [
            "Event intervals can include stream idle time while Host launch work is submitted; these are not profiler kernel durations",
            "successful collector rows did not independently verify numerical output correctness",
            "short operator samples are noisy; increasing repeat count does not remove systematic launch bias",
            "only official tilings are covered; altered tiling configurations require separate calibration and validation",
            "S1 L1 reuse variants, unseen keys, layouts, dtypes and head dimensions are outside the trained scope",
            "fixed D/DV couples BMM, Softmax and memory work; their individual costs cannot be identified from these data",
            "DropMask adaptation is recorded but lacks enough independent calibration cases for a separate cost coefficient",
            "six calibration and two holdout shapes per new family do not establish general accuracy",
            "the declared split was respected, but these holdout measurements had already been inspected; next validation should use fresh shapes",
        ],
    }
    report = {**common, "data_audit": audit, "families": reports}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models.json").write_text(json.dumps({**common, "families": models}, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "predictions.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    print(f"FA_LATENCY_MODELS={output_dir / 'models.json'}")
    print(f"FA_LATENCY_REPORT={output_dir / 'report.json'}")
    print(f"FA_LATENCY_PREDICTIONS={output_dir / 'predictions.csv'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="*", type=Path, default=[Path("output/fa_working_routes_latency_v1/routes.csv")])
    parser.add_argument("--output-dir", type=Path, default=Path("output") / MODEL_VERSION)
    parser.add_argument("--sameab-model", type=Path, default=Path("output/sameab_latency_model_v1/model.json"))
    args = parser.parse_args()
    build_models(args.csv, args.output_dir, args.sameab_model)


if __name__ == "__main__":
    main()
