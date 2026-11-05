from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import collections
import csv
import gc
import hashlib
import json
import math
import resource
import signal
import statistics as summary_statistics
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls

from clean.runtime import fa_latency_transfer as transfer
from prediction import fa_latency_refinement as micro

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "research/reports"
CACHE = ROOT / "out/fa_joint_calibration"
BASE = ROOT / "out/fa_branch_host"
PIPES = tuple(micro.PIPES)
NAMES = micro.parameter_names()
ROUNDS = 12


def binding(cost_class, pipe, feature):
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
        context = "manual_nd2nz" if feature in transfer.FEATURES else controller + ":" + layout
    elif pipe in ("vmte2", "vmte3"):
        context = vector
    else:
        context = "shared"
    return "/".join((pipe, context, feature))


def group(row):
    return row["inputs"]["S1"], row["inputs"]["S2"]


def fold(row):
    return int(row["sequence_group"][:8], 16) % 4


def load_graph(row):
    description = row["input_graph"]
    graph = {key: value for key, value in description.items() if key != "arrays"}
    with Path(row["graph_file"]).open("rb") as stream:
        for name, entry in description["arrays"].items():
            stream.seek(entry["offset"])
            raw = zlib.decompress(stream.read(entry["compressed_bytes"]))
            if len(raw) != entry["raw_bytes"]:
                raise ValueError("incomplete graph array")
            graph[name] = np.frombuffer(raw, dtype=entry["dtype"]).reshape(entry["shape"])
    return graph


def save_graph(graph, stream):
    description = {key: value for key, value in graph.items()
                   if key not in ("a", "b", "which", "templates")}
    description["arrays"] = {}
    for name in ("a", "b", "which", "templates"):
        array = np.ascontiguousarray(graph[name])
        raw = array.tobytes()
        packed = zlib.compress(raw, 1)
        description["arrays"][name] = {
            "offset": stream.tell(), "dtype": str(array.dtype), "shape": list(array.shape),
            "compressed_bytes": len(packed), "raw_bytes": len(raw),
        }
        stream.write(packed)
    return description


def deadline(signum, frame):
    raise TimeoutError("CPU source graph construction exceeded 60 seconds")


def supplemental_samples():
    reserved = set()
    for name in ("20261104_004934.txt", "20261104_102023.txt"):
        path = ROOT.parent / name
        if path.is_file():
            for row in csv.DictReader(path.open()):
                if row.get("status") == "ok":
                    reserved.add((int(row["S1"]), int(row["S2"])))
    path = ROOT.parent / "20261103_211222.txt"
    grouped = collections.defaultdict(list)
    if path.is_file():
        for row in csv.DictReader(path.open()):
            if row.get("status") != "ok" or row.get("actual_implementation") != "varlen_l1":
                continue
            if float(row.get("checked_elements") or 0) <= 0 or float(row.get("failed_elements") or 0) > 0:
                continue
            if row.get("schedule_disagreements") != "{}":
                continue
            inputs = micro.source.cost.normalize({key: row[key] for key in micro.source.cost.INPUT_FIELDS})
            if (inputs["S1"], inputs["S2"]) in reserved:
                continue
            grouped[micro.source.cost.sha256(inputs)].append((inputs, row))
    result = []
    for identity, repeats in grouped.items():
        inputs, row = repeats[0]
        result.append({"case": row["case"], "input_sha256": identity, "inputs": inputs,
                       "sequence_group": micro.source.cost.sha256({key: inputs[key] for key in ("S1", "S2")}),
                       "duration_us": summary_statistics.median(float(r["duration_us"]) for _, r in repeats),
                       "implementation": "varlen_l1",
                       "targets": {pipe: 0.0 for pipe in PIPES}, "measurement_scope": "kernel_latency_only"})
    return result


def dataset():
    metadata = json.loads((BASE / "micro_graphs.json").read_text())
    configuration = json.loads((DEST / "fa_joint_costs.json").read_text())
    contract = {"api_input_normalization": configuration["api_input_normalization"]}
    rows = metadata["samples"]
    identities = {row["input_sha256"] for row in rows}
    rows.extend(row for row in supplemental_samples() if row["input_sha256"] not in identities)
    for row in rows:
        row["implementation"] = row.get("implementation") or row["cost_class"].split("|")[0]
        row["names"] = NAMES
        row["graph_file"] = str(BASE / "micro_graphs.bin")
    CACHE.mkdir(parents=True, exist_ok=True)
    index = CACHE / "l1_graphs.json"
    identity = hashlib.sha256(b"".join(
        path.read_bytes() for path in sorted((ROOT / "clean/runtime").rglob("*.py"))
    ) + json.dumps(contract, sort_keys=True).encode()
      + json.dumps(sorted(row["input_sha256"] for row in rows)).encode()).hexdigest()
    cached = json.loads(index.read_text()) if index.exists() else {}
    failures = []
    if cached.get("source_identity") != identity:
        entries = {}
        signal.signal(signal.SIGALRM, deadline)
        with (CACHE / "l1_graphs.bin").open("wb") as stream:
            for number, row in enumerate((r for r in rows if r["implementation"] == "varlen_l1"), 1):
                graph = work = None
                try:
                    signal.alarm(60)
                    effective = transfer.api_kernel_inputs(row["inputs"], contract["api_input_normalization"])
                    work = transfer.workload(effective, configuration["hardware"], source_vector_reads=True)
                    graph = transfer.branch_graph(work)
                    active = transfer.micro.active_features(work)
                    active["mte2"] = transfer.features(work)
                    entries[row["input_sha256"]] = {
                        "input_graph": save_graph(graph, stream), "input_features": active,
                        "cost_class": transfer.micro.cost_class(work.p, work.s),
                    }
                    print("L1 SOURCE GRAPH", number, "ok", flush=True)
                except (ValueError, MemoryError, TimeoutError) as error:
                    failures.append({"input_sha256": row["input_sha256"], "error": str(error)})
                    print("L1 SOURCE GRAPH", number, "unavailable", str(error), flush=True)
                finally:
                    signal.alarm(0)
                    del graph, work
                    gc.collect()
        cached = {"source_identity": identity, "entries": entries, "failures": failures}
        index.write_text(json.dumps(cached))
    accepted = []
    for row in rows:
        if row["implementation"] == "varlen_l1":
            entry = cached["entries"].get(row["input_sha256"])
            if entry is None:
                continue
            row.update(entry)
            row["graph_file"] = str(CACHE / "l1_graphs.bin")
            row["names"] = transfer.PARAMETERS
        accepted.append(row)
    return accepted, cached["failures"], configuration["hardware"]


def matrices(rows):
    keys = sorted({binding(row["cost_class"], *name)
                   for row in rows for name in row["names"]})
    positions = {key: index for index, key in enumerate(keys)}
    features = np.zeros((len(rows), len(PIPES), len(keys)))
    targets = np.zeros((len(rows), len(PIPES)))
    for i, row in enumerate(rows):
        row["mapping"] = np.array([
            positions[binding(row["cost_class"], *name)]
            for name in row["names"]
        ])
        for j, pipe in enumerate(PIPES):
            targets[i, j] = row["targets"][pipe]
            for name, value in row["input_features"][pipe].items():
                key = binding(row["cost_class"], pipe, name)
                features[i, j, positions[key]] += value
    return keys, features, targets


def initial(train, features, targets, times):
    observed = targets[train] > 0
    data = features[train] / np.maximum(targets[train, :, None], 1e-9) * observed[:, :, None]
    data[:, PIPES.index("cscalar")] = 0
    data[:, PIPES.index("vscalar")] = 0
    a = data.reshape(-1, data.shape[-1])
    active = np.max(a, axis=0) > 0
    scale = np.max(a, axis=0)
    scalar = features[train][:, [PIPES.index("cscalar"), PIPES.index("vscalar")]]
    scalar_targets = targets[train][:, [PIPES.index("cscalar"), PIPES.index("vscalar")]]
    scalar_observed = scalar_targets > 0
    scale = np.maximum(scale, np.max(scalar / np.maximum(scalar_targets[:, :, None], 1e-9)
                                     * scalar_observed[:, :, None], axis=(0, 1)))
    fallback = np.max(features[train] / times[train, None, None], axis=(0, 1))
    scale = np.where(scale > 0, scale, fallback)
    supported = scale > 0
    scale = np.maximum(scale, 1e-30)
    rates = np.zeros(a.shape[1])
    normalized = a[:, active] / scale[active]
    if np.any(active):
        rates[active] = nnls(np.vstack((normalized, np.eye(sum(active)) * .01)),
                             np.r_[np.ones(len(a)), np.zeros(sum(active))], maxiter=10000)[0] / scale[active]
    return rates, scale, supported


def linear_fit(train, gradients, times, features, targets, current, scale, supported, prior):
    active = np.flatnonzero(supported)
    kernel = gradients[train][:, active] / times[train, None] / scale[active]
    observed = targets[train] > 0
    pmu = features[train][:, :, active] / np.maximum(targets[train, :, None], 1e-9) / scale[active] * observed[:, :, None]
    upper = np.array([pipe in ("cscalar", "vscalar") for pipe in PIPES])
    start = current[active] * scale[active]
    anchor = prior[active] * scale[active]
    def residual(x, jac=False):
        r1 = kernel @ x - 1
        r2 = pmu @ x - 1
        mask = ((r2 > 0) | ~upper[None, :]) & observed
        if jac:
            return np.vstack((kernel, (.2 * pmu * mask[:, :, None]).reshape(-1, len(active)),
                              .01 * np.eye(len(active)), .05 * np.eye(len(active))))
        return np.r_[r1, (.2 * r2 * mask).ravel(), .01 * (x - anchor), .05 * (x - start)]
    optimum = least_squares(residual, np.maximum(start, 1e-12), jac=lambda x: residual(x, True),
                            bounds=(0, np.inf), loss="soft_l1", f_scale=.1,
                            max_nfev=60, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    result = np.zeros(len(scale))
    result[active] = optimum.x / scale[active]
    return result


def statistics(values):
    if not len(values):
        return {"n": 0}
    a = np.asarray(values)
    return {"n": len(a), "median_ape_percent": float(np.median(a)),
            "p90_ape_percent": float(np.quantile(a, .9)), "max_ape_percent": float(np.max(a)),
            **{"above%d_count" % cut: int(sum(a > cut)) for cut in (10, 20, 30)}}


def write_report(report):
    fields = ("record_type", "scope", "implementation", "fold", "status", "actual_us", "predicted_us",
              "ape_percent", "repeats") + micro.source.cost.INPUT_FIELDS + ("quality_flags", "unsupported", "error", "value_json")
    with (DEST / "fa_joint_calibration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for scope, value in report.items():
            if scope in ("cases", "external_validation"):
                continue
            writer.writerow({"record_type": "metadata", "scope": scope, "value_json": json.dumps(value, separators=(",", ":"))})
        external = report["external_validation"]
        for scope, value in external.items():
            if scope != "cases":
                writer.writerow({"record_type": "external_summary", "scope": scope,
                                 "value_json": json.dumps(value, separators=(",", ":"))})
        for name, rows in (("grouped_holdout", report["cases"]), ("external_holdout", external["cases"])):
            for row in rows:
                item = {"record_type": name, **row, **row["inputs"]}
                for field in ("quality_flags", "unsupported"):
                    if field in item:
                        item[field] = json.dumps(item[field], separators=(",", ":"))
                writer.writerow(item)


def predict(inputs, model):
    normalized = micro.source.cost.normalize(inputs)
    schedule = micro.source.schedule(normalized, model["hardware"])
    if normalized["layout"] == "TND" and not schedule["same_ab"]:
        schedule["l1"] = schedule["s1_base"] <= 128 and max(normalized["D"], normalized["DV"]) <= 256
    implementation = micro.implementation(normalized, schedule)
    if implementation == "varlen_l1":
        normalized = transfer.api_kernel_inputs(normalized, model["api_input_normalization"])
        work = transfer.workload(normalized, model["hardware"], source_vector_reads=True)
        graph = transfer.branch_graph(work)
        names = transfer.PARAMETERS
    else:
        work = micro.workload(normalized, model["hardware"])
        graph = micro.branch_graph(work)
        names = NAMES
    key = micro.cost_class(work.p, work.s)
    keys = [binding(key, *name) for name in names]
    calibrated = set(model["coefficients"]) | set(model["zero_costs"])
    unsupported = [key for j, key in enumerate(keys)
                   if key not in calibrated and np.any(graph["templates"][:, j] > 0)]
    if unsupported:
        raise ValueError("uncalibrated source operations: " + ",".join(unsupported))
    rates = [model["coefficients"].get(key, 0.0) for key in keys]
    value, _, phases = transfer.micro.source.evaluate(graph, rates)
    return {"predicted_us": float(value), "implementation": implementation,
            "critical_path_phase_us": phases}


def external_validation(model, training):
    identities = {row["input_sha256"] for row in training}
    groups = {group(row) for row in training}
    grouped, failed_status = collections.defaultdict(list), collections.Counter()
    for name in ("20261104_004934.txt", "20261104_102023.txt"):
        path = ROOT.parent / name
        if not path.is_file():
            continue
        for row in csv.DictReader(path.open()):
            if row["status"] != "ok":
                failed_status[row["status"]] += 1
                continue
            inputs = micro.source.cost.normalize({key: row[key] for key in micro.source.cost.INPUT_FIELDS})
            identity = micro.source.cost.sha256(inputs)
            grouped[identity].append((inputs, row))
    detail, eligible_errors, raw_errors = [], [], []
    signal.signal(signal.SIGALRM, deadline)
    frozen = json.dumps(model, sort_keys=True)
    for number, (identity, repeats) in enumerate(sorted(grouped.items()), 1):
        inputs = repeats[0][0]
        flags = set()
        if identity in identities or (inputs["S1"], inputs["S2"]) in groups:
            flags.add("overlaps_calibration_group")
        for _, row in repeats:
            duration = float(row["duration_us"])
            if row["pmu_complete"] != "True":
                flags.add("incomplete_pmu")
            if float(row.get("checked_elements") or 0) <= 0 or float(row.get("failed_elements") or 0) > 0:
                flags.add("numerical_check")
            if row.get("schedule_disagreements") != "{}" or row.get("actual_implementation") != "varlen_l1":
                flags.add("route_or_schedule")
            if duration <= 0 or not math.isfinite(duration):
                flags.add("invalid_duration")
            for key, value in row.items():
                if key in ("aic_time_max_us", "aiv_time_max_us") or key.endswith("_active_max_us"):
                    if float(value or 0) > duration * 1.02:
                        flags.add("per_core_time_exceeds_task")
        actual = summary_statistics.median(float(row["duration_us"]) for _, row in repeats)
        result = {"inputs": inputs, "repeats": len(repeats), "actual_us": actual,
                  "quality_flags": sorted(flags), "status": "unavailable"}
        try:
            signal.alarm(60)
            prediction = predict(inputs, model)
            error = abs(prediction["predicted_us"] / actual - 1) * 100
            result.update(predicted_us=prediction["predicted_us"], ape_percent=error, status="ok")
            raw_errors.append(error)
            if not flags:
                eligible_errors.append(error)
        except (ValueError, TimeoutError, MemoryError) as error:
            result["error"] = str(error)
        finally:
            signal.alarm(0)
            gc.collect()
        detail.append(result)
        if number % 20 == 0:
            print("EXTERNAL VALIDATION", number, "/", len(grouped), flush=True)
    if json.dumps(model, sort_keys=True) != frozen:
        raise ValueError("external validation mutated the frozen coefficients")
    return {"scope": "previously collected inputs, excluded entirely from joint fitting and initialization; no coefficient changes",
            "qualified": statistics(eligible_errors), "all_predicted_including_quality_flags": statistics(raw_errors),
            "unavailable_count": sum(row["status"] != "ok" for row in detail),
            "quality_flags": dict(collections.Counter(flag for row in detail for flag in row["quality_flags"])),
            "failed_measurement_status_counts": dict(failed_status), "cases": detail}


def run():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1100 * 1024 ** 2, 1100 * 1024 ** 2))
    rows, failures, hardware = dataset()
    keys, features, targets = matrices(rows)
    times = np.array([row["duration_us"] for row in rows])
    folds = np.array([fold(row) for row in rows])
    masks = [folds != number for number in range(4)] + [np.ones(len(rows), dtype=bool)]
    states = []
    for train in masks:
        rates, scale, supported = initial(train, features, targets, times)
        states.append({"rates": rates, "prior": rates.copy(), "scale": scale, "supported": supported})
    predictions = np.zeros((5, len(rows)))
    gradients = np.zeros((5, len(rows), len(keys)))
    history = []
    best = [None] * len(states)
    print("JOINT DATA", len(rows), "sequence groups", len({group(row) for row in rows}),
          "potential shared costs", len(keys), "graph failures", len(failures), flush=True)
    for iteration in range(ROUNDS + 1):
        for i, row in enumerate(rows):
            graph = load_graph(row)
            for number, state in enumerate(states):
                rates = state["rates"][row["mapping"]]
                value, derivative = micro.source.graph_time(graph, rates, True)
                predictions[number, i] = value
                gradients[number, i] = np.bincount(row["mapping"], derivative, minlength=len(keys))
                if not math.isclose(float(gradients[number, i] @ state["rates"]), value,
                                    rel_tol=1e-8, abs_tol=1e-7):
                    raise ValueError("critical path does not conserve time")
            del graph
        train_errors = [float(np.median(np.abs(predictions[n, mask] / times[mask] - 1)) * 100)
                        for n, mask in enumerate(masks)]
        objectives = []
        for n, (state, train) in enumerate(zip(states, masks)):
            pmu = np.einsum("ijk,k->ij", features[train], state["rates"]) / np.maximum(targets[train], 1e-9) - 1
            valid = targets[train] > 0
            upper = np.array([pipe in ("cscalar", "vscalar") for pipe in PIPES])
            pmu *= valid & ((pmu > 0) | ~upper[None, :])
            residual = np.r_[predictions[n, train] / times[train] - 1, .2 * pmu.ravel(),
                             .01 * state["scale"] * (state["rates"] - state["prior"])]
            objective = float(np.sum(np.sqrt(1 + (residual / .1) ** 2) - 1))
            objectives.append(objective)
            if best[n] is None or objective < best[n]["objective"]:
                best[n] = {"objective": objective, "iteration": iteration, "rates": state["rates"].copy(),
                           "predictions": predictions[n].copy(), "gradients": gradients[n].copy()}
        history.append({"iteration": iteration, "training_median_ape_percent": train_errors,
                        "training_objectives": objectives})
        print("JOINT ITERATION", iteration, "training medians", train_errors, flush=True)
        if iteration == ROUNDS:
            break
        for n, (state, train) in enumerate(zip(states, masks)):
            state["rates"] = linear_fit(train, gradients[n], times, features, targets,
                                         state["rates"], state["scale"], state["supported"], state["prior"])
    for n, state in enumerate(states):
        state["rates"] = best[n]["rates"]
        predictions[n] = best[n]["predictions"]
        gradients[n] = best[n]["gradients"]
    detail, errors, by_family = [], [], collections.defaultdict(list)
    for i, row in enumerate(rows):
        n = folds[i]
        unsupported = np.flatnonzero(~states[n]["supported"] & np.any(features[i] > 0, axis=0))
        predicted = float(predictions[n, i])
        error = abs(predicted / times[i] - 1) * 100
        status = "unsupported_operation" if len(unsupported) else "ok"
        if status == "ok":
            errors.append(error)
            by_family[row["implementation"]].append(error)
        detail.append({"inputs": row["inputs"], "fold": int(n), "implementation": row["implementation"],
                       "actual_us": float(times[i]), "predicted_us": predicted,
                       "ape_percent": error, "status": status,
                       "unsupported": [keys[j] for j in unsupported]})
    final = states[-1]
    normalized = (features / np.maximum(targets[:, :, None], 1e-9) * (targets > 0)[:, :, None]).reshape(-1, len(keys)) / final["scale"]
    sensitivity = np.vstack((gradients[-1] / times[:, None] / final["scale"], normalized))
    active = final["supported"]
    report = {
        "evaluation": "four deterministic folds withholding every input sharing complete (S1,S2)",
        "source_scope": "input-derived microtile dependency graphs; source-corrected manual ND2NZ graph for Varlen L1",
        "fit_scope": "kernel latency plus nine PMU totals; scalar PMU only upper bounds; no measured isolated stages",
        "old_coefficients_used": False, "heldout_targets_used_for_fit_or_initialization": False,
        "calibration_samples": len(rows), "sequence_groups": len({group(row) for row in rows}),
        "samples_with_complete_pmu": int(sum(np.all(targets > 0, axis=1))),
        "samples_with_latency_only": int(sum(np.all(targets == 0, axis=1))),
        "potential_costs": len(keys), "supported_costs": int(sum(active)),
        "nonzero_costs": int(sum(final["rates"] > 1e-10)),
        "coefficient_scope": "shared instruction and software-operation contexts; no shape IDs or per-shape multipliers",
        "coefficient_interpretation": "nonnegative effective service costs; a fitted zero does not establish zero physical instruction latency",
        "validation_scope": "retrospective grouped validation; these inputs were previously available to source-model development",
        "normalized_joint_sensitivity_rank": int(np.linalg.matrix_rank(sensitivity[:, active])),
        "cross_validation": statistics(errors),
        "by_implementation": {key: statistics(value) for key, value in sorted(by_family.items())},
        "unsupported_validation_samples": sum(row["status"] != "ok" for row in detail),
        "selected_iterations": [entry["iteration"] for entry in best],
        "iteration_selection": "minimum training-only robust joint objective; no validation error used",
        "graph_failures": failures, "optimization_history": history, "cases": detail,
        "production_replaced": False,
    }
    legacy = {row["input_sha256"]: row for row in csv.DictReader((DEST / "fa_branch_audit.csv").open())
              if row["record_type"] == "kernel_case" and row["heldout_latency_used_for_fit"] == "False"}
    comparison = []
    for row, measured in zip(rows, detail):
        previous = legacy.get(row["input_sha256"])
        if previous and measured["status"] == "ok":
            if int(previous["fold"]) != measured["fold"]:
                raise ValueError("historical comparison uses a different heldout fold")
            comparison.append((measured["ape_percent"], float(previous["ape_percent"])))
    report["matched_historical_grouped_validation"] = {
        "scope": "same inputs and heldout folds; archived independent-class fit, not a frozen in-sample replay",
        "comparison_limit": "historical fit used an earlier calibration set and baseline graph; not a controlled parameter-count ablation",
        "shared": statistics([pair[0] for pair in comparison]),
        "historical": statistics([pair[1] for pair in comparison]),
    }
    artifact = {"version": "fa_joint_shared_costs_v2", "hardware": hardware,
                "api_input_normalization": json.loads((DEST / "fa_joint_costs.json").read_text())["api_input_normalization"],
                "coefficients": {key: float(rate) for key, rate in zip(keys, final["rates"]) if rate > 1e-10},
                "zero_costs": [key for key, rate, supported in zip(keys, final["rates"], active)
                               if supported and rate <= 1e-10],
                "unsupported": [key for key, supported in zip(keys, active) if not supported]}
    report["external_validation"] = external_validation(artifact, rows)
    DEST.mkdir(exist_ok=True)
    write_report(report)
    (DEST / "fa_joint_costs.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print("JOINT VALIDATION", json.dumps({key: value for key, value in report.items()
                                          if key not in ("cases", "optimization_history", "external_validation")}), flush=True)
    print("EXTERNAL SUMMARY", json.dumps({key: value for key, value in report["external_validation"].items()
                                           if key != "cases"}), flush=True)


if __name__ == "__main__":
    run()
