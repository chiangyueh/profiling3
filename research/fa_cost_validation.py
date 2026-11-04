from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import collections
import ast
import csv
import gc
import hashlib
import inspect
import json
import resource
import shutil
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.stats import binomtest

from research import fa_joint_calibration as joint

CACHE = joint.ROOT / "out/fa_cost_validation"
REPORT = joint.DEST / "fa_cost_validation.csv"
MODEL = joint.DEST / "fa_validated_costs.json"
PIPE_ORDER = tuple(joint.PIPES)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)


class Graphs:
    def __init__(self, rows, signature):
        self.path = CACHE / "graphs.raw"
        index = CACHE / "raw_index.json"
        saved = json.loads(index.read_text()) if index.exists() else {}
        if saved.get("signature") != signature or not self.path.exists():
            self.path.unlink(missing_ok=True)
            index.unlink(missing_ok=True)
            needed = sum(d["raw_bytes"] + 8 for row in rows for d in row["input_graph"]["arrays"].values())
            if shutil.disk_usage(CACHE).free < needed + 256 * 1024 ** 2:
                raise RuntimeError("insufficient room for temporary CPU graph cache")
            descriptions = []
            with self.path.open("wb") as stream:
                for row in rows:
                    graph = joint.load_graph(row)
                    item = {key: value for key, value in graph.items() if key not in ("a", "b", "which", "templates")}
                    item["arrays"] = {}
                    for name in ("a", "b", "which", "templates"):
                        a = np.ascontiguousarray(graph[name])
                        stream.write(b"\0" * ((-stream.tell()) % 8))
                        item["arrays"][name] = {"offset": stream.tell(), "shape": a.shape, "dtype": str(a.dtype)}
                        stream.write(a.tobytes())
                    descriptions.append(item)
            saved = {"signature": signature, "graphs": descriptions}
            atomic_json(index, saved)
        self.descriptions = saved["graphs"]

    def load(self, i):
        item = self.descriptions[i]
        graph = {key: value for key, value in item.items() if key != "arrays"}
        for key, description in item["arrays"].items():
            graph[key] = np.memmap(self.path, mode="r", offset=description["offset"],
                                   shape=tuple(description["shape"]), dtype=description["dtype"])
        return graph

    @staticmethod
    def close(graph):
        for key in ("a", "b", "which", "templates"):
            graph[key]._mmap.close()

    def clean(self):
        self.path.unlink(missing_ok=True)
        (CACHE / "raw_index.json").unlink(missing_ok=True)


def variants(keys, observed):
    groups = []
    for pipe, feature in joint.NAMES:
        if pipe == "cube" or pipe == "vector" and feature == "cast_k":
            groups.append((pipe, feature, "dtype"))
        elif pipe == "mte2":
            groups.extend((pipe, feature, axis) for axis in ("api", "layout"))
        elif pipe == "vector" and feature in ("softmax_k", "reduction_k", "online_k", "chunks"):
            groups.append((pipe, feature, "function"))
        elif pipe in ("vmte2", "vmte3"):
            groups.append((pipe, feature, "function"))
        elif pipe == "cscalar" and feature in ("workers", "commands"):
            groups.append((pipe, feature, "controller"))
        elif pipe == "vscalar" and feature in ("workers", "chunks"):
            groups.append((pipe, feature, "controller" if feature == "workers" else "function"))
    result = [{"name": "baseline", "kind": "baseline", "merges": [], "removed": []}]
    result.extend({"name": "merge:" + "/".join(item), "kind": "merge", "merges": [item], "removed": []} for item in groups)
    result.extend({"name": "pool:" + pipe, "kind": "pipe_pool", "merges": [(pipe, "*", "all")], "removed": []} for pipe in PIPE_ORDER)
    result.append({"name": "pool:all", "kind": "global_pool", "merges": [("*", "*", "all")], "removed": []})
    result.extend({"name": "remove:" + key, "kind": "coefficient_removal", "merges": [], "removed": [key]}
                  for key, active in zip(keys, observed) if active)
    return result


def labels_for(keys, specification):
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


def projection(keys, specification):
    labels = labels_for(keys, specification)
    names = sorted({key for key in labels if key is not None})
    positions = {key: i for i, key in enumerate(names)}
    matrix = np.zeros((len(keys), len(names)))
    for i, key in enumerate(labels):
        if key is not None:
            matrix[i, positions[key]] = 1
    return names, matrix


def source_paths(rows, keys, graphs, signature):
    path = CACHE / "source_paths.npz"
    metadata = CACHE / "source_paths.json"
    if path.exists() and metadata.exists() and json.loads(metadata.read_text()).get("signature") == signature:
        with np.load(path) as saved:
            return saved["paths"]
    units = np.array([.05 if key.endswith("flops_m") or key.endswith("_k") else
                      .01 if key.endswith("kb") else 10 if key.endswith("workers") else .1 for key in keys])
    probes = [units * np.array([key.startswith(pipe + "/") for key in keys]) for pipe in PIPE_ORDER]
    probes.append(units)
    for pipes in (("cube", "mte1"), ("mte2", "fixpipe", "vmte2", "vmte3"), ("vector", "cscalar", "vscalar")):
        probes.append(units * np.array([8 if key.split("/")[0] in pipes else 1 for key in keys]))
    paths = np.zeros((len(rows), len(probes), len(keys)))
    for i, row in enumerate(rows):
        graph = graphs.load(i)
        try:
            for j, probe in enumerate(probes):
                value, derivative = joint.micro.source.graph_time(graph, probe[row["mapping"]], True)
                paths[i, j] = np.bincount(row["mapping"], derivative, minlength=len(keys))
                if not np.isclose(paths[i, j] @ probe, value, rtol=1e-8, atol=1e-7):
                    raise ValueError("source path does not conserve time")
        finally:
            graphs.close(graph)
        if (i + 1) % 200 == 0:
            print("SOURCE PATHS", i + 1, "/", len(rows), flush=True)
    np.savez_compressed(path, paths=paths)
    atomic_json(metadata, {"signature": signature, "scope": "source-only fixed probes; no measured times or fitted costs"})
    return paths


class Bank:
    def __init__(self, paths):
        self.paths = np.zeros((paths.shape[0], 32, paths.shape[2]))
        self.paths[:, :paths.shape[1]] = paths
        self.counts = np.full(paths.shape[0], paths.shape[1], dtype=int)

    def envelope(self, rates):
        values = self.paths.reshape(-1, len(rates)) @ rates
        values = values.reshape(self.paths.shape[:2])
        choices = np.argmax(values, axis=1)
        return values[np.arange(len(choices)), choices], self.paths[np.arange(len(choices)), choices]

    def add(self, i, path):
        n = self.counts[i]
        if n == self.paths.shape[1]:
            self.paths = np.pad(self.paths, ((0, 0), (0, n), (0, 0)))
        self.paths[i, n] = path
        self.counts[i] += 1


def objective(predicted, times, pmu_matrix, parameters, anchor):
    kernel = predicted / times - 1
    pmu = pmu_matrix @ parameters - 1
    upper = np.tile([pipe in ("cscalar", "vscalar") for pipe in PIPE_ORDER], len(times))
    present = np.any(pmu_matrix != 0, axis=1)
    pmu *= present & ((pmu > 0) | ~upper)
    residual = np.r_[kernel, .2 * pmu, .01 * (parameters - anchor)]
    return float(np.sum(np.sqrt(1 + (residual / .1) ** 2) - 1))


def optimize(bank, matrix, features, targets, times, train, previous=None):
    active_features = features @ matrix
    prior, scale, supported = joint.initial(train, active_features, targets, times)
    selected = np.flatnonzero(supported)
    transform = matrix[:, selected] / scale[selected]
    anchor = prior[selected] * scale[selected]
    x = anchor.copy() if previous is None else previous[selected] * scale[selected]
    observed = targets[train] > 0
    pmu = active_features[train][:, :, selected] / np.maximum(targets[train, :, None], 1e-9)
    pmu *= observed[:, :, None]
    pmu = (pmu / scale[selected]).reshape(-1, len(selected))
    if previous is not None:
        previous_score = objective(bank.envelope(transform @ x)[0][train], times[train], pmu, x, anchor)
        prior_score = objective(bank.envelope(transform @ anchor)[0][train], times[train], pmu, anchor, anchor)
        if prior_score < previous_score:
            x = anchor.copy()
    upper = np.tile([pipe in ("cscalar", "vscalar") for pipe in PIPE_ORDER], sum(train))
    present = observed.ravel()
    steps = 0
    for steps in range(80):
        predicted, paths = bank.envelope(transform @ x)
        a = paths[train] @ transform / times[train, None]
        r1 = predicted[train] / times[train] - 1
        r2 = pmu @ x - 1
        w1 = (1 + (r1 / .1) ** 2) ** -.25
        w2 = .2 * (1 + (.2 * r2 / .1) ** 2) ** -.25
        w2 *= present & ((r2 > 0) | ~upper)
        w3 = .01 * (1 + (.01 * (x - anchor) / .1) ** 2) ** -.25
        ak, ap = a * w1[:, None], pmu * w2[:, None]
        gram = ak.T @ ak + ap.T @ ap + np.diag(w3 ** 2 + .05 ** 2)
        rhs = ak.T @ w1 + ap.T @ w2 + w3 ** 2 * anchor + .05 ** 2 * x
        chol = np.linalg.cholesky(gram)
        proposal = nnls(chol.T, solve_triangular(chol, rhs, lower=True), maxiter=10000)[0]
        current = objective(predicted[train], times[train], pmu, x, anchor)
        accepted = False
        for power in range(16):
            candidate = x + (proposal - x) * (0.5 ** power)
            values, _ = bank.envelope(transform @ candidate)
            score = objective(values[train], times[train], pmu, candidate, anchor)
            if score < current - 1e-9:
                improvement = current - score
                x, accepted = candidate, True
                break
        if not accepted or improvement < 1e-7 * max(1, current):
            break
    rates = np.zeros(len(scale))
    rates[selected] = x / scale[selected]
    return {"rates": rates, "full_rates": matrix @ rates, "supported": supported,
            "objective": objective(bank.envelope(matrix @ rates)[0][train], times[train], pmu, x, anchor),
            "steps": steps + 1, "at_iteration_limit": steps == 79}


def fit_scope(name, specifications, train, test, rows, keys, features, targets, times, paths, graphs, signature):
    target = CACHE / (name + ".json")
    scope_signature = hashlib.sha256((signature + json.dumps(specifications, sort_keys=True)
                                     + json.dumps(np.flatnonzero(train).tolist())
                                     + json.dumps(np.flatnonzero(test).tolist())).encode()).hexdigest()
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["signature"] == scope_signature:
            print("VALIDATION CACHE", name, flush=True)
            return saved["results"]
    bank = Bank(paths)
    matrices = {spec["name"]: projection(keys, spec) for spec in specifications}
    requested = {spec["name"] for spec in specifications}
    pending = {"baseline"} if "baseline" in requested else set(requested)
    states, results = {}, {}
    test_indices = np.flatnonzero(test)
    required = np.flatnonzero(train | test)
    for round_number in range(30):
        for number, spec in enumerate(specifications):
            key = spec["name"]
            if key not in pending:
                continue
            names, matrix = matrices[key]
            previous = states[key]["rates"] if key in states else None
            if previous is None and "baseline" in results:
                weights = np.sum(features[train], axis=(0, 1))
                previous = matrix.T @ (weights * np.array(results["baseline"]["full_rates"]))
                previous /= np.maximum(matrix.T @ weights, 1e-30)
            states[key] = optimize(bank, matrix, features, targets, times, train, previous)
            if (number + 1) % 10 == 0:
                print("FIT", name, "oracle", round_number, number + 1, "/", len(specifications), flush=True)
        exact = {key: np.zeros(len(rows)) for key in pending}
        gap = {key: 0.0 for key in pending}
        violations = {key: False for key in pending}
        for i in required:
            graph = graphs.load(i)
            try:
                for key in sorted(pending):
                    full = states[key]["full_rates"]
                    value, derivative = joint.micro.source.graph_time(graph, full[rows[i]["mapping"]], True)
                    exact[key][i] = value
                    if train[i]:
                        envelope = float(np.max(bank.paths[i] @ full))
                        discrepancy = value - envelope
                        tolerance = 1e-7 + 1e-5 * abs(value)
                        gap[key] = max(gap[key], discrepancy)
                        if discrepancy < -tolerance:
                            raise ValueError("path envelope exceeds the source graph")
                        if discrepancy > tolerance:
                            path = np.bincount(rows[i]["mapping"], derivative, minlength=len(keys))
                            bank.add(i, path)
                            violations[key] = True
            finally:
                graphs.close(graph)
        for key in sorted(pending):
            if violations[key]:
                continue
            state = states[key]
            names, matrix = matrices[key]
            supported_full = (matrix @ state["supported"].astype(float)) > 0
            dropped = np.sum(matrix, axis=1) == 0
            missing = (~supported_full & ~dropped)[None, :] & np.any(features[test_indices] > 0, axis=1)
            results[key] = {"predicted": exact[key][test_indices].tolist(), "indices": test_indices.tolist(),
                            "unsupported": np.any(missing, axis=1).tolist(),
                            "full_rates": state["full_rates"].tolist(), "names": names,
                            "rates": state["rates"].tolist(), "supported": state["supported"].tolist(),
                            "parameters": int(sum(state["supported"])),
                            "nonzero": int(sum(state["rates"] > 1e-10)),
                            "training_objective": state["objective"], "oracle_rounds": round_number + 1,
                            "optimizer_at_iteration_limit": state["at_iteration_limit"],
                            "max_training_envelope_gap_us": gap[key]}
        pending -= set(results)
        if "baseline" in results:
            pending |= requested - set(results)
        print("EXACT SOURCE CHECK", name, "round", round_number, "remaining", len(pending), flush=True)
        if not pending:
            saved = {"signature": scope_signature, "results": results}
            atomic_json(target, saved)
            return results
    raise RuntimeError("source path oracle did not converge: " + name + " " + repr(sorted(pending)))


def risk(predictions, rows, times):
    indices = [i for i, value in enumerate(predictions) if np.isfinite(value)]
    errors = np.abs(predictions[indices] / times[indices] - 1) * 100
    grouped = collections.defaultdict(list)
    for i, error in zip(indices, errors):
        grouped[joint.group(rows[i])].append(error)
    group_errors = np.array([np.median(values) for _, values in sorted(grouped.items())])
    result = joint.statistics(errors)
    result["groups"] = len(group_errors)
    result["group_median_ape_percent"] = float(np.median(group_errors)) if len(group_errors) else None
    if len(group_errors):
        rng = np.random.default_rng(20261104)
        resamples = group_errors[rng.integers(len(group_errors), size=(1000, len(group_errors)))]
        result["group_median_se"] = float(np.std(np.median(resamples, axis=1), ddof=1))
        result["group_p90_ape_percent"] = float(np.quantile(group_errors, .9))
        result["group_p90_se"] = float(np.std(np.quantile(resamples, .9, axis=1), ddof=1))
    return result


def array_for(result, n):
    values = np.full(n, np.nan)
    indices = np.array(result["indices"])
    supported = ~np.array(result["unsupported"])
    values[indices[supported]] = np.array(result["predicted"])[supported]
    return values


def choose(predictions, specifications, rows, times, parameters):
    scores = {name: risk(value, rows, times) for name, value in predictions.items()}
    coverage = max(score["n"] for score in scores.values())
    eligible = [name for name, score in scores.items() if score["n"] == coverage]
    best = min(eligible, key=lambda name: scores[name]["group_median_ape_percent"])
    baseline = scores["baseline"]
    median_limit = scores[best]["group_median_ape_percent"] + scores[best]["group_median_se"]
    tail_limit = baseline["group_p90_ape_percent"] + baseline["group_p90_se"]
    eligible = [name for name in eligible if scores[name]["group_median_ape_percent"] <= median_limit
                and scores[name]["group_p90_ape_percent"] <= tail_limit]
    if not eligible:
        eligible = ["baseline"]
    selected = min(eligible, key=lambda name: (parameters[name], scores[name]["group_median_ape_percent"], name))
    merged = []
    for spec in specifications:
        if spec["kind"] == "merge" and spec["name"] in eligible:
            merged.extend(spec["merges"])
    return selected, {"median_limit": median_limit, "p90_limit": tail_limit, "eligible": eligible, "scores": scores}, merged


def paired_audit(base, candidate, rows, times):
    common = np.isfinite(base) & np.isfinite(candidate)
    differences = collections.defaultdict(list)
    for i in np.flatnonzero(common):
        difference = (abs(candidate[i] / times[i] - 1) - abs(base[i] / times[i] - 1)) * 100
        differences[joint.group(rows[i])].append(difference)
    values = np.array([np.median(v) for _, v in sorted(differences.items())])
    rng = np.random.default_rng(20261104)
    draws = values[rng.integers(len(values), size=(2000, len(values)))]
    interval = np.quantile(np.median(draws, axis=1), (.025, .975))
    positive, negative = int(sum(values > 1e-7)), int(sum(values < -1e-7))
    p = float(binomtest(positive, positive + negative, .5, alternative="greater").pvalue) if positive + negative else 1.0
    return {"paired_groups": len(values), "paired_group_median_delta_pp": float(np.median(values)),
            "paired_ci_low_pp": float(interval[0]), "paired_ci_high_pp": float(interval[1]),
            "positive_groups": positive, "negative_groups": negative, "split_benefit_p": p,
            "max_prediction_change_us": float(np.max(np.abs(candidate[common] - base[common])))}


def compact_model(specification, fitted, hardware):
    original = json.loads((joint.DEST / "fa_joint_costs.json").read_text())
    return {"version": "fa_validated_shared_costs_v1", "hardware": hardware,
            "api_input_normalization": original["api_input_normalization"],
            "merges": specification["merges"], "removed": specification["removed"],
            "coefficients": {key: value for key, value in zip(fitted["names"], fitted["rates"]) if value > 1e-10},
            "zero_costs": [key for key, value, supported in zip(fitted["names"], fitted["rates"], fitted["supported"])
                           if supported and value <= 1e-10],
            "unsupported": [key for key, supported in zip(fitted["names"], fitted["supported"]) if not supported]}


def expanded_model(model, keys):
    aliases = labels_for(keys, model)
    known = set(model["coefficients"]) | set(model["zero_costs"])
    result = {key: value for key, value in model.items() if key not in ("coefficients", "zero_costs", "unsupported")}
    result["coefficients"] = {key: model["coefficients"][alias] for key, alias in zip(keys, aliases) if alias in model["coefficients"]}
    result["zero_costs"] = [key for key, alias in zip(keys, aliases) if alias is None or alias in model["zero_costs"]]
    result["unsupported"] = [key for key, alias in zip(keys, aliases) if alias is not None and alias not in known]
    return result


def predict(inputs, model):
    normalized = joint.micro.source.cost.normalize(inputs)
    schedule = joint.micro.source.schedule(normalized, model["hardware"])
    if normalized["layout"] == "TND" and not schedule["same_ab"]:
        schedule["l1"] = schedule["s1_base"] <= 128 and max(normalized["D"], normalized["DV"]) <= 256
    implementation = joint.micro.implementation(normalized, schedule)
    if implementation == "varlen_l1":
        effective = joint.transfer.api_kernel_inputs(normalized, model["api_input_normalization"])
        work = joint.transfer.Work(effective, model["hardware"])
        names = joint.transfer.PARAMETERS
    else:
        work = joint.micro.Work(normalized, model["hardware"])
        names = joint.NAMES
    cost_class = joint.micro.cost_class(work.p, work.s)
    keys = [joint.binding(cost_class, *name) for name in names]
    return joint.predict(inputs, expanded_model(model, keys))


def numerical_digest(rows, hardware):
    functions = (Graphs, variants, labels_for, projection, source_paths, Bank, objective, optimize,
                 fit_scope, risk, array_for, choose, paired_audit, compact_model, expanded_model,
                 joint.initial, joint.binding, joint.matrices, joint.fold, joint.group)
    code = "\n".join(inspect.getsource(function) for function in functions)
    flow = ast.parse(inspect.getsource(run))
    flow.body[0].body = [node for node in flow.body[0].body if not (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "signature"
                                             for target in node.targets))]
    data = json.dumps({"hardware": hardware, "samples": [(row["input_sha256"], row["duration_us"], row["targets"])
                                                         for row in rows]}, sort_keys=True)
    return hashlib.sha256((code + ast.dump(flow, include_attributes=False) + data).encode()
                          + (joint.BASE / "micro_graphs.json").read_bytes()
                          + (joint.CACHE / "l1_graphs.json").read_bytes()).hexdigest()


def numerical_signature(rows, hardware):
    digest = numerical_digest(rows, hardware)
    path = CACHE / "numerical_cache_key.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get("numerical_signature") != digest:
        saved = {"numerical_signature": digest, "scope_cache_key": digest}
        atomic_json(path, saved)
    return saved["scope_cache_key"]


def run():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1100 * 1024 ** 2, 1100 * 1024 ** 2))
    CACHE.mkdir(exist_ok=True)
    rows, failures, hardware = joint.dataset()
    if failures:
        raise RuntimeError("source graph inventory is incomplete")
    keys, features, targets = joint.matrices(rows)
    times = np.array([row["duration_us"] for row in rows])
    observed = np.any(features > 0, axis=(0, 1))
    specifications = variants(keys, observed)
    selectable = [spec for spec in specifications if spec["kind"] != "coefficient_removal"]
    signature = numerical_signature(rows, hardware)
    graph_signature = hashlib.sha256((joint.BASE / "micro_graphs.json").read_bytes()
                                     + (joint.CACHE / "l1_graphs.json").read_bytes()
                                     + json.dumps(keys).encode()).hexdigest()
    path_signature = hashlib.sha256((graph_signature + inspect.getsource(source_paths)).encode()).hexdigest()
    graphs = Graphs(rows, graph_signature)
    try:
        paths = source_paths(rows, keys, graphs, path_signature)
        outer = np.array([joint.fold(row) for row in rows])
        inner = np.array([(int(row["sequence_group"][:8], 16) // 4) % 3 for row in rows])
        predictions = {spec["name"]: np.full(len(rows), np.nan) for spec in specifications}
        selected_predictions = np.full(len(rows), np.nan)
        outer_models, selection_records, coefficient_folds = [], [], []
        parameter_counts = collections.defaultdict(list)
        for fold in range(4):
            train, test = outer != fold, outer == fold
            assert {joint.group(rows[i]) for i in np.flatnonzero(train)}.isdisjoint(
                {joint.group(rows[i]) for i in np.flatnonzero(test)})
            inner_predictions = {spec["name"]: np.full(len(rows), np.nan) for spec in selectable}
            inner_counts = collections.defaultdict(list)
            for inside in range(3):
                fitting, validation = train & (inner != inside), train & (inner == inside)
                fitted = fit_scope("inner_%d_%d" % (fold, inside), selectable, fitting, validation,
                                   rows, keys, features, targets, times, paths, graphs, signature)
                for name, result in fitted.items():
                    value = array_for(result, len(rows))
                    mask = np.isfinite(value)
                    inner_predictions[name][mask] = value[mask]
                    inner_counts[name].append(result["parameters"])
            counts = {name: max(value) for name, value in inner_counts.items()}
            chosen, selection, merges = choose(inner_predictions, selectable, rows, times, counts)
            combined = {"name": "combined", "kind": "combined", "merges": merges, "removed": []}
            for inside in range(3):
                fitting, validation = train & (inner != inside), train & (inner == inside)
                result = fit_scope("combined_%d_%d" % (fold, inside), [combined], fitting, validation,
                                   rows, keys, features, targets, times, paths, graphs, signature)["combined"]
                if "combined" not in inner_predictions:
                    inner_predictions["combined"] = np.full(len(rows), np.nan)
                value = array_for(result, len(rows));mask = np.isfinite(value)
                inner_predictions["combined"][mask] = value[mask]
                inner_counts["combined"].append(result["parameters"])
            counts["combined"] = max(inner_counts["combined"])
            chosen, selection, _ = choose(inner_predictions, selectable + [combined], rows, times, counts)
            selected_spec = next(spec for spec in selectable + [combined] if spec["name"] == chosen)
            outer_specs = specifications + ([combined] if chosen == "combined" else [])
            fitted = fit_scope("outer_%d" % fold, outer_specs, train, test,
                               rows, keys, features, targets, times, paths, graphs, signature)
            for spec in specifications:
                name = spec["name"];value = array_for(fitted[name], len(rows));mask = np.isfinite(value)
                predictions[name][mask] = value[mask]
                parameter_counts[name].append(fitted[name]["parameters"])
            selected_value = array_for(fitted[chosen], len(rows));mask = np.isfinite(selected_value)
            selected_predictions[mask] = selected_value[mask]
            coefficient_folds.append(fitted["baseline"]["full_rates"])
            outer_models.append(selected_spec)
            selection_records.append({"fold": fold, "selected": selected_spec, "parameters": fitted[chosen]["parameters"],
                                      "training_groups": len({joint.group(row) for row, included in zip(rows, train) if included}),
                                      "inner_selection": selection})
            print("OUTER FOLD COMPLETE", fold, "selected", chosen, "parameters", fitted[chosen]["parameters"], flush=True)
        baseline = risk(predictions["baseline"], rows, times)
        audits = []
        for spec in specifications:
            name = spec["name"]
            audits.append({"name": name, "kind": spec["kind"], "parameters_max": max(parameter_counts[name]),
                           **risk(predictions[name], rows, times),
                           **paired_audit(predictions["baseline"], predictions[name], rows, times)})
        tested = sorted((row for row in audits if row["kind"] != "baseline"), key=lambda row: row["split_benefit_p"])
        adjusted = 0.0
        for rank, row in enumerate(tested):
            adjusted = max(adjusted, min(1.0, row["split_benefit_p"] * (len(tested) - rank)))
            row["split_benefit_holm_p"] = adjusted
            if row["max_prediction_change_us"] < 1e-7:
                row["conclusion"] = "no_prediction_effect_in_grouped_validation"
            elif adjusted <= .05 and row["paired_ci_low_pp"] > 0:
                row["conclusion"] = "retention_supported_on_existing_groups"
            elif row["paired_ci_high_pp"] <= 0 and row["p90_ape_percent"] <= baseline["p90_ape_percent"] and row["n"] >= baseline["n"]:
                row["conclusion"] = "reduction_supported_on_existing_groups"
            else:
                row["conclusion"] = "inconclusive"
        counts = {name: max(value) for name, value in parameter_counts.items()}
        selection_predictions = {spec["name"]: predictions[spec["name"]] for spec in selectable}
        _, _, merges = choose(selection_predictions, selectable, rows, times, counts)
        combined = {"name": "combined", "kind": "combined", "merges": merges, "removed": []}
        selection_predictions["combined"] = np.full(len(rows), np.nan)
        combined_counts = []
        for fold in range(4):
            result = fit_scope("combined_final_%d" % fold, [combined], outer != fold, outer == fold,
                               rows, keys, features, targets, times, paths, graphs, signature)["combined"]
            values = array_for(result, len(rows));included = np.isfinite(values)
            selection_predictions["combined"][included] = values[included]
            combined_counts.append(result["parameters"])
        counts["combined"] = max(combined_counts)
        chosen, final_selection, _ = choose(selection_predictions, selectable + [combined], rows, times, counts)
        final_spec = next(spec for spec in selectable + [combined] if spec["name"] == chosen)
        final_results = fit_scope("final", [selectable[0]] + ([final_spec] if chosen != "baseline" else []),
                                  np.ones(len(rows), dtype=bool), np.ones(len(rows), dtype=bool),
                                  rows, keys, features, targets, times, paths, graphs, signature)
        final = final_results[chosen]
        model = compact_model(final_spec, final, hardware)
        external = joint.external_validation(expanded_model(model, keys), rows)
        model_path = MODEL.with_suffix(".tmp")
        model_path.write_text(json.dumps(model, indent=2) + "\n");model_path.replace(MODEL)
        nested = risk(selected_predictions, rows, times)
        metadata = {"samples": len(rows), "sequence_groups": len({joint.group(row) for row in rows}),
                    "observed_parameters": int(sum(observed)), "single_coefficient_removals": int(sum(spec["kind"] == "coefficient_removal" for spec in specifications)),
                    "group_merges": int(sum(spec["kind"] == "merge" for spec in specifications)),
                    "all_fixed_schemas": len(specifications), "baseline": baseline, "nested_selected": nested,
                    "final_schema": final_spec, "final_parameters": final["parameters"], "final_nonzero": final["nonzero"],
                    "selection_rule": "training-only nested grouped CV; simplest within one bootstrap SE of best group median and baseline group P90",
                    "inference": "paired complete sequence-group bootstrap, 2000 replicates; one-sided sign test with Holm correction",
                    "limits": "retrospective fixed hypothesis audit, not proof of a global minimum or independent physical instruction costs; external data previously exposed",
                    "production_changed": False, "npu_runs": 0}
        rates = np.asarray(final_results["baseline"]["full_rates"])
        gradients = []
        for i, row in enumerate(rows):
            graph = graphs.load(i)
            try:
                _, derivative = joint.micro.source.graph_time(graph, rates[row["mapping"]], True)
                gradients.append(np.bincount(row["mapping"], derivative, minlength=len(keys)))
            finally:
                graphs.close(graph)
        service = np.einsum("ijk,k->ij", features, rates)
        informative = targets > 0
        for j, pipe in enumerate(PIPE_ORDER):
            if pipe in ("cscalar", "vscalar"):
                informative[:, j] &= service[:, j] > targets[:, j]
        pmu_sensitivity = features / np.maximum(targets[:, :, None], 1e-9) * informative[:, :, None]
        sensitivity = np.vstack((np.asarray(gradients) / times[:, None], pmu_sensitivity.reshape(-1, len(keys))))[:, observed]
        scale = np.maximum(np.linalg.norm(sensitivity, axis=0), 1e-30)
        singular = np.linalg.svd(sensitivity / scale, compute_uv=False)
        metadata["informative_data_sensitivity_rank"] = int(np.linalg.matrix_rank(sensitivity / scale))
        metadata["informative_sensitivity_condition"] = float(singular[0] / singular[-1]) if singular[-1] > 0 else "infinite"
        records = [{"record_type": "metadata", "name": name, "value_json": json.dumps(value, separators=(",", ":"))} for name, value in metadata.items()]
        records.extend({"record_type": "ablation", **row} for row in audits)
        records.extend({"record_type": "nested_selection", "fold": row["fold"], "value_json": json.dumps(row, separators=(",", ":"))} for row in selection_records)
        coefficient_folds = np.asarray(coefficient_folds)
        for i, key in enumerate(keys):
            selected_rows = np.any(features[:, :, i] > 0, axis=1)
            records.append({"record_type": "parameter_coverage", "name": key,
                            "samples": int(sum(selected_rows)), "groups": len({joint.group(rows[j]) for j in np.flatnonzero(selected_rows)}),
                            "value_json": json.dumps({"outer_fit_values": coefficient_folds[:, i].tolist(),
                                                      "zero_outer_fits": int(sum(coefficient_folds[:, i] <= 1e-10))}, separators=(",", ":"))})
        for i, row in enumerate(rows):
            records.append({"record_type": "nested_case", **row["inputs"], "fold": int(outer[i]), "implementation": row["implementation"],
                            "actual_us": times[i], "predicted_us": selected_predictions[i] if np.isfinite(selected_predictions[i]) else "",
                            "baseline_us": predictions["baseline"][i] if np.isfinite(predictions["baseline"][i]) else ""})
        records.extend({"record_type": "external_summary", "name": key, "value_json": json.dumps(value, separators=(",", ":"))}
                       for key, value in external.items() if key != "cases")
        records.extend({"record_type": "external_case", **row, **row["inputs"], "value_json": json.dumps(row["quality_flags"], separators=(",", ":"))}
                       for row in external["cases"])
        fields = ["record_type", "name", "kind", "conclusion", "fold", "implementation"] + list(joint.micro.source.cost.INPUT_FIELDS)
        fields += sorted({key for row in records for key in row if key not in fields and key not in ("inputs", "quality_flags")})
        with REPORT.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader();writer.writerows(records)
        print("VALIDATION COMPLETE", json.dumps(metadata), flush=True)
    finally:
        graphs.clean()


if __name__ == "__main__":
    run()
    from research.fa_cost_evidence import annotate
    annotate()
