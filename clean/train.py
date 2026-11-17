import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import hashlib
import json
import math
import resource
import zlib
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

if __package__:
    from .runtime import costs, graphs, inputs, schedules, workloads
    from .runtime.features import PARAMETERS, PIPES as PIPE_FEATURES, TRANSFER_PARAMETERS
    from .runtime.native.graph import graph_time
    from .runtime.native.matmul import tiler
else:
    from runtime import costs, graphs, inputs, schedules, workloads
    from runtime.features import PARAMETERS, PIPES as PIPE_FEATURES, TRANSFER_PARAMETERS
    from runtime.native.graph import graph_time
    from runtime.native.matmul import tiler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training" / "measurements.csv"
CONFIG = ROOT / "training" / "config.json"
CACHE = ROOT / "out" / "training"
PIPES = tuple(PIPE_FEATURES)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dataset(path=DATA, configuration=None):
    if configuration is None:
        configuration = json.loads(CONFIG.read_text())
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("training measurements are empty")
    if len({row["case"] for row in rows}) != len(rows):
        raise ValueError("duplicate training case")
    identities = set()
    for row in rows:
        row["inputs"] = inputs.normalize({key: row[key] for key in inputs.INPUT_FIELDS})
        identity = inputs.sha256(row["inputs"])
        if identity in identities:
            raise ValueError("duplicate training input: " + row["case"])
        identities.add(identity)
        row["duration_us"] = float(row["duration_us"])
        if not math.isfinite(row["duration_us"]) or row["duration_us"] <= 0:
            raise ValueError("invalid measured kernel time: " + row["case"])
        work = None
        try:
            work, _ = workloads.build(row["inputs"], configuration["hardware"], configuration["api_input_normalization"])
            row["cost_class"] = schedules.cost_class(work.p, work.s)
            row["names"] = TRANSFER_PARAMETERS if work.manual_transfer else PARAMETERS
            active = workloads.active_features(work)
            row.update({pipe + "." + feature: value for pipe, values in active.items() for feature, value in values.items()})
        except Exception as error:
            raise RuntimeError("workload calculation failed for " + row["case"] + ": " + str(error)) from error
        finally:
            del work
            gc.collect()
    keys = sorted({costs._binding(row["cost_class"], *name) for row in rows for name in row["names"]})
    positions = {key: index for index, key in enumerate(keys)}
    features = np.zeros((len(rows), len(PIPES), len(keys)))
    targets = np.zeros((len(rows), len(PIPES)))
    for i, row in enumerate(rows):
        row["mapping"] = np.array([positions[costs._binding(row["cost_class"], *name)] for name in row["names"]])
        for j, (pipe, (counter, _)) in enumerate(PIPE_FEATURES.items()):
            targets[i, j] = float(row.get(counter) or 0)
            for selected_pipe, feature in row["names"]:
                if selected_pipe == pipe:
                    features[i, j, positions[costs._binding(row["cost_class"], pipe, feature)]] += float(row.get(pipe + "." + feature) or 0)
    if not np.isfinite(features).all() or not np.isfinite(targets).all() or np.any(features < 0) or np.any(targets < 0):
        raise ValueError("invalid training workloads or PMU observations")
    return rows, keys, features, targets


class Graphs:
    def __init__(self, rows, configuration):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / "graphs.bin"
        self.index = CACHE / "graphs.json"
        identity = hashlib.sha256(DATA.read_bytes() + CONFIG.read_bytes())
        for path in sorted((ROOT / "runtime").rglob("*")):
            if path.suffix in (".py", ".cpp"):
                identity.update(str(path.relative_to(ROOT)).encode())
                identity.update(path.read_bytes())
        identity.update(tiler()[1].encode())
        self.signature = identity.hexdigest()
        saved = json.loads(self.index.read_text()) if self.index.is_file() else {}
        transition = CACHE / "graphs.signature"
        if transition.is_file() and transition.read_text().split() == [saved.get("signature"), self.signature]:
            saved["signature"] = self.signature
            atomic_json(self.index, saved)
        if saved.get("signature") != self.signature or not self.path.is_file():
            saved = {"signature": self.signature, "entries": [], "end": 0}
            self.path.write_bytes(b"")
            atomic_json(self.index, saved)
        if self.path.stat().st_size < saved["end"] or len(saved["entries"]) > len(rows):
            raise ValueError("incomplete training graph cache")
        self.entries = saved["entries"]
        with self.path.open("r+b") as stream:
            stream.truncate(saved["end"])
            stream.seek(saved["end"])
            for i in range(len(self.entries), len(rows)):
                row = rows[i]
                work = graph = None
                try:
                    work, implementation = workloads.build(row["inputs"], configuration["hardware"], configuration["api_input_normalization"])
                    if implementation != row["cost_class"].split("|")[0] or schedules.cost_class(work.p, work.s) != row["cost_class"]:
                        raise ValueError("source route differs from historical training record")
                    graph = graphs.build(work)
                    entry = {"root": int(graph["root"]), "arrays": {}}
                    for name in ("a", "b", "which", "templates"):
                        array = np.ascontiguousarray(graph[name])
                        packed = zlib.compress(array.tobytes(), 1)
                        entry["arrays"][name] = {"offset": stream.tell(), "bytes": len(packed), "dtype": str(array.dtype), "shape": list(array.shape)}
                        stream.write(packed)
                    stream.flush()
                    self.entries.append(entry)
                    atomic_json(self.index, {"signature": self.signature, "entries": self.entries, "end": stream.tell()})
                except Exception as error:
                    raise RuntimeError("training graph failed for " + row["case"] + ": " + str(error)) from error
                finally:
                    del work, graph
                    gc.collect()

    def load(self, i):
        entry = self.entries[i]
        graph = {"root": entry["root"]}
        with self.path.open("rb") as stream:
            for name, description in entry["arrays"].items():
                stream.seek(description["offset"])
                raw = zlib.decompress(stream.read(description["bytes"]))
                graph[name] = np.frombuffer(raw, dtype=description["dtype"]).reshape(description["shape"])
        return graph


def source_paths(rows, keys, store):
    path = CACHE / "source_paths.npz"
    if path.is_file():
        with np.load(path) as saved:
            signature = str(saved["signature"])
            transition = CACHE / "graphs.signature"
            if signature == store.signature or transition.is_file() and transition.read_text().split() == [signature, store.signature]:
                result = saved["paths"]
                if result.shape == (len(rows), len(PIPES) + 4, len(keys)) and np.isfinite(result).all():
                    return result
    units = np.array([.05 if key.endswith("flops_m") or key.endswith("_k") else
                      .01 if key.endswith("kb") else 10 if key.endswith("workers") else .1 for key in keys])
    probes = [units * np.array([key.startswith(pipe + "/") for key in keys]) for pipe in PIPES]
    probes.append(units)
    for pipes in (("cube", "mte1"), ("mte2", "fixpipe", "vmte2", "vmte3"), ("vector", "cscalar", "vscalar")):
        probes.append(units * np.array([8 if key.split("/")[0] in pipes else 1 for key in keys]))
    paths = np.zeros((len(rows), len(probes), len(keys)))
    for i, row in enumerate(rows):
        graph = store.load(i)
        for j, probe in enumerate(probes):
            value, derivative = graph_time(graph, probe[row["mapping"]], True)
            paths[i, j] = np.bincount(row["mapping"], derivative, minlength=len(keys))
            if not np.isclose(paths[i, j] @ probe, value, rtol=1e-8, atol=1e-7):
                raise ValueError("source path does not conserve time")
        del graph
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, paths=paths, signature=store.signature)
    temporary.replace(path)
    return paths


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
    upper = np.tile([pipe in ("cscalar", "vscalar") for pipe in PIPES], len(times))
    present = np.any(pmu_matrix != 0, axis=1)
    pmu *= present & ((pmu > 0) | ~upper)
    residual = np.r_[kernel, .2 * pmu, .01 * (parameters - anchor)]
    return float(np.sum(np.sqrt(1 + (residual / .1) ** 2) - 1))


def optimize(bank, matrix, features, targets, times, train, previous=None):
    active_features = features @ matrix
    prior, scale, supported = initial(train, active_features, targets, times)
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
    upper = np.tile([pipe in ("cscalar", "vscalar") for pipe in PIPES], sum(train))
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


def projection(keys, specification):
    labels = costs._aliases(keys, specification)
    names = sorted({key for key in labels if key is not None})
    positions = {key: i for i, key in enumerate(names)}
    matrix = np.zeros((len(keys), len(names)))
    for i, key in enumerate(labels):
        if key is not None:
            matrix[i, positions[key]] = 1
    return names, matrix


def fit(rows, keys, features, targets, times, paths, store, configuration):
    train = np.ones(len(rows), dtype=bool)
    specifications = {"baseline": {"merges": [], "removed": []}, "selected": configuration}
    bank = Bank(paths)
    matrices = {name: projection(keys, spec) for name, spec in specifications.items()}
    pending, states, results = {"baseline"}, {}, {}
    for round_number in range(30):
        for name in specifications:
            if name not in pending:
                continue
            _, matrix = matrices[name]
            previous = states[name]["rates"] if name in states else None
            if previous is None and "baseline" in results:
                weights = np.sum(features, axis=(0, 1))
                previous = matrix.T @ (weights * results["baseline"]["full_rates"])
                previous /= np.maximum(matrix.T @ weights, 1e-30)
            states[name] = optimize(bank, matrix, features, targets, times, train, previous)
        exact = {name: np.zeros(len(rows)) for name in pending}
        violations = set()
        for i, row in enumerate(rows):
            graph = store.load(i)
            for name in sorted(pending):
                full = states[name]["full_rates"]
                value, derivative = graph_time(graph, full[row["mapping"]], True)
                exact[name][i] = value
                envelope = float(np.max(bank.paths[i] @ full))
                discrepancy = value - envelope
                tolerance = 1e-7 + 1e-5 * abs(value)
                if discrepancy < -tolerance:
                    raise ValueError("path envelope exceeds the source graph")
                if discrepancy > tolerance:
                    bank.add(i, np.bincount(row["mapping"], derivative, minlength=len(keys)))
                    violations.add(name)
            del graph
        for name in sorted(pending - violations):
            results[name] = {**states[name], "names": matrices[name][0], "predicted": exact[name], "oracle_rounds": round_number + 1}
        pending -= set(results)
        if "baseline" in results:
            pending |= set(specifications) - set(results)
        if not pending:
            return results["selected"]
    raise RuntimeError("source path oracle did not converge; existing model was not replaced")


def write_result(result, configuration, rows, times, output):
    model = {key: configuration[key] for key in ("version", "hardware", "api_input_normalization", "merges", "removed")}
    model["coefficients"] = {key: float(value) for key, value in zip(result["names"], result["rates"]) if value > 1e-10}
    model["zero_costs"] = [key for key, value, supported in zip(result["names"], result["rates"], result["supported"]) if supported and value <= 1e-10]
    model["unsupported"] = [key for key, supported in zip(result["names"], result["supported"]) if not supported]
    if not np.isfinite(result["rates"]).all() or np.any(result["rates"] < 0) or not np.isfinite(result["predicted"]).all() or np.any(result["predicted"] <= 0):
        raise ValueError("invalid fitted model; existing model was not replaced")
    errors = np.abs(result["predicted"] / times - 1) * 100
    report = {"records": len(rows), "with_pmu": sum(any(float(row.get(counter) or 0) > 0 for counter, _ in PIPE_FEATURES.values()) for row in rows),
              "nonzero_parameters": len(model["coefficients"]), "zero_parameters": len(model["zero_costs"]), "unsupported_parameters": len(model["unsupported"]),
              "training_median_ape_percent": float(np.median(errors)), "training_p90_ape_percent": float(np.quantile(errors, .9)),
              "training_within_10_percent": float(np.mean(errors <= 10) * 100), "objective": float(result["objective"]),
              "oracle_rounds": result["oracle_rounds"], "optimizer_at_iteration_limit": bool(result["at_iteration_limit"]),
              "evaluation_scope": "In-sample fit diagnostics; this is not independent validation.",
              "schema_selection": configuration["schema_selection"], "measurement_scope": configuration["measurement_scope"],
              "existing_model_used_for_initialization": False, "npu_measurements": 0}
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = CACHE / "fit.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "measured_us", "predicted_us", "absolute_error_percent"], lineterminator="\n")
        writer.writeheader()
        for row, measured, predicted, error in zip(rows, times, result["predicted"], errors):
            writer.writerow({"case": row["case"], "measured_us": measured, "predicted_us": predicted, "absolute_error_percent": error})
    temporary.replace(CACHE / "fit.csv")
    atomic_json(CACHE / "fit.json", report)
    atomic_json(output, model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "model.json")
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    limit = 1024 * 1024 ** 2
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = min(limit, hard) if hard != resource.RLIM_INFINITY else limit
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    configuration = json.loads(CONFIG.read_text())
    rows, keys, features, targets = dataset(configuration=configuration)
    store = Graphs(rows, configuration)
    paths = source_paths(rows, keys, store)
    times = np.array([row["duration_us"] for row in rows])
    result = fit(rows, keys, features, targets, times, paths, store, configuration)
    write_result(result, configuration, rows, times, args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
