import csv
import gc
import hashlib
import json
import os
import resource
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out/fa_branch_host"
CACHE = ROOT / "out/fa_joint_calibration"
DATA = Path(__file__).with_name("measurements.csv")
CONFIG = Path(__file__).parent / "reports/fa_joint_costs.json"


def timeout(signum, frame):
    raise TimeoutError("CPU source graph construction exceeded 60 seconds")


def source_identity(identities, contract):
    paths = sorted((ROOT / "prediction").rglob("*.py")) + sorted((ROOT / "clean/runtime").rglob("*.py"))
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths) + Path(__file__).read_bytes()
                          + json.dumps(contract, sort_keys=True).encode()
                          + json.dumps(sorted(identities)).encode()).hexdigest()


def build_sample(row, configuration):
    from prediction import fa_latency_refinement as micro
    from clean.runtime import graphs, schedules, workloads

    inputs = micro.source.cost.normalize({name: row[name] for name in micro.source.cost.INPUT_FIELDS})
    implementation = row["cost_class"].split("|")[0]
    work, selected = workloads.build(inputs, configuration["hardware"], configuration["api_input_normalization"])
    graph = graphs.build(work)
    if selected != implementation or schedules.cost_class(work.p, work.s) != row["cost_class"]:
        raise ValueError("source route differs from saved calibration case: " + row["case"])
    features = workloads.active_features(work)
    for pipe, values in features.items():
        for name, value in values.items():
            saved = float(row.get(pipe + "." + name) or 0)
            if abs(value - saved) > 1e-9 * max(1, abs(saved)):
                raise ValueError("source workload differs from saved calibration case: " + row["case"])
    sample = {
        "case": row["case"], "inputs": inputs,
        "input_sha256": micro.source.cost.sha256(inputs),
        "sequence_group": micro.source.cost.sha256({key: inputs[key] for key in ("S1", "S2")}),
        "cost_class": row["cost_class"], "implementation": implementation,
        "duration_us": float(row["duration_us"]),
        "targets": {pipe: float(row.get(target) or 0) for pipe, (target, _) in micro.PIPES.items()},
        "measurement_scope": row["measurement_scope"],
        "geometry_differences": row["host_geometry_disagreements"].split(";") if row["host_geometry_disagreements"] else [],
        "input_features": features,
    }
    return sample, graph


def ensure_data():
    if (BASE / "micro_graphs.json").is_file() and (BASE / "micro_graphs.bin").is_file():
        return
    from calibration.fa_joint_calibration import save_graph

    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1100 * 1024 ** 2, 1100 * 1024 ** 2))
    configuration = json.loads(CONFIG.read_text())
    with DATA.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("calibration measurements.csv is empty")
    BASE.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    normal = BASE / "micro_graphs.bin"
    manual = CACHE / "l1_graphs.bin"
    temporary = [path.with_suffix(".building") for path in (normal, manual)]
    samples, entries = [], {}
    previous = signal.signal(signal.SIGALRM, timeout)
    try:
        with temporary[0].open("wb") as normal_stream, temporary[1].open("wb") as manual_stream:
            for number, row in enumerate(rows, 1):
                try:
                    signal.alarm(60)
                    sample, graph = build_sample(row, configuration)
                    is_manual = sample["implementation"] == "varlen_l1"
                    description = save_graph(graph, manual_stream if is_manual else normal_stream)
                    sample["input_graph"] = description
                    samples.append(sample)
                    if is_manual:
                        entries[sample["input_sha256"]] = {
                            "input_graph": description, "input_features": sample["input_features"],
                            "cost_class": sample["cost_class"],
                        }
                    del graph
                finally:
                    signal.alarm(0)
                    gc.collect()
                if number % 100 == 0:
                    print("CALIBRATION CPU GRAPHS", number, "/", len(rows), flush=True)
        contract = {"api_input_normalization": configuration["api_input_normalization"]}
        identity = source_identity([row["input_sha256"] for row in samples], contract)
        temporary[0].replace(normal)
        temporary[1].replace(manual)
        (BASE / "micro_graphs.json").write_text(json.dumps({"samples": samples, "rejected": []}))
        (CACHE / "l1_graphs.json").write_text(json.dumps({"source_identity": identity, "entries": entries, "failures": []}))
    finally:
        signal.signal(signal.SIGALRM, previous)
        for path in temporary:
            path.unlink(missing_ok=True)
