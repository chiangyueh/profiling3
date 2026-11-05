import csv
import json
from collections import Counter
from pathlib import Path

from calibration.fa_cost_validation import labels_for
from calibration.fa_joint_calibration import NAMES, binding, transfer

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).with_name("measurements.csv")
MODEL = ROOT / "clean/model.json"
OUTPUT = Path(__file__).with_name("parameter_sources.csv")
FORMULAS = {
    "flops_m": "sum(2*align(M)*align(N)*align(K))/1e6",
    "tiles": "source microtile or transfer block count",
    "output_tiles": "source output microtile count",
    "l0_kb": "sum(2*(align(M)+align(N))*align(K))/1024",
    "read_kb": "source read bytes after modeled reuse /1024",
    "loads": "source load event count",
    "address_gap_kb": "sum((rows-1)*(stride-row_width)*element_bytes)/1024; address span, not payload",
    "nd_write_kb": "ND output bytes /1024",
    "nz_write_kb": "NZ output bytes /1024",
    "stores": "source output/store event count",
    "calls": "source Matmul API/output call count",
    "softmax_k": "aligned softmax rows * padded columns /1000",
    "reduction_k": "aligned softmax rows * ceil(log2(padded columns))/1000; workload proxy",
    "cast_k": "cast element count /1000",
    "scale_k": "Vec1 scaled element count /1000",
    "nz2nd_k": "NZ-to-ND element count /1000",
    "nd2nz_k": "ND-to-NZ element count /1000",
    "online_k": "nonfirst KV-window rows * 8/1000; workload proxy",
    "rescale_k": "nonfirst KV-window output elements * 2/1000",
    "normalize_k": "last KV-window output elements /1000",
    "drop_k": "dropout element count /1000",
    "unpack_k": "mask adapter expanded elements /1000",
    "duplicate_k": "mask adapter initialization buffer elements /1000",
    "chunks": "source vector chunk count",
    "alignment_read_kb": "vector format-alignment extra read bytes /1024",
    "write_kb": "source intermediate/output/statistic write bytes /1024",
    "workers": "participating source workers",
    "batch_entries": "TND batch entries visited across participating workers",
    "commands": "source API/MAD/barrier control event count; not disassembled instruction count",
    "drop_chunks": "source dropout chunk count",
    "adapter_workers": "workers participating in mask adaptation",
    "payload_kb": "rows * width * 2/1024",
    "copy_calls": "manual DMA copy call count",
    "strided_rows": "rows when stride > width, otherwise 0",
    "nd2nz_partial_rows": "Q/K/V rows with a nonzero 256-byte row tail",
    "capacity_pressure_kb": "payload_kb * max(0,1-L2_bytes/KV_footprint_bytes); capacity proxy",
}


def workload_source(pipe, feature):
    if pipe == "mte2" and feature in transfer.FEATURES:
        return "clean/runtime/workloads/transfer.py:transfer,add_copies"
    if pipe in ("cube", "mte1"):
        return "clean/runtime/workloads/matmul.py:refine_matmul_job"
    if pipe in ("mte2", "fixpipe"):
        return "clean/runtime/workloads/matmul.py:cube_job,matmul_cache_trace"
    if pipe == "vector" and feature in ("unpack_k", "duplicate_k"):
        return "clean/runtime/workloads/adapter.py:adapter_jobs,jobs"
    if feature in ("workers", "batch_entries", "adapter_workers"):
        return "clean/runtime/workloads/__init__.py:active_features"
    if pipe == "cscalar":
        return "clean/runtime/workloads/matmul.py:refine_matmul_job"
    return "clean/runtime/workloads/vector.py:vector_job;clean/runtime/workloads/adapter.py:jobs"


def unit(feature):
    if feature.endswith("_kb"):
        return "us/KiB"
    if feature == "flops_m":
        return "us/million_FLOPs"
    if feature.endswith("_k"):
        return "us/1000_work_units"
    return "us/" + feature


def export_measurements():
    from calibration import fa_joint_calibration as joint

    rows, failures, hardware = joint.dataset()
    if failures:
        raise ValueError("source graphs are incomplete; accepted measurements were not replaced")
    names = list(NAMES) + [name for name in transfer.PARAMETERS if name not in NAMES]
    fields = ["case", *joint.micro.source.cost.INPUT_FIELDS, "duration_us", "measurement_scope",
              "cost_class", "host_geometry_disagreements"]
    fields += [target for target, _ in joint.micro.PIPES.values()]
    fields += [pipe + "." + feature for pipe, feature in names]
    temporary = DATA.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = {
                "case": row["case"], **row["inputs"], "duration_us": row["duration_us"],
                "measurement_scope": row.get("measurement_scope", "kernel_latency_and_whole_kernel_pmu"),
                "cost_class": row["cost_class"],
                "host_geometry_disagreements": ";".join(row.get("geometry_differences", [])),
            }
            item.update({target: row["targets"][pipe] if row["targets"][pipe] > 0 else ""
                         for pipe, (target, _) in joint.micro.PIPES.items()})
            item.update({pipe + "." + feature: row["input_features"].get(pipe, {}).get(feature, 0)
                         for pipe, feature in names})
            writer.writerow(item)
    temporary.replace(DATA)


def export():
    model = json.loads(MODEL.read_text())
    names = sorted(set(model["coefficients"]) | set(model["zero_costs"]))
    support, measured = Counter(), Counter()
    with DATA.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        parameters = transfer.PARAMETERS if row["cost_class"].split("|")[0] == "varlen_l1" else NAMES
        keys = [binding(row["cost_class"], *name) for name in parameters]
        labels = labels_for(keys, model)
        present = set()
        for (pipe, feature), label in zip(parameters, labels):
            if label is not None and float(row.get(pipe + "." + feature) or 0) > 0:
                present.add(label)
        for label in present:
            support[label] += 1
            pipe = label.split("/")[0]
            field = transfer.micro.PIPES[pipe][0]
            measured[label] += int(float(row.get(field) or 0) > 0)
    fields = ("parameter", "value", "status", "unit", "method", "directly_measured", "measured_counter",
              "measurement_scope", "fit_constraint", "positive_workload_samples", "with_pmu_samples",
              "workload_formula", "workload_source", "calibration_source", "notes")
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in names:
            pipe, context, feature = key.split("/")
            scalar = pipe in ("cscalar", "vscalar")
            zero = key in model["zero_costs"]
            writer.writerow({
                "parameter": key, "value": model["coefficients"].get(key, 0),
                "status": "fitted_zero" if zero else "fitted_nonzero", "unit": unit(feature),
                "method": "joint_nonnegative_fit_kernel_latency_and_pmu", "directly_measured": False,
                "measured_counter": transfer.micro.PIPES[pipe][0],
                "measurement_scope": "whole_kernel_sum_across_cores; no independent phase isolation",
                "fit_constraint": "upper_bound_includes_waiting" if scalar else "relative_residual",
                "positive_workload_samples": support[key], "with_pmu_samples": measured[key],
                "workload_formula": FORMULAS[feature], "workload_source": workload_source(pipe, feature),
                "calibration_source": "calibration/fa_cost_validation.py:optimize,objective",
                "notes": "nonnegative fit boundary does not imply zero hardware cost" if zero else "context=" + context + "; shared across qualifying inputs",
            })
    print("PARAMETER_SOURCES=" + str(OUTPUT))
    print("PARAMETERS=" + str(len(names)))


if __name__ == "__main__":
    export()
