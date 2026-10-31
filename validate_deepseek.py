from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import ast
import csv
import gc
import hashlib
import inspect
import json
import re
import statistics
from pathlib import Path

import fa_latency_model as model_api


ROOT = Path(__file__).resolve().parent


def target_cases():
    tree = ast.parse((ROOT / "main.py").read_text())
    cases = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets))
    result = []
    for case in cases:
        if case["name"].startswith("deepseek_"):
            inputs = model_api.cost.normalize(dict(zip(model_api.cost.INPUT_FIELDS, case["shape"] + (0,))))
            result.append({"case": case["name"], "inputs": inputs,
                           "shape_identity": model_api.cost.sha256(inputs)})
    if not result:
        raise RuntimeError("no DeepSeek cases in main.py")
    return result


def load_training(reference, targets, data_root):
    selected_groups = {
        model_api.cost_group({"inputs": row["inputs"], "schedule": model_api.schedule(row["inputs"], reference["hardware"]),
                              "unpaired_reuse_segments": 0}) for row in targets
    }
    held_out = {row["shape_identity"] for row in targets}
    allowed_shapes = set().union(*(set(reference["classes"][group]["calibration_shapes"])
                                   for group in selected_groups)) - held_out
    paths, inventory = [], []
    for item in reference["development_data"]:
        path = data_root / item["name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"measurement source changed: {path}")
        paths.append(path)
        inventory.append({"name": path.name, "sha256": digest})
    with (data_root / "20261028_175018.txt").open(newline="") as stream:
        schema = next(cells for cells in csv.reader(stream) if "duration_us" in cells and "status" in cells)
    corpus = model_api.measurement_corpus(paths, reference["hardware"],
                                         {"20261029_024431.txt": schema})
    removed = [row for row in corpus["measurements"] if row["shape_identity"] in held_out]
    selected = [row for row in corpus["measurements"] if row["shape_identity"] in allowed_shapes]
    if any(row["shape_identity"] in held_out for row in selected):
        raise RuntimeError("DeepSeek target data entered training")
    return {"measurements": selected, "inventory": inventory}, {
        "excluded_target_shapes": sorted(held_out),
        "excluded_target_measurements": len(removed),
        "excluded_target_configurations": len({row["configuration_identity"] for row in removed}),
        "allowed_original_training_shapes": sorted(allowed_shapes),
        "source_files": inventory,
        "training_rule": "All seven complete DeepSeek input shapes, every tiling and repeat excluded together. Only other shapes from the reference model calibration cohorts are eligible.",
    }


def fit_without_targets(reference, targets, corpus, audit, output):
    samples, rejected = model_api.pipeline_samples(corpus, reference["hardware"], ROOT / "out" / "deepseek_holdout" / "graphs")
    audit["excluded_metadata_configurations"] = rejected
    groups = sorted({row["class_id"] for row in samples})
    if not groups:
        raise RuntimeError("no usable non-target calibration configurations")
    classes = {}
    held_out = set(audit["excluded_target_shapes"])
    for group in groups:
        rows = [row for row in samples if row["class_id"] == group]
        old_entry = reference["classes"].get(group)
        if old_entry is None:
            raise RuntimeError("training selected an uncalibrated implementation")
        if any(row["shape_identity"] in held_out for row in rows):
            raise RuntimeError("target leakage before calibration")
        streaming = bool(old_entry.get("pipeline_k_streaming"))
        fences = bool(old_entry.get("matmul_end_fences"))
        if streaming or fences:
            for row in rows:
                design = model_api.workload(row["inputs"], reference["hardware"],
                                            calibration_geometry=row["geometry"],
                                            calibration_schedule=row["configuration"])
                row["graph"] = model_api.timing_graph({**design, "pipeline_k_streaming": streaming,
                                                       "matmul_end_fences": fences})
        print(f"CALIBRATE NON-DEEPSEEK: shapes={len({row['shape_identity'] for row in rows})} configurations={len(rows)} class={group}", flush=True)
        entry = model_api.fit_pipeline_group(
            rows, reference["hardware"],
            transport_counters_are_upper_bounds=bool(old_entry.get("transport_counters_are_upper_bounds")),
            max_nfev=800)
        entry.update(pipeline_k_streaming=streaming, matmul_end_fences=fences,
                     transport_counters_are_upper_bounds=bool(old_entry.get("transport_counters_are_upper_bounds")))
        classes[group] = entry
    trained = {row["shape_identity"] for row in samples}
    if trained & held_out:
        raise RuntimeError("target leakage in frozen artifact")
    audit.update(training_shapes=len(trained), training_configurations=len(samples),
                 missing_original_training_shapes=sorted(set(audit["allowed_original_training_shapes"]) - trained),
                 target_training_overlap=0)
    artifact = {"version": model_api.VERSION, "source_sha256": model_api.source_hash(),
                "hardware": reference["hardware"], "classes": classes,
                "training_shapes": sorted(trained), "validation_audit": audit,
                "reference_model_sha256": model_api.cost.sha256(reference),
                "calibrator_sha256": hashlib.sha256(inspect.getsource(model_api.fit_pipeline_group).encode()).hexdigest(),
                "calibration_max_evaluations": 800,
                "development_data": audit["source_files"],
                "qualification": "retrospective_complete_target_shape_holdout; architecture already exposed to these targets; not prospective blind validation",
                "target": "FlashAttentionScore kernel Task Duration(us)",
                "target_data_used_for_fit": False, "prospective_pass": False}
    (output / "model.json").write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")
    del samples
    gc.collect()
    return artifact


def evaluate(targets, artifact, measurements, output, alternate_measurements=None):
    rows = []
    for target in targets:
        row = {"case": target["case"], **target["inputs"], "status": "ok",
               "target_used_for_calibration": False,
               "training_shapes": artifact["validation_audit"]["training_shapes"],
               "training_configurations": artifact["validation_audit"]["training_configurations"],
               "model_sha256": model_api.cost.sha256(artifact),
               "prediction_scope": "input_only_official_tiling_kernel_duration"}
        try:
            result = model_api.predict(target["inputs"], artifact)
            row.update(predicted_us=result["predicted_us"], cost_class=result["cost_policy"],
                       outside_calibration_range=";".join(result["out_of_calibration_range"]))
            entry = artifact["classes"][result["cost_policy"]]
            row.update(calibration_converged=entry["optimization_success"],
                       calibration_evaluations=entry["optimization_evaluations"])
        except (ValueError, RuntimeError, OSError) as error:
            row.update(status="unavailable", error=str(error))
        rows.append(row)
    (output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
    actual = {}
    alternate = {}
    if alternate_measurements is not None and alternate_measurements.is_file():
        for line in alternate_measurements.read_text(errors="replace").splitlines():
            match = re.match(r"OFFICIAL BASELINE: route=(\S+) duration_us=([\d.]+)", line)
            if match:
                alternate[match.group(1)] = float(match.group(2))
    name = None
    for line in measurements.read_text(errors="replace").splitlines():
        if line.startswith("MODEL="):
            name = line.split("MODEL=", 1)[1].split()[0]
        elif line.startswith("OFFICIAL_US=") and name:
            actual[name] = float(line.split("=", 1)[1])
    for row in rows:
        row.update(measurement_source=measurements.name,
                   measurement_sha256=hashlib.sha256(measurements.read_bytes()).hexdigest(),
                   timing_protocol="historical msprof Task Duration(us)",
                   evaluation="retrospective_all_seven_target_shapes_held_out")
        if row["case"] not in actual:
            row.update(status="missing_measurement", error="official baseline not present")
            continue
        row["actual_us"] = actual[row["case"]]
        if row["status"] == "ok":
            row["signed_error_percent"] = 100 * (row["predicted_us"] / row["actual_us"] - 1)
            row["ape_percent"] = abs(row["signed_error_percent"])
            row["absolute_error_us"] = abs(row["predicted_us"] - row["actual_us"])
            if row["case"] in alternate:
                row.update(alternate_actual_us=alternate[row["case"]],
                           alternate_measurement_source=alternate_measurements.name,
                           alternate_signed_error_percent=100 * (row["predicted_us"] / alternate[row["case"]] - 1))
        print(f"{row['case']}: actual={row['actual_us']:.2f} predicted={row.get('predicted_us', float('nan')):.2f} error={row.get('signed_error_percent', float('nan')):+.2f}%", flush=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    errors = [row["ape_percent"] for row in rows if row["status"] == "ok"]
    summary = {"targets": len(rows), "predicted": len(errors),
               "mape_percent": statistics.mean(errors) if errors else None,
               "median_ape_percent": statistics.median(errors) if errors else None,
               "max_ape_percent": max(errors) if errors else None,
               "within_10_percent": sum(error <= 10 for error in errors),
               "target_training_overlap": 0,
               "new_npu_measurements": False,
               "evaluation": "Retrospective shape holdout. Source structure and model selection were previously exposed to target data. Historical baseline effective tiling and device state have not been fully reverified."}
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "deepseek_holdout")
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    reference_path = args.reference or (ROOT / "fa_source_model.json" if (ROOT / "fa_source_model.json").is_file() else args.output / "model.json")
    reference = json.loads(reference_path.read_text())
    if reference["source_sha256"] != model_api.source_hash():
        raise RuntimeError("reference predictor source differs")
    targets = target_cases()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus, audit = load_training(reference, targets, args.data_root)
    print(f"EXCLUDED: shapes={len(targets)} measurements={audit['excluded_target_measurements']} configurations={audit['excluded_target_configurations']}", flush=True)
    artifact = fit_without_targets(reference, targets, corpus, audit, args.output)
    evaluate(targets, artifact, args.data_root / "46.txt", args.output, args.data_root / "42.txt")


if __name__ == "__main__":
    main()
