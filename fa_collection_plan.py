from __future__ import annotations

import contextlib
import fcntl
import io
import itertools
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fa_latency_inputs as cost
import fa_latency_model as source


ROOT = Path(__file__).resolve().parent
GROUPS = ("same_ab", "general", "s1", "b", "varlen", "drop_adapter")
TARGETS = {group: 20 if group == "same_ab" else 100 for group in GROUPS}
REQUIRED_PMU = tuple(source.PIPES[pipe][0] for pipe in source.PIPES) + (
    "gm_to_l1_kb", "l0c_to_gm_kb", "gm_to_ub_kb", "ub_to_gm_kb",
)


def storage_bytes(p):
    return 2 * p["B"] * (p["N1"] * p["S1"] * p["D"] + p["N2"] * p["S2"] * (p["D"] + p["DV"])) + 4 * p["B"] * p["N1"] * p["S1"] * p["DV"]


def make_case(inputs, group, hardware, seed=""):
    p = cost.normalize(inputs)
    schedule = source.schedule(p, hardware)
    family = {94: "varlen", 95: "same_ab", 96: "general", 97: "s1", 98: "b"}[schedule["priority"]]
    if family == "b" and p["has_drop_mask"]:
        family = "b_drop"
    return {"name": f"{group}_{cost.sha256(p)[:16]}", "inputs": p, "family": family,
            "expected": schedule["priority"], "split": "external", "official_schedule_only": True,
            "timeout_seconds": 300, "build_timeout_seconds": 1800,
            "derived_schedule": {"collection_group": group, "seed": seed,
                                 "estimated_tensor_bytes": storage_bytes(p)}}


def design(collector, specification):
    hardware = specification["frozen_model"]["hardware"]
    plan = {group: [] for group in GROUPS}
    seen = set(specification["previous_input_hashes"])
    for item in specification["same_ab_cases"]:
        case = make_case(item["inputs"], "same_ab", hardware)
        identity = cost.sha256(case["inputs"])
        if identity in seen or case["expected"] != 95:
            raise RuntimeError("a SameAB validation input is repeated or selects another family")
        case["derived_schedule"].update(shape_distance_log2=item["distance_log2_linf"],
                                       prediction_status="pending")
        seen.add(identity)
        plan["same_ab"].append(case)
    if len(plan["same_ab"]) != 20:
        raise RuntimeError("exactly 20 new SameAB validation shapes are required")
    catalog = collector.external_design(hardware)
    for shape in ((2, 4, 1, 160, 320, 256, 256, "FP16", "TND", 0),
                  (4, 4, 1, 2048, 2048, 96, 96, "FP16", "TND", 0),
                  (5, 2, 1, 2048, 512, 128, 128, "FP16", "TND", 0),
                  (3, 2, 1, 4096, 256, 128, 128, "FP16", "TND", 0),
                  (1, 96, 8, 4096, 4096, 128, 128, "BF16", "BNSD", 0),
                  (1, 96, 8, 8192, 8192, 128, 128, "BF16", "BNSD", 0)):
        p = cost.normalize(dict(zip(cost.INPUT_FIELDS, shape)))
        catalog.append({"inputs": p, "coverage": "additional_source_path"})
    seeds = {group: [] for group in GROUPS[1:]}
    for item in catalog:
        p = item["inputs"]
        s = source.schedule(p, hardware)
        if s["drop_adapter"]:
            group = "drop_adapter"
        else:
            group = {94: "varlen", 96: "general", 97: "s1", 98: "b"}.get(s["priority"])
        if group:
            seeds[group].append(item)
    variants = list(itertools.product((1, 2, 3, 4, 6, 8), (1, 2, 3, 4),
                                     (0, -16, 16, 32, 64, 128, 256), (0, -16, 16, 32, 64, 128), (0, 1)))
    variants.sort(key=lambda values: cost.sha256(list(values)))
    for factors in variants:
        for group in GROUPS[1:]:
            if len(plan[group]) >= 400:
                continue
            for seed in seeds[group]:
                p = dict(seed["inputs"])
                p.update(B=p["B"] * factors[0], N1=p["N1"] * factors[1],
                         S1=max(1, p["S1"] + factors[2]), S2=max(1, p["S2"] + factors[3]))
                if factors[4]:
                    p["dtype"] = "BF16" if p["dtype"] == "FP16" else "FP16"
                if p["N1"] % p["N2"] or storage_bytes(p) > 1024 ** 3:
                    continue
                identity = cost.sha256(p)
                if identity in seen:
                    continue
                try:
                    s = source.schedule(p, hardware)
                    actual_group = "drop_adapter" if s["drop_adapter"] else {94: "varlen", 96: "general", 97: "s1", 98: "b"}.get(s["priority"])
                    if actual_group != group:
                        continue
                    case = make_case(p, group, hardware, seed.get("coverage", ""))
                except (ValueError, KeyError):
                    continue
                seen.add(identity)
                plan[group].append(case)
                if len(plan[group]) >= 400:
                    break
        if all(len(plan[group]) == 400 for group in GROUPS[1:]):
            break
    if any(len(plan[group]) < 100 for group in GROUPS[1:]):
        raise RuntimeError("not enough distinct inputs for every requested branch")
    return plan


def freeze(plan, model, hardware, destination):
    path = destination / "predictions_before_measurement.json"
    identity = cost.sha256(model)
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["model_sha256"] != identity:
            raise RuntimeError("saved forecasts belong to another model")
        for case in plan["same_ab"]:
            if saved["predictions"][case["name"]]["input_sha256"] != cost.sha256(case["inputs"]):
                raise RuntimeError("saved forecast inputs differ")
        return saved
    saved = {"model_sha256": identity, "model": model, "frozen_utc": datetime.now(timezone.utc).isoformat(), "predictions": {}}
    keys = ("soc", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes", "ub_bytes")
    compatible = all(str(model["hardware"][key]) == str(hardware[key]) for key in keys)
    for case in plan["same_ab"]:
        item = {"input_sha256": cost.sha256(case["inputs"])}
        try:
            if not compatible:
                raise RuntimeError("physical NPU hardware differs from the frozen predictor")
            item.update(status="predicted", result=source.predict(case["inputs"], model))
        except Exception as error:
            item.update(status="unavailable", error=str(error))
        saved["predictions"][case["name"]] = item
        print(f"FREEZE {case['name']}: {item['status']}", flush=True)
    cost.save_json(path, saved)
    return saved


def extract_pmu(collector, directory, case, row):
    if not row.get("tiling_key"):
        return False
    complete = True
    for number in range(1, collector.ROUNDS + 1):
        folder = directory / "cases" / case["name"] / f"round_{number}"
        try:
            result = collector.profile_counters(folder, int(row["tiling_key"]))
            metrics = result["samples"]
            good = result["launches"] == collector.REPEAT and all(
                len(metrics.get(field, [])) == collector.REPEAT
                and all(collector.finite(value) is not None and value >= 0 for value in metrics[field])
                for field in REQUIRED_PMU)
            complete &= good
            if good:
                cache = {**result, "notes": sorted(result["notes"])}
                cost.save_json(folder / "pmu_cache.json", {"tiling_key": int(row["tiling_key"]), "result": cache,
                                                         "profile_sha256": cost.sha256((folder / "profile.log").read_text(errors="replace"))})
                for table in folder.glob("*.csv"):
                    table.unlink()
        except Exception as error:
            complete = False
            row["derived_schedule"]["pmu_error"] = str(error)
    return complete


def report(collector, directory):
    with contextlib.redirect_stdout(io.StringIO()):
        collector.text_report(directory, collector.ALL_FAMILIES)


def execute(collector, plan, hardware, forecasts, directory):
    csv_path = directory / "routes.csv"
    records = collector.recorded(csv_path)
    successes = Counter()
    cases = {case["name"]: case for pool in plan.values() for case in pool}
    for row in records.values():
        if row.get("status") == "ok" and "pmu_complete" not in row.get("derived_schedule", {}) and row["case"] in cases:
            row["derived_schedule"]["pmu_complete"] = extract_pmu(collector, directory, cases[row["case"]], row)
            collector.append(csv_path, row)
        if row.get("status") == "ok" and row.get("derived_schedule", {}).get("pmu_complete"):
            successes[row["derived_schedule"]["collection_group"]] += 1
    queues = {group: iter(plan[group]) for group in GROUPS}
    exhausted, ensured, build_failures = set(), set(), {}
    attempted = Counter()
    try:
        while len(exhausted) < len(GROUPS):
            for group in GROUPS:
                if group in exhausted:
                    continue
                if successes[group] >= TARGETS[group]:
                    exhausted.add(group)
                    continue
                case = next((case for case in queues[group] if case["name"] not in records), None)
                if case is None:
                    exhausted.add(group)
                    continue
                attempted[group] += 1
                item = forecasts["predictions"].get(case["name"], {})
                case_model = forecasts["model"]
                if (ROOT / "fa_branch_model.json").is_file():
                    import fa_branch_latency as branches
                    try:
                        item, case_model = branches.freeze_case(case, hardware, directory)
                    except Exception as error:
                        item, case_model = {"status": "unavailable", "error": str(error)}, None
                prediction = item.get("result") if item.get("status") == "predicted" else None
                case["derived_schedule"].update(prediction_status=item.get("status", "unmodeled_branch"),
                                                prediction_error=item.get("error", ""))
                if prediction is not None:
                    if prediction.get("implementation"):
                        case["derived_schedule"]["branch_prediction_json"] = json.dumps({name: prediction.get(name) for name in
                            ("implementation", "cost_class", "unidentified_pipe_terms", "execution_events", "selected_cost_model", "refinement",
                             "phase_pipe_active_us", "phase_pipe_active_scope", "max_resource_bytes", "adapter_us", "startup_us",
                             "calibration_sdk_api_sha256", "prediction_sdk_api_sha256", "sdk_api_matches_calibration")}, sort_keys=True)
                    case["derived_schedule"]["prediction_extrapolated_inputs"] = prediction.get("out_of_calibration_range", [])
                    case["derived_schedule"].update({f"predicted_{phase}_critical_us": value
                                                    for phase, value in prediction.get("critical_path_phase_us", {}).items()})
                collector.profile_counters.cache_clear()
                try:
                    print(f"INPUT {case['name']}: " + str(tuple(case["inputs"][field] for field in cost.INPUT_FIELDS)), flush=True)
                    collector.collect(case, hardware, directory, records, csv_path, ensured,
                                      frozen=case_model if prediction is not None else None,
                                      precomputed=prediction, build_failures=build_failures)
                    row = records[case["name"]]
                    row["derived_schedule"].update(case["derived_schedule"])
                    if row.get("priority") is not None and row.get("tiling_key"):
                        branch = collector.describe_branch(int(row["priority"]), int(row["tiling_key"]), case["inputs"])
                        row["derived_schedule"]["actual_branch_json"] = json.dumps(branch, separators=(",", ":"), sort_keys=True)
                    complete = extract_pmu(collector, directory, case, row)
                    row["derived_schedule"]["pmu_complete"] = complete
                    if row.get("status") == "ok" and not complete:
                        row.update(status="partial", stage="pmu", error="required PMU counters or profiler launches missing; latency retained")
                    if row.get("status") == "ok":
                        successes[group] += 1
                    collector.append(csv_path, row)
                except Exception as error:
                    row = records.get(case["name"], {"version": collector.VERSION, "case": case["name"], "split": "external",
                           "family": case["family"], "inputs": case["inputs"], "derived_schedule": case["derived_schedule"],
                           "status": "failed", "stage": "collection", "error": str(error)})
                    row.update(status="failed", stage="collection", error=str(error))
                    collector.append(csv_path, row)
                    records[case["name"]] = row
                    print(f"CASE FAILED: {case['name']}: {error}; continuing", flush=True)
                finally:
                    (ROOT / "output/output.bin").unlink(missing_ok=True)
                    try:
                        report(collector, directory)
                    except Exception as error:
                        print(f"CSV export error: {error}; raw row retained in {csv_path}", flush=True)
                print("PROGRESS " + " ".join(f"{name}={successes[name]}/{TARGETS[name]}" for name in GROUPS), flush=True)
    finally:
        report(collector, directory)
    print(f"CSV_REPORT={directory / 'diagnostics.csv'}", flush=True)
    for group in GROUPS:
        print(f"FINISHED {group}: valid={successes[group]}/{TARGETS[group]} new_attempts={attempted[group]} shortfall={max(0, TARGETS[group] - successes[group])}", flush=True)
    validation = [row for row in records.values() if row.get("status") == "ok"
                  and row.get("derived_schedule", {}).get("collection_group") == "same_ab"
                  and row.get("predicted_us") and row.get("duration_us")]
    errors = [100 * abs(float(row["predicted_us"]) / float(row["duration_us"]) - 1) for row in validation]
    if errors:
        bands = " ".join(f">{limit}%={sum(value > limit for value in errors)}/{len(errors)}" for limit in (10, 20, 30))
        print(f"SAMEAB VALIDATION: measured_predictions={len(errors)}/20 MdAPE={statistics.median(errors):.4f}% "
              f"MAPE={statistics.mean(errors):.4f}% max_error={max(errors):.4f}%; {bands}; frozen model was not recalibrated", flush=True)


def run(collector, npu_id, report_only=False):
    specification = json.loads((ROOT / "fa_collection_plan.json").read_text())
    directory = ROOT / "output/fa_branch_collection"
    directory.mkdir(parents=True, exist_ok=True)
    if report_only:
        report(collector, directory)
        print(f"CSV_REPORT={directory / 'diagnostics.csv'}")
        return
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.environ["FA_BUILD_JOBS"] = "1"
    collector.fa.NPU_ID = npu_id
    collector.fa.PROFILE_WARMUP = collector.WARMUP
    collector.fa.PROFILE_REPEAT = collector.REPEAT
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = design(collector, specification)
    manifest = {"plan_sha256": cost.sha256(plan), "model_sha256": cost.sha256(specification["frozen_model"]),
                "physical_npu_id": npu_id, "targets": TARGETS, "rounds": collector.ROUNDS,
                "repeat": collector.REPEAT, "warmup": collector.WARMUP}
    path = directory / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("this collection folder belongs to another plan, NPU, or frozen model; existing data were retained")
    cost.save_json(path, manifest)
    print(f"NPU: physical {npu_id}; one CPU; serial case execution; round-robin branches; no total runtime limit", flush=True)
    print("TARGETS " + " ".join(f"{group}={count}" for group, count in TARGETS.items()), flush=True)
    env = collector.fa.run_env([], None)
    env.update(FA_DISCOVER_ONLY="1", FA_BUILD_JOBS="1")
    print(f"HOST CACHE CHECK log={directory / 'host_build.log'}", flush=True)
    with (directory / "host_build.log").open("w") as log:
        process = collector.run_case_process(env, log, 1800, command=["bash", "build.sh"])
    if process.returncode:
        raise RuntimeError(f"Host launcher build failed; see {directory / 'host_build.log'}")
    collector.HOST_READY = True
    hardware = collector.hardware_profile(directory, npu_id)
    forecasts = freeze(plan, specification["frozen_model"], hardware, directory)
    execute(collector, plan, hardware, forecasts, directory)
