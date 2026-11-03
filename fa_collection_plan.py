from __future__ import annotations

import contextlib
import csv
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
    focused = specification.get("implementation") == "varlen_l1"
    directory = ROOT / "output" / (specification["version"] if focused else "fa_branch_collection")
    directory.mkdir(parents=True, exist_ok=True)
    if report_only:
        if focused:
            validation_report(collector, directory, final=True)
        else:
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
    plan = validation_design(specification) if focused else design(collector, specification)
    forecasts = validation_model(specification, directory) if focused else None
    model = forecasts["model"] if focused else specification["frozen_model"]
    targets = {"varlen_l1": len(plan)} if focused else TARGETS
    manifest = {"plan_sha256": cost.sha256(plan), "model_sha256": cost.sha256(model),
                "physical_npu_id": npu_id, "targets": targets, "rounds": collector.ROUNDS,
                "repeat": collector.REPEAT, "warmup": collector.WARMUP}
    path = directory / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("this collection folder belongs to another plan, NPU, or frozen model; existing data were retained")
    cost.save_json(path, manifest)
    print(f"NPU: physical {npu_id}; one CPU; serial case execution; no total runtime limit", flush=True)
    print("TARGETS " + " ".join(f"{group}={count}" for group, count in targets.items()), flush=True)
    env = collector.fa.run_env([], None)
    env.update(FA_DISCOVER_ONLY="1", FA_BUILD_JOBS="1")
    print(f"HOST CACHE CHECK log={directory / 'host_build.log'}", flush=True)
    with (directory / "host_build.log").open("w") as log:
        process = collector.run_case_process(env, log, 1800, command=["bash", "build.sh"])
    if process.returncode:
        raise RuntimeError(f"Host launcher build failed; see {directory / 'host_build.log'}")
    collector.HOST_READY = True
    hardware = collector.hardware_profile(directory, npu_id)
    if focused:
        if any(str(hardware.get(key)) != str(value) for key, value in model["hardware"].items()):
            raise RuntimeError("physical NPU hardware differs from the frozen predictor")
        execute_validation(collector, plan, hardware, forecasts, directory)
        return
    forecasts = freeze(plan, specification["frozen_model"], hardware, directory)
    execute(collector, plan, hardware, forecasts, directory)


def validation_design(specification):
    hardware = specification["hardware"]
    excluded = {tuple(pair) for pair in specification["previous_sequence_pairs"]}
    seen, pairs, plan = set(), set(), []
    for item in specification["cases"]:
        case = make_case(item["inputs"], "varlen", hardware, item["stratum"])
        p = case["inputs"]
        identity, pair = cost.sha256(p), (p["S1"], p["S2"])
        s = source.schedule(p, hardware)
        if identity in seen or identity in specification["previous_input_hashes"] or pair in excluded:
            raise RuntimeError("validation input or sequence pair is repeated from previous data")
        if p["layout"] != "TND" or case["expected"] != 94 or s["same_ab"] or s["s1_base"] > 128 or max(p["D"], p["DV"]) > 256:
            raise RuntimeError("validation input does not select Varlen L1 from the official source conditions")
        if storage_bytes(p) > specification["max_tensor_bytes"]:
            raise RuntimeError("validation input exceeds the planned tensor memory bound")
        case["derived_schedule"].update(stratum=item["stratum"], sequence_pair_unseen=True,
                                        expected_implementation="varlen_l1")
        seen.add(identity)
        pairs.add(pair)
        plan.append(case)
    if len(plan) != specification["shape_count"] or len(pairs) != specification["sequence_pair_count"]:
        raise RuntimeError("validation shape counts differ from the committed plan")
    return plan


def validation_model(specification, directory):
    import fa_branch_latency as branches
    import fa_latency_refinement as micro
    path = directory / "predictions_before_measurement.json"
    if path.is_file():
        saved = json.loads(path.read_text())
    else:
        model = json.loads((ROOT / "fa_branch_model.json").read_text())
        saved = {"model": model, "model_sha256": cost.sha256(model), "predictions": {},
                 "frozen_utc": datetime.now(timezone.utc).isoformat()}
    model = saved["model"]
    if saved["model_sha256"] != specification["model_sha256"] or cost.sha256(model) != saved["model_sha256"]:
        raise RuntimeError("validation plan and frozen model differ; existing measurements were retained")
    if model["source_sha256"] != branches.signature() or model["refined"]["source_sha256"] != micro.signature():
        raise RuntimeError("frozen predictor and execution source differ")
    if model["hardware"] != specification["hardware"]:
        raise RuntimeError("validation hardware differs from the frozen model")
    training = set(model["baseline"]["training_inputs"]) | set(model["refined"]["training_inputs"])
    if not training.issubset(specification["previous_input_hashes"]):
        raise RuntimeError("validation exclusions omit model training inputs")
    cost.save_json(path, saved)
    return saved


def validation_forecast(case, forecasts, directory):
    import fa_branch_latency as branches
    identity = cost.sha256(case["inputs"])
    saved = forecasts["predictions"].get(case["name"])
    if saved is not None:
        if saved["input_sha256"] != identity or saved["model_sha256"] != forecasts["model_sha256"]:
            raise RuntimeError("saved forecast belongs to another input or model")
        return saved
    item = {"input_sha256": identity, "model_sha256": forecasts["model_sha256"],
            "frozen_utc": datetime.now(timezone.utc).isoformat()}
    try:
        prediction = branches.predict(case["inputs"], forecasts["model"])
        if prediction["implementation"] != "varlen_l1" or prediction["selected_cost_model"] != "micro_costs":
            raise RuntimeError("frozen model selected another implementation or cost model")
        item.update(status="predicted", result=prediction)
    except Exception as error:
        item.update(status="unavailable", error=str(error))
    try:
        item["baseline_predicted_us"] = branches.predict(case["inputs"], forecasts["model"]["baseline"])["predicted_us"]
    except Exception as error:
        item["baseline_prediction_error"] = str(error)
    forecasts["predictions"][case["name"]] = item
    cost.save_json(directory / "predictions_before_measurement.json", forecasts)
    return item


def validation_route(collector, case, folder, hardware):
    collector.select_shape(case)
    route = collector.discover(collector.fa.shape_params(), folder, timeout=case["timeout_seconds"])
    branch = collector.describe_branch(*route, case["inputs"])
    effective = collector.fa.read_effective(folder / "host_effective.tsv")
    case["derived_schedule"].update(observed_priority=route[0], observed_tiling_key=route[1],
                                    actual_branch_json=json.dumps(branch, sort_keys=True))
    collector.verify_case_schedule(case, hardware, route[0], effective or {})
    expected = {"key_ub0": "S1", "key_ub1": "S2", "key_block": "NONE"}
    if route[0] != 94 or any(branch.get(key) != value for key, value in expected.items()) or not 0 < int(effective["FA_S1_BASE"]) <= 128:
        raise RuntimeError("official Host did not select the Varlen L1 implementation")
    schedule = source.schedule(case["inputs"], hardware)
    mapping = {"s1_base": "FA_S1_BASE", "s2_base": "FA_S2_BASE", "n_ratio": "FA_N_RATIO",
               "core_num": "FA_CORE_NUM", "total_tasks": "FA_TOTAL_TASKS", "split_factor": "FA_SPLIT_FACTOR"}
    case["derived_schedule"].update(actual_implementation="varlen_l1", actual_branch_json=json.dumps(branch, sort_keys=True),
        schedule_disagreements={key: {"predicted": schedule[key], "host": effective.get(field)}
                                for key, field in mapping.items() if schedule[key] != effective.get(field)})
    return route


def execute_validation(collector, plan, hardware, forecasts, directory):
    csv_path = directory / "routes.csv"
    records, ensured, failures = collector.recorded(csv_path), set(), {}
    try:
        for index, case in enumerate(plan, 1):
            name = case["name"]
            if name in records:
                print(f"SKIP {index}/{len(plan)} case={name} status={records[name].get('status')}; saved result retained", flush=True)
                continue
            print(f"VALIDATION {index}/{len(plan)} stratum={case['derived_schedule']['stratum']} input={tuple(case['inputs'][key] for key in cost.INPUT_FIELDS)}", flush=True)
            folder = directory / "cases" / name
            folder.mkdir(parents=True, exist_ok=True)
            stage = "predict"
            row = {"version": collector.VERSION, "case": name, "split": "external", "family": "varlen",
                   "inputs": case["inputs"], "derived_schedule": case["derived_schedule"]}
            try:
                item = validation_forecast(case, forecasts, directory)
                prediction = item.get("result") if item["status"] == "predicted" else None
                case["derived_schedule"].update(prediction_status=item["status"], prediction_error=item.get("error", ""))
                row.update(model_sha256=item["model_sha256"], prediction_before_measurement=prediction is not None)
                stage = "discover"
                started = time.monotonic()
                route = validation_route(collector, case, folder, hardware)
                case["probe_host_wall_seconds"] = time.monotonic() - started
                stage = "measure"
                collector.profile_counters.cache_clear()
                collector.collect(case, hardware, directory, records, csv_path, ensured,
                                  frozen=forecasts["model"] if prediction else None, precomputed=prediction,
                                  build_failures=failures, discovered_route=route)
                row = records[name]
                row["derived_schedule"].update(case["derived_schedule"])
                row["derived_schedule"]["pmu_complete"] = extract_pmu(collector, directory, case, row)
            except Exception as error:
                row.update(status="failed", stage=stage, error=str(error))
                print(f"CASE FAILED: {name} stage={stage}: {error}; continuing", flush=True)
            finally:
                if row.get("status"):
                    row.setdefault("priority", case["derived_schedule"].get("observed_priority"))
                    row.setdefault("tiling_key", case["derived_schedule"].get("observed_tiling_key"))
                    records[name] = row
                    collector.append(csv_path, row)
                (ROOT / "output/output.bin").unlink(missing_ok=True)
                validation_report(collector, directory, forecasts)
    finally:
        validation_report(collector, directory, forecasts, final=True)
    print(f"CSV_REPORT={directory / 'diagnostics.csv'}", flush=True)


def validation_report(collector, directory, forecasts=None, final=False):
    if forecasts is None:
        path = directory / "predictions_before_measurement.json"
        forecasts = json.loads(path.read_text()) if path.is_file() else {"predictions": {}}
    records = collector.recorded(directory / "routes.csv")
    hardware_path = directory / "hardware.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.is_file() else {}
    target = directory / "diagnostics.csv"
    phases = ("startup", "drop_adapter", "bmm1", "vec1", "bmm2", "vec2")
    fields = ("case", "stratum", *cost.INPUT_FIELDS,
              "soc", "physical_npu_id", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes", "ub_bytes",
              "status", "stage", "error", "priority", "tiling_key",
              "model_sha256", "prediction_before_measurement", "prediction_frozen_utc", "prediction_status", "prediction_error",
              "expected_implementation", "actual_implementation", "cost_class", "selected_cost_model", "schedule_disagreements",
              "predicted_us", "baseline_predicted_us", "baseline_prediction_error", "duration_us", "signed_error_percent", "ape_percent", "baseline_ape_percent",
              "stable", "round1_us", "round2_us", "round3_us", "sample_count", "ci95_low_us", "ci95_high_us", "repeat_halfspan_percent",
              "checked_elements", "failed_elements", "failed_rows", "max_abs_error", "validation_version", "pmu_complete",
              "refinement_status", "source_conditions", "unseen_source_conditions", "micro_context_extrapolation", "sdk_api_matches_calibration",
              *(f"predicted_{phase}_critical_us" for phase in phases),
              *collector.REPORT_TILING, "profile_launches", "freq_min_mhz", "freq_max_mhz", "pmu_notes", *collector.PMU_FIELDS)
    errors, baseline_errors, noise, unstable, paired_errors, strata = [], [], [], 0, [], {}
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, row in records.items():
            derived = row.get("derived_schedule") or {}
            item = forecasts["predictions"].get(name, {})
            prediction = item.get("result") or {}
            refined = prediction.get("refinement") or {}
            effective = row.get("effective_tiling") or {}
            if isinstance(effective, str):
                effective = json.loads(effective)
            result = {**hardware, **row, **row["inputs"], **derived, **effective, **(row.get("correctness") or {})}
            result.update(prediction_frozen_utc=item.get("frozen_utc", ""), predicted_us=prediction.get("predicted_us", ""),
                          model_sha256=item.get("model_sha256", row.get("model_sha256", "")),
                          baseline_predicted_us=item.get("baseline_predicted_us", ""), cost_class=prediction.get("cost_class", ""),
                          baseline_prediction_error=item.get("baseline_prediction_error", ""),
                          selected_cost_model=prediction.get("selected_cost_model", ""), refinement_status=refined.get("status", ""),
                          sdk_api_matches_calibration=prediction.get("sdk_api_matches_calibration", ""),
                          ci95_low_us=row.get("median_ci95_low_us", ""), ci95_high_us=row.get("median_ci95_high_us", ""))
            result.update({key: refined.get(key, "") for key in ("source_conditions", "unseen_source_conditions", "micro_context_extrapolation")})
            result.update({f"predicted_{phase}_critical_us": prediction.get("critical_path_phase_us", {}).get(phase, "") for phase in phases})
            medians = row.get("round_medians_us") or []
            result.update({f"round{number}_us": value for number, value in enumerate(medians, 1)})
            actual = collector.finite(row.get("duration_us"))
            if actual and actual > 0:
                halfspan = 100 * (max(medians) - min(medians)) / (2 * actual) if medians else None
                result["repeat_halfspan_percent"] = halfspan
                valid = row.get("status") == "ok" and (row.get("correctness") or {}).get("checked_elements", 0) > 0 and derived.get("actual_implementation") == "varlen_l1"
                if result["predicted_us"]:
                    result["signed_error_percent"] = 100 * (result["predicted_us"] / actual - 1)
                    result["ape_percent"] = abs(result["signed_error_percent"])
                    if valid:
                        errors.append(result["ape_percent"])
                        strata.setdefault(derived.get("stratum", "unknown"), []).append(result["ape_percent"])
                        unstable += int(str(row.get("stable")).lower() != "true")
                        if halfspan is not None:
                            noise.append(halfspan)
                if result["baseline_predicted_us"]:
                    result["baseline_ape_percent"] = 100 * abs(result["baseline_predicted_us"] / actual - 1)
                    if valid and result["predicted_us"]:
                        baseline_errors.append(result["baseline_ape_percent"])
                        paired_errors.append(result["ape_percent"])
            metrics, frequencies, notes, launches = {}, [], set(), 0
            if row.get("tiling_key"):
                for number in range(1, collector.ROUNDS + 1):
                    try:
                        counters = collector.profile_counters(directory / "cases" / name / f"round_{number}", int(row["tiling_key"]))
                        launches += counters["launches"]
                        notes.update(counters["notes"])
                        frequencies.extend(counters["freq"])
                        for metric, values in counters["samples"].items():
                            metrics.setdefault(metric, []).extend(values)
                    except Exception as error:
                        notes.add(str(error)[:160])
            result.update({key: statistics.median(values) for key, values in metrics.items() if values})
            result.update(profile_launches=launches, pmu_notes=";".join(sorted(notes)),
                          freq_min_mhz=min(frequencies) if frequencies else "", freq_max_mhz=max(frequencies) if frequencies else "")
            writer.writerow({key: json.dumps(result[key], sort_keys=True, separators=(",", ":")) if isinstance(result.get(key), (dict, list)) else result.get(key, "") for key in fields})
    temporary.replace(target)
    states = Counter(row.get("status", "unknown") for row in records.values())
    print("VALIDATION PROGRESS: " + " ".join(f"{key}={value}" for key, value in states.items()), flush=True)
    if not final:
        return
    for label, values in (("FROZEN", errors), ("BASELINE_PAIRED", baseline_errors), ("MICRO_PAIRED", paired_errors), *strata.items()):
        if values:
            bands = " ".join(f">{limit}%={sum(value > limit for value in values)}/{len(values)}({100*sum(value > limit for value in values)/len(values):.1f}%)" for limit in (10, 20, 30))
            p90 = float(source.np.quantile(values, .9))
            print(f"{label}: n={len(values)} MdAPE={statistics.median(values):.3f}% P90={p90:.3f}% max={max(values):.3f}%; {bands}", flush=True)
    if noise:
        print(f"REPEAT NOISE: median_halfspan={statistics.median(noise):.3f}%; unstable={unstable}/{len(errors)}; all numerically valid cases retained", flush=True)
