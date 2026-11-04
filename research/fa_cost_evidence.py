from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import collections
import csv
import itertools
import json

import numpy as np
from scipy.stats import binomtest

from research import fa_cost_validation as validation
from research import fa_joint_calibration as joint


def format_records(records):
    extra_names = {"same_input_dtype_pairs", "same_input_layout_pairs", "structurally_zero_slots", "unmeasured_nonzero_slots"}
    records = [row for row in records if not (row["record_type"] == "evidence_metadata" and row.get("name") in extra_names)]
    structural = {
        **{"mte2/" + name + "/address_gap_kb": "BNSD stride equals row width in matmul_cache_trace and cube_job"
           for name in ("batch:BNSD", "direct:BNSD", "kfc:BNSD")},
        **{"vector/" + name + "/online_k": "batch and S1 workloads have one S2 window, so first is always true"
           for name in ("batch", "s1")},
        **{"vmte2/" + name + "/alignment_read_kb": "padded equals align(columns) outside S1 and SameAB vector functions"
           for name in ("batch", "flash")},
    }
    unmeasured = []
    pair_counts = collections.Counter()
    for row in records:
        if row["record_type"] == "metadata" and row["name"] == "inference":
            row["value_json"] = json.dumps("paired sequence-group bootstrap; final two-sided sign tests with Holm correction; frozen and refitted effects separated")
        if row["record_type"] == "exposed_ablation" and row.get("scope") == "schema_was_already_shared":
            row["p90_ape_percent"] = row["baseline_p90_ape_percent"] = ""
        if row["record_type"] == "heldout_pipe_service" and row.get("scope") == "scalar_upper_bound_violation" and row.get("median_ape_percent") not in (None, ""):
            values = {"median_upper_violation_percent": float(row.get("median_ape_percent") or 0),
                      "p90_upper_violation_percent": float(row.get("p90_ape_percent") or 0),
                      "max_upper_violation_percent": float(row.get("max_ape_percent") or 0)}
            row["value_json"] = json.dumps(values, separators=(",", ":"))
            for key in ("median_ape_percent", "p90_ape_percent", "max_ape_percent", "baseline_median_error_percent",
                        "above10_count", "above20_count", "above30_count"):
                row[key] = ""
        if row["record_type"] == "parameter_coverage" and int(row["samples"]) == 0:
            details = json.loads(row["value_json"])
            if row["name"] in structural:
                row["scope"] = "structurally_zero_in_current_source_model"
                details["source_rule"] = structural[row["name"]]
            else:
                row["scope"] = "unmeasured_nonzero_possible"
                unmeasured.append(row["name"])
            row["value_json"] = json.dumps(details, separators=(",", ":"))
        if row["record_type"] == "matched_control":
            pair_counts[row["name"], row["pipe"]] += int(row["samples"])
    for axis in ("dtype", "layout"):
        count = max((value for (name, _), value in pair_counts.items() if name == axis), default=0)
        records.append({"record_type": "evidence_metadata", "name": "same_input_" + axis + "_pairs",
                        "value_json": json.dumps(count)})
    records.append({"record_type": "evidence_metadata", "name": "structurally_zero_slots", "value_json": json.dumps(structural, separators=(",", ":"))})
    records.append({"record_type": "evidence_metadata", "name": "unmeasured_nonzero_slots", "value_json": json.dumps(unmeasured)})
    return records


def interval_distance(predictions, low, high, actual):
    return np.maximum(np.maximum(low - predictions, predictions - high), 0) / actual * 100


def paired_interval(candidate, baseline, rows, indices):
    groups = collections.defaultdict(list)
    for i in indices:
        groups[joint.group(rows[i])].append(candidate[i] - baseline[i])
    values = np.array([np.median(value) for _, value in sorted(groups.items())])
    rng = np.random.default_rng(20261104)
    bootstrap = values[rng.integers(len(values), size=(2000, len(values)))]
    low, high = np.quantile(np.median(bootstrap, axis=1), (.025, .975))
    positive, negative = int(sum(values > 1e-7)), int(sum(values < -1e-7))
    p = float(binomtest(positive, positive + negative, .5, alternative="greater").pvalue) if positive + negative else 1.0
    return {"interval_paired_delta_pp": float(np.median(values)),
            "interval_ci_low_pp": float(low), "interval_ci_high_pp": float(high),
            "interval_positive_groups": positive, "interval_negative_groups": negative,
            "interval_split_benefit_p": p}


def frozen_predictions(rows, keys, features, baseline, specifications):
    matrices = {spec["name"]: validation.projection(keys, spec)[1] for spec in specifications}
    folds = np.array([joint.fold(row) for row in rows])
    rates = {}
    for fold in range(4):
        saved = json.loads((validation.CACHE / ("outer_%d.json" % fold)).read_text())["results"]["baseline"]
        reference = np.asarray(saved["full_rates"])
        weights = np.sum(features[folds != fold], axis=(0, 1))
        rates[fold] = {}
        for spec in specifications:
            if spec["kind"] == "coefficient_removal":
                changed = reference.copy()
                changed[keys.index(spec["removed"][0])] = 0
            else:
                matrix = matrices[spec["name"]]
                shared = matrix.T @ (weights * reference) / np.maximum(matrix.T @ weights, 1e-30)
                for column in range(matrix.shape[1]):
                    indices = np.flatnonzero(matrix[:, column])
                    if len(indices) == 1:
                        shared[column] = reference[indices[0]]
                changed = matrix @ shared
            rates[fold][spec["name"]] = changed
    results = {spec["name"]: np.full(len(rows), np.nan) for spec in specifications}
    monotonicity_checks = 0
    for i, row in enumerate(rows):
        graph = joint.load_graph(row)
        reference = rates[folds[i]]["baseline"]
        for spec in specifications:
            changed = rates[folds[i]][spec["name"]]
            if np.array_equal(changed, reference):
                value = baseline[i]
            else:
                value = joint.micro.source.graph_time(graph, changed[row["mapping"]])
            if spec["kind"] == "coefficient_removal":
                if value > baseline[i] + 1e-7 + abs(baseline[i]) * 1e-9:
                    raise ValueError("removing a nonnegative cost increased the frozen DAG latency")
                monotonicity_checks += 1
            results[spec["name"]][i] = value
        del graph
        if (i + 1) % 200 == 0:
            print("FROZEN COST AUDIT", i + 1, "/", len(rows), flush=True)
    return results, monotonicity_checks


def annotate():
    with validation.REPORT.open() as stream:
        report = list(csv.DictReader(stream))
    if any(row["record_type"] == "matched_control" for row in report):
        raise RuntimeError("evidence has already been appended")
    rows, _, _ = joint.dataset()
    keys, features, targets = joint.matrices(rows)
    times = np.array([row["duration_us"] for row in rows])
    old = {row["input_sha256"]: row for row in csv.DictReader((joint.DEST / "fa_branch_audit.csv").open())
           if row["record_type"] == "kernel_case"}
    low, high = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if row["input_sha256"] in old:
            record = old[row["input_sha256"]]
            low[i], high[i] = float(record["measurement_interval_low_us"]), float(record["measurement_interval_high_us"])
    full = {}
    services = {}
    coefficient_folds = []
    fits = []
    nested_services = np.full(targets.shape, np.nan)
    nested_predictions = np.full(len(rows), np.nan)
    choices = {int(row["fold"]): json.loads(row["value_json"])["selected"]["name"]
               for row in report if row["record_type"] == "nested_selection"}
    for fold in range(4):
        scope = json.loads((validation.CACHE / ("outer_%d.json" % fold)).read_text())["results"]
        coefficient_folds.append(scope["baseline"]["full_rates"])
        fits.extend(scope.values())
        selected = scope[choices[fold]]
        indices = np.array(selected["indices"])
        nested_services[indices] = np.einsum("ijk,k->ij", features[indices], selected["full_rates"])
        values = validation.array_for(selected, len(rows));included = np.isfinite(values)
        nested_predictions[included] = values[included]
        for name, fitted in scope.items():
            if name not in full:
                full[name] = np.full(len(rows), np.nan)
                services[name] = np.full(targets.shape, np.nan)
            prediction = validation.array_for(fitted, len(rows))
            included = np.isfinite(prediction)
            full[name][included] = prediction[included]
            indices = np.array(fitted["indices"])
            services[name][indices] = np.einsum("ijk,k->ij", features[indices], fitted["full_rates"])
    baseline_interval = interval_distance(full["baseline"], low, high, times)
    noop_changes = np.maximum(np.abs(full["pool:mte1"] - full["baseline"]),
                              np.abs(full["pool:fixpipe"] - full["baseline"]))
    optimization_resolution = float(np.nanmax(noop_changes / times * 100))
    specifications = validation.variants(keys, np.any(features > 0, axis=(0, 1)))
    frozen, monotonicity_checks = frozen_predictions(rows, keys, features, full["baseline"], specifications)
    geometry_status = ["manual_l1_source_corrected" if row["implementation"] == "varlen_l1" else
                       "tiling_not_recorded" if "geometry_differences" not in row else
                       "recorded_tiling_differs" if row["geometry_differences"] else "recorded_tiling_matches"
                       for row in rows]
    ablations = {row["name"]: row for row in report if row["record_type"] == "ablation"}
    specifications_by_name = {spec["name"]: spec for spec in specifications}
    family_labels = ["drop_adapter" if row.get("collection_group") == "drop_adapter" else row["implementation"] for row in rows]
    branch_baseline = {}
    for family in sorted(set(family_labels)):
        included = np.array([label == family for label in family_labels])
        branch_baseline[family] = validation.risk(np.where(included, full["baseline"], np.nan), rows, times)
    pipe_records = []
    baseline_pipe_errors = {}
    for j, pipe in enumerate(validation.PIPE_ORDER):
        observed = targets[:, j] > 0
        residual = services["baseline"][observed, j] / targets[observed, j] - 1
        error = np.maximum(residual, 0) * 100 if pipe in ("cscalar", "vscalar") else np.abs(residual) * 100
        baseline_pipe_errors[pipe] = joint.statistics(error)
    for name, item in ablations.items():
        candidate_interval = interval_distance(full[name], low, high, times)
        included = np.flatnonzero(np.isfinite(candidate_interval) & np.isfinite(baseline_interval))
        stats = joint.statistics(candidate_interval[included])
        item["repeat_interval_cases"] = len(included)
        item["interval_median_error_percent"] = stats["median_ape_percent"]
        item["interval_p90_error_percent"] = stats["p90_ape_percent"]
        item.update(paired_interval(candidate_interval, baseline_interval, rows, included))
        stats = validation.risk(frozen[name], rows, times)
        item["frozen_median_ape_percent"] = stats["median_ape_percent"]
        item["frozen_p90_ape_percent"] = stats["p90_ape_percent"]
        item["frozen_max_prediction_change_us"] = float(np.nanmax(np.abs(frozen[name] - full["baseline"])))
        item["frozen_max_prediction_change_percent"] = float(np.nanmax(np.abs(frozen[name] - full["baseline"]) / times * 100))
        item["frozen_no_latency_effect"] = item["frozen_max_prediction_change_us"] < 1e-7
        item["refit_changes_when_frozen_does_not"] = item["frozen_no_latency_effect"] and float(item["max_prediction_change_us"]) >= 1e-7
        branch_deltas = []
        for family in sorted(set(family_labels)):
            members = np.array([label == family for label in family_labels])
            stats = validation.risk(np.where(members, full[name], np.nan), rows, times)
            branch_deltas.append((family, stats["median_ape_percent"] - branch_baseline[family]["median_ape_percent"],
                                  stats["p90_ape_percent"] - branch_baseline[family]["p90_ape_percent"]))
        item["worst_family_median_delta_pp"] = max(value[1] for value in branch_deltas)
        worst = max(branch_deltas, key=lambda value: value[2])
        item["worst_family_p90_delta_pp"] = worst[2]
        item["worst_family_p90"] = worst[0]
        item["no_family_median_or_p90_regression"] = all(value[1] <= 1e-7 and value[2] <= 1e-7 for value in branch_deltas)
        for status in sorted(set(geometry_status)):
            subset = np.array([label == status for label in geometry_status])
            predictions = np.where(subset, full[name], np.nan)
            stats = validation.risk(predictions, rows, times)
            report.append({"record_type": "geometry_stratum", "name": name, "scope": status, **stats})
        for j, pipe in enumerate(validation.PIPE_ORDER):
            observed = (targets[:, j] > 0) & np.isfinite(services[name][:, j])
            residual = services[name][observed, j] / targets[observed, j] - 1
            error = np.maximum(residual, 0) * 100 if pipe in ("cscalar", "vscalar") else np.abs(residual) * 100
            stats = joint.statistics(error)
            pipe_records.append({"record_type": "heldout_pipe_service", "name": name, "pipe": pipe,
                                 "scope": "scalar_upper_bound_violation" if pipe in ("cscalar", "vscalar") else "whole_device_busy_service",
                                 **stats, "baseline_median_error_percent": baseline_pipe_errors[pipe]["median_ape_percent"]})
    ranked = sorted((row for row in ablations.values() if row["kind"] != "baseline"),
                    key=lambda row: row["interval_split_benefit_p"])
    adjusted = 0
    for position, row in enumerate(ranked):
        adjusted = max(adjusted, min(1, row["interval_split_benefit_p"] * (len(ranked) - position)))
        row["interval_holm_p"] = adjusted
        row["retention_supported_with_repeat_intervals"] = adjusted <= .05 and row["interval_ci_low_pp"] > 0
    for prefix in ("", "interval_"):
        for row in ranked:
            positive = int(row[prefix + "positive_groups"])
            negative = int(row[prefix + "negative_groups"])
            row[prefix + "two_sided_p"] = float(binomtest(positive, positive + negative, .5).pvalue) if positive + negative else 1.0
        ordered = sorted(ranked, key=lambda row: row[prefix + "two_sided_p"])
        adjusted = 0
        for position, row in enumerate(ordered):
            adjusted = max(adjusted, min(1, row[prefix + "two_sided_p"] * (len(ordered) - position)))
            row[prefix + "two_sided_holm_p"] = adjusted
            if prefix:
                row["retention_supported_with_repeat_intervals"] = adjusted <= .05 and row["interval_ci_low_pp"] > optimization_resolution
            elif row["frozen_no_latency_effect"]:
                row["conclusion"] = "no_direct_latency_effect_in_frozen_outer_models"
            elif adjusted <= .05 and float(row["paired_ci_low_pp"]) > optimization_resolution:
                row["conclusion"] = "retention_supported_for_recalibration_recipe"
            elif adjusted <= .05 and float(row["paired_ci_high_pp"]) < -optimization_resolution and float(row["p90_ape_percent"]) <= float(ablations["baseline"]["p90_ape_percent"]):
                row["conclusion"] = "reduction_supported_for_recalibration_recipe"
            else:
                row["conclusion"] = "inconclusive"
    exposure_records = []
    for name, item in ablations.items():
        if item["kind"] == "baseline":
            continue
        spec = specifications_by_name[name]
        matrix = validation.projection(keys, spec)[1]
        if spec["kind"] == "coefficient_removal":
            affected = np.array([key in spec["removed"] for key in keys])
        else:
            affected = np.any(matrix[:, np.sum(matrix, axis=0) > 1] > 0, axis=1)
        exposed = np.any(features[:, :, affected] > 0, axis=(1, 2))
        if not np.any(exposed):
            exposure_records.append({"record_type": "exposed_ablation", "name": name, "n": 0, "groups": 0,
                                     "two_sided_p": 1.0, "paired_ci_low_pp": 0.0, "paired_ci_high_pp": 0.0,
                                     "p90_ape_percent": None, "baseline_p90_ape_percent": None,
                                     "scope": "schema_was_already_shared"})
            continue
        a = np.where(exposed, full["baseline"], np.nan)
        b = np.where(exposed, full[name], np.nan)
        stats = validation.risk(b, rows, times)
        paired = validation.paired_audit(a, b, rows, times)
        positive, negative = paired["positive_groups"], paired["negative_groups"]
        p = float(binomtest(positive, positive + negative, .5).pvalue) if positive + negative else 1.0
        exposure_records.append({"record_type": "exposed_ablation", "name": name, **stats, **paired,
                                 "two_sided_p": p, "baseline_p90_ape_percent": validation.risk(a, rows, times)["p90_ape_percent"]})
    adjusted = 0
    for position, item in enumerate(sorted(exposure_records, key=lambda row: row["two_sided_p"])):
        adjusted = max(adjusted, min(1, item["two_sided_p"] * (len(exposure_records) - position)))
        item["two_sided_holm_p"] = adjusted
        main = ablations[item["name"]]
        if main["frozen_no_latency_effect"]:
            conclusion = "no_direct_latency_effect_in_frozen_outer_models"
        elif adjusted <= .05 and item["paired_ci_low_pp"] > optimization_resolution:
            conclusion = "retention_supported_for_exposed_recalibration_cases"
        elif adjusted <= .05 and item["paired_ci_high_pp"] < -optimization_resolution and item["p90_ape_percent"] <= item["baseline_p90_ape_percent"]:
            conclusion = "reduction_supported_for_exposed_recalibration_cases"
        else:
            conclusion = "inconclusive"
        item["conclusion"] = conclusion
        main["exposed_conclusion"] = conclusion
        main["exposed_groups"] = item["groups"]
        main["exposed_holm_p"] = adjusted
    report.extend(exposure_records)
    report.extend(pipe_records)
    for family in sorted(set(family_labels)):
        included = np.array([label == family for label in family_labels])
        for name, prediction in (("baseline", full["baseline"]), ("nested_selected", nested_predictions)):
            report.append({"record_type": "family_holdout", "name": name, "implementation": family,
                           **validation.risk(np.where(included, prediction, np.nan), rows, times)})
    for j, pipe in enumerate(validation.PIPE_ORDER):
        observed = (targets[:, j] > 0) & np.isfinite(nested_services[:, j])
        residual = nested_services[observed, j] / targets[observed, j] - 1
        error = np.maximum(residual, 0) * 100 if pipe in ("cscalar", "vscalar") else np.abs(residual) * 100
        report.append({"record_type": "heldout_pipe_service", "name": "nested_selected", "pipe": pipe,
                       "scope": "scalar_upper_bound_violation" if pipe in ("cscalar", "vscalar") else "whole_device_busy_service",
                       **joint.statistics(error), "baseline_median_error_percent": baseline_pipe_errors[pipe]["median_ape_percent"]})
    controls = collections.defaultdict(list)
    for axis in ("dtype", "layout"):
        grouped = collections.defaultdict(list)
        for row in rows:
            key = tuple(row["inputs"][field] for field in joint.micro.source.cost.INPUT_FIELDS if field != axis)
            grouped[key].append(row)
        for matches in grouped.values():
            for a, b in itertools.combinations(sorted(matches, key=lambda row: row["inputs"][axis]), 2):
                if a["inputs"][axis] == b["inputs"][axis]:
                    continue
                for pipe in validation.PIPE_ORDER:
                    if a["targets"][pipe] <= 0 or b["targets"][pipe] <= 0:
                        continue
                    columns = sorted(set(a["input_features"][pipe]) | set(b["input_features"][pipe]))
                    equal = np.allclose([a["input_features"][pipe].get(key, 0) for key in columns],
                                        [b["input_features"][pipe].get(key, 0) for key in columns], rtol=1e-9, atol=1e-8)
                    ignored = 1 if axis == "dtype" else 2
                    first = [value for i, value in enumerate(a["cost_class"].split("|")) if i != ignored]
                    second = [value for i, value in enumerate(b["cost_class"].split("|")) if i != ignored]
                    traced = a.get("geometry_differences") == [] and b.get("geometry_differences") == []
                    label = axis, pipe, a["inputs"][axis], b["inputs"][axis], bool(equal), first == second, traced
                    controls[label].append((joint.group(a), b["targets"][pipe] / a["targets"][pipe]))
    for label, values in sorted(controls.items()):
        axis, pipe, left, right, equal, same_structure, traced = label
        grouped = collections.defaultdict(list)
        for group, ratio in values:
            grouped[group].append(ratio)
        group_values = np.array([np.median(value) for _, value in sorted(grouped.items())])
        rng = np.random.default_rng(20261104)
        bootstrap = group_values[rng.integers(len(group_values), size=(2000, len(group_values)))]
        interval = np.quantile(np.median(bootstrap, axis=1), (.025, .975))
        report.append({"record_type": "matched_control", "name": axis, "pipe": pipe,
                       "left_context": left, "right_context": right, "source_work_equal": equal,
                       "source_class_equal_except_axis": same_structure, "recorded_tiling_matches_both": traced,
                       "samples": len(values), "groups": len(grouped), "median_service_ratio": float(np.median(group_values)),
                       "service_ratio_ci_low": float(interval[0]), "service_ratio_ci_high": float(interval[1])})
    folds = np.array(coefficient_folds)
    zero_in_every_fit = [key for i, key in enumerate(keys) if np.all(folds[:, i] <= 1e-10) and np.any(features[:, :, i] > 0)]
    metadata = {
        "repeat_interval_metric": "distance outside historical min/max repeat interval divided by measured median; all 311 matched medians lie inside their intervals",
        "repeat_interval_cases": int(sum(np.isfinite(low))),
        "median_repeat_full_span_percent": float(np.nanmedian((high - low) / times * 100)),
        "zero_in_all_outer_fits": zero_in_every_fit,
        "pipeline_metric_scope": "PMU whole-device pipeline busy sums, not independently measured BMM1/Vec1/BMM2/Vec2 wall times; scalar busy is only an upper bound",
        "frozen_audit_scope": "all other fitted costs held fixed; merged values weighted only by training input work counts; literal zeros certified directly",
        "refit_audit_scope": "complete recalibration under each schema, including a newly estimated training-only PMU prior; effects can include redistribution of other costs",
        "single_cost_removal_monotonicity_checks": monotonicity_checks,
        "matched_control_scope": "same operator inputs except dtype or layout; equal source-work subset separated; ratios are diagnostic, not causal instruction timing",
        "multiple_comparisons": "two-sided sign tests with Holm correction across all fixed ablations, for both retention and reduction claims; frozen direct effects reported separately",
        "exposure_scope": "additional tests restricted to inputs that actually use removed coefficients or merged source contexts; not diluted by unrelated branches",
        "api_control_limitation": "official dispatch selects API from inputs; no same-input randomized API comparison is available",
        "outer_fit_models": len(fits), "outer_fits_at_iteration_limit": sum(fit["optimizer_at_iteration_limit"] for fit in fits),
        "geometry_status_counts": dict(collections.Counter(geometry_status)),
        "geometry_status_scope": "recorded CPU-reconstructed versus traced Matmul geometry; mismatch rows retained, not filtered or corrected by fitted costs",
        "training_envelope_relative_tolerance": 1e-5,
        "heldout_prediction_engine": "every heldout prediction evaluated on the full source DAG; never the training path envelope",
        "max_noop_error_resolution_pp": optimization_resolution,
        "noop_control_scope": "MTE1 and Fixpipe were already shared; their identical-schema continuation bounds observed optimizer effects, not instruction cost",
    }
    for name in ("numeric_check", "input_class_check"):
        path = validation.CACHE / (name + ".json")
        if path.exists():
            metadata[name] = json.loads(path.read_text())
    scope_files = [path for path in validation.CACHE.glob("*.json")
                   if path.name.startswith(("inner_", "outer_", "combined_", "final."))]
    completed = [fit for path in scope_files for fit in json.loads(path.read_text())["results"].values()]
    metadata["total_fitting_runs"] = len(completed)
    metadata["all_fits_at_iteration_limit"] = sum(fit["optimizer_at_iteration_limit"] for fit in completed)
    report.extend({"record_type": "evidence_metadata", "name": key, "value_json": json.dumps(value, separators=(",", ":"))}
                  for key, value in metadata.items())
    report = format_records(report)
    fields = list(report[0])
    fields.extend(sorted({key for row in report for key in row if key not in fields}))
    temporary = validation.REPORT.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader();writer.writerows(report)
    temporary.replace(validation.REPORT)
    print("COST EVIDENCE COMPLETE", json.dumps(metadata), flush=True)
