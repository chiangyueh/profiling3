from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import collections
import csv
import gc
import hashlib
import json
import statistics
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

import fa_condition_study as conditions
import fa_latency_transfer as transfer


AXES = {
    "sequence_pair": lambda p: (p["S1"], p["S2"]),
    "head_combination": lambda p: (p["N1"], p["N2"]),
    "dimension_pair": lambda p: (p["D"], p["DV"]),
}
REFIT_PIPES = ("fixpipe", "vmte2", "vmte3")
QUANTITIES = ("gm_to_l1_kb", "l0c_to_gm_kb", "gm_to_ub_kb", "ub_to_gm_kb",
              "aic_cube_instructions", "aiv_mte2_instructions", "aiv_mte3_instructions")
ROOT = Path(__file__).resolve().parent


def read_samples(paths, model):
    samples, failed = [], []
    for path in paths:
        grouped = collections.defaultdict(list)
        for row in csv.DictReader(path.open()):
            if row["status"] != "ok":
                failed.append({"dataset": path.name, "case": row["case"],
                               "quality_flags": row.get("error", row["status"])})
                continue
            p = transfer.micro.source.cost.normalize({k: row[k] for k in transfer.micro.source.cost.INPUT_FIELDS})
            grouped[transfer.micro.source.cost.sha256(p)].append((row, p))
        for identity, rows in grouped.items():
            row, inputs = rows[0]
            effective = transfer.api_kernel_inputs(inputs, model["varlen_l1_transfer"]["api_input_normalization"])
            work = transfer.workload(effective, model["hardware"])
            flags = sorted({flag for r, _ in rows for flag in conditions.quality(r, effective, work)})
            active = transfer.micro.active_features(work)
            entry = transfer.micro.graph_entry(model["refined"], transfer.micro.cost_class(work.p, work.s),
                                              transfer.micro.execution_conditions(work))
            predicted = {pipe: sum(active[pipe].get(name, 0) * rate for name, rate in value["coefficients"].items())
                         for pipe, value in entry["pipes"].items()}
            predicted["mte2"] = sum(transfer.features(work)[name] * rate
                                    for name, rate in model["varlen_l1_transfer"]["coefficients"].items())
            quantities = {"gm_to_l1_kb": transfer.features(work)["payload_kb"],
                          "l0c_to_gm_kb": active["fixpipe"]["nd_write_kb"] + active["fixpipe"]["nz_write_kb"],
                          "gm_to_ub_kb": active["vmte2"]["read_kb"] + active["vmte2"]["alignment_read_kb"],
                          "ub_to_gm_kb": active["vmte3"]["write_kb"],
                          "aic_cube_instructions": active["cube"]["tiles"],
                          "aiv_mte2_instructions": active["vmte2"]["loads"],
                          "aiv_mte3_instructions": active["vmte3"]["stores"]}
            read_kb = loads = 0
            for pair in work.pairs:
                for tasks in pair:
                    for unit in tasks:
                        for _, stage2 in unit["vector"]:
                            if not stage2["accumulator_read_bytes"]:
                                read_kb += 4 * unit["rows"] * transfer.micro.source.cost.align(effective["DV"]) / 1024
                                loads += stage2["chunks"]
            samples.append({"id": identity, "inputs": inputs, "effective": effective,
                            "dataset": path.name, "stratum": row.get("stratum") or "previous_retest",
                            "flags": flags, "repeats": len(rows),
                            "actual_us": statistics.median(float(r["duration_us"]) for r, _ in rows),
                            "recorded_prediction_us": float(row["predicted_us"]),
                            "pipe_targets": {pipe: statistics.median(float(r[field]) for r, _ in rows)
                                             for pipe, (field, _) in transfer.micro.PIPES.items()},
                            "pipe_predicted": predicted, "active": active, "source_quantities": quantities,
                            "measured_quantities": {name: statistics.median(float(r[name]) for r, _ in rows)
                                                    for name in QUANTITIES},
                            "added_read_kb": read_kb, "added_loads": loads, "entry": entry})
        print("PIPELINE DATA:", path.name, "unique inputs=", len(grouped), flush=True)
    return samples, failed


def corrected_active(sample, pipe):
    values = dict(sample["active"][pipe])
    if pipe == "vmte2":
        values["read_kb"] += sample["added_read_kb"]
        values["loads"] += sample["added_loads"]
    return values


def fit_pipe(samples, pipe):
    names = transfer.micro.PIPES[pipe][1]
    x = np.array([[corrected_active(row, pipe).get(name, 0) for name in names] for row in samples])
    y = np.array([row["pipe_targets"][pipe] for row in samples])
    active = np.max(x, axis=0) > 0
    xx = x[:, active]
    scale = np.maximum(np.max(xx / y[:, None], axis=0), 1e-30)
    a = xx / y[:, None] / scale
    n = len(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        result = linprog(np.r_[np.zeros(sum(active)), np.ones(n) / n],
                         A_ub=np.vstack((np.c_[a, -np.eye(n)], np.c_[-a, -np.eye(n)])),
                         b_ub=np.r_[np.ones(n), -np.ones(n)], bounds=(0, None), method="highs",
                         options={"threads": 1})
    if not result.success:
        raise RuntimeError(result.message)
    rates = np.zeros(len(names))
    rates[active] = result.x[:sum(active)] / scale
    return dict(zip(names, rates)), [name for name, present in zip(names, active) if not present]


def evaluate(samples, model):
    qualified = [row for row in samples if not row["flags"]]
    fitted = {}
    for axis, group in AXES.items():
        for label in sorted({group(row["effective"]) for row in qualified}):
            train = [row for row in qualified if group(row["effective"]) != label]
            for pipe in REFIT_PIPES:
                fitted[axis, label, pipe] = fit_pipe(train, pipe)
    for index, row in enumerate(samples, 1):
        work = transfer.workload(row["effective"], model["hardware"])
        rates = np.array(list(row["entry"]["rates"]) +
                         [model["varlen_l1_transfer"]["coefficients"][name] for name in transfer.FEATURES])
        graph = transfer.branch_graph(work)
        row["before_us"] = float(transfer.micro.graph_time(graph, rates))
        del graph, work
        work = transfer.workload(row["effective"], model["hardware"], source_vector_reads=True)
        graph = transfer.branch_graph(work)
        value, gradient = transfer.micro.graph_time(graph, rates, True)
        row["corrected_us"] = float(value)
        row["critical"] = {pipe: sum(float(g * rate) for (key, _), g, rate
                                     in zip(transfer.PARAMETERS, gradient, rates) if key == pipe)
                           for pipe in transfer.micro.PIPES}
        row["cv"], row["cv_service"], row["diagnostic_oracle"] = {}, {}, {}
        if not row["flags"]:
            for axis, group in AXES.items():
                for changed in [(pipe,) for pipe in REFIT_PIPES] + [REFIT_PIPES]:
                    candidate = rates.copy()
                    supported = True
                    for pipe in changed:
                        coefficients, missing = fitted[axis, group(row["effective"]), pipe]
                        active = corrected_active(row, pipe)
                        supported &= not any(active.get(name, 0) > 0 for name in missing)
                        for j, (key, name) in enumerate(transfer.PARAMETERS):
                            if key == pipe:
                                candidate[j] = coefficients.get(name, 0)
                    row["cv"][axis + "/" + ",".join(changed)] = float(transfer.micro.graph_time(graph, candidate)) if supported else None
                for pipe in REFIT_PIPES:
                    coefficients, missing = fitted[axis, group(row["effective"]), pipe]
                    active = corrected_active(row, pipe)
                    row["cv_service"][axis + "/" + pipe] = sum(active.get(name, 0) * rate for name, rate in coefficients.items()) if not any(active.get(name, 0) > 0 for name in missing) else None
            for pipe in ("cube", "mte1", "mte2", "fixpipe", "vector", "vmte2", "vmte3"):
                candidate = rates.copy()
                service = row["pipe_predicted"][pipe]
                if pipe == "vmte2":
                    service = sum(corrected_active(row, pipe).get(name, 0) * rate
                                  for name, rate in row["entry"]["pipes"][pipe]["coefficients"].items())
                for j, (key, _) in enumerate(transfer.PARAMETERS):
                    if key == pipe:
                        candidate[j] *= row["pipe_targets"][pipe] / service
                row["diagnostic_oracle"][pipe] = float(transfer.micro.graph_time(graph, candidate))
        del graph, work
        gc.collect()
        if index % 20 == 0:
            print("PIPELINE AUDIT:", index, "/", len(samples), flush=True)


def ape(predicted, measured):
    return 100 * abs(predicted / measured - 1)


def write_report(path, samples, failed, paths, model):
    rows = []
    rows.append(dict(record_kind="methodology", notes=json.dumps({
        "model_sha256": transfer.micro.source.cost.sha256(model),
        "study_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "datasets": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "quality_policy": "same flags as fa_condition_study; no latency-error filtering; flagged timings retained in full-set metrics",
        "fit_target": "whole-device FixPipe/AIV MTE2/AIV MTE3 activity; existing features only; no kernel latency target",
        "held_out": "entire effective sequence pair, head combination, or dimension pair; all dtypes and repeats grouped",
        "counterfactual_scope": "diagnostic_oracle columns insert measured activity and cannot be used for input-only prediction",
        "pipe_predicted_scope": "engine service before the first-window accumulator read correction; corrected latency and GM-to-UB volume are reported separately",
        "scalar_scope": "measured scalar busy includes waiting; useful issue cost is an upper-bound calibration, not a scalar busy forecast",
        "stage_scope": "PMU measures whole-kernel engine sums; per-BMM/per-VEC independent times have not been measured",
        "cost_refits_activated": False, "new_measurements": 0, "npu_compilations": 0,
    }, sort_keys=True)))
    evidence = [
        ("vec2_first_window_read", "corrected", "ProcessVec2 unconditionally DataCopy(vec2Res) before s2LoopCount==0; adds 4*rows*align(Dv,16) bytes and one read call per Vec2 chunk; existing coefficients unchanged", "flash_attention_score_s1s2_bn2gs1.h:2235"),
        ("vec2_row_partition", "source_matched", "ComputeBmm1Tail uses aligned Dqk rather than Dv to set vec2S1BaseSize; existing partition preserved", "flash_attention_var_len_score.h:811"),
        ("softmax_stat_outputs", "source_matched", "SoftMaxCompute writes max in Vec1; SoftmaxDataCopyOut writes sum in Vec2; one 32-byte statistic of each kind per real row", "flash_attention_score_s1s2_bn2gs1.h:2062;2447"),
        ("short_kv", "source_matched", "S2<8 runs one Vec1 chunk; other Vec1 chunks use 8192/aligned_S2 rows; existing partition preserved", "flash_attention_var_len_score.h:802"),
        ("softmax_modes", "partially_identified", "first/update window and 8-row/64-column basic-block conditions checked; aggregate Vector activity does not independently identify each Softmax API mode", "flash_attention_score_common.h:143;flash_attention_score_s1s2_bn2gs1.h:1955"),
        ("cube_tail_and_mte1", "counter_checked", "source/Host MMAD counts exact on all quality-qualified inputs; arithmetic rounds each microtile to hardware granularity", "fa_latency_refinement.py:87"),
        ("fixpipe_transfer", "refit_not_promoted", "logical store bytes and measured transport counters differ at tails; blanket 16-row output padding has not been justified; independent cost refit fails held-out dimensions", "fa_latency_transfer.py:store"),
        ("query_key_value_probability_reuse", "accounted_not_fully_timed", "Q once per query task; K/V/P 256-row copies and ping-pong buffers counted; source payload matches qualified measurements but copy cost still varies", "flash_attention_var_len_score.h:837;901;950"),
        ("core_waves_and_gqa", "source_matched", "splitFactor assigns query tasks to AIV workers paired on AIC; GQA changes input stride and repeated KV reads; no per-stratum cost tables added", "flash_attention_var_len_score.h:560;950"),
        ("pipeline_start_and_drain", "source_traced_not_independently_timed", "C1-V1-C2-V2 peer waits and final drain compared with ProcessL1Carry; event graph models dependencies, independent hardware event times are absent", "flash_attention_var_len_score.h:560-735"),
        ("l2_capacity_and_addressing", "unresolved_generalization", "existing capacity and ND2NZ sector terms remain proxies; previous reuse refit failed held-out dimensions; no cache-hit assumption promoted", "fa_condition_audit.csv"),
        ("profiler_counter_scope", "quarantined_not_deleted", "time/count/payload inconsistencies flagged before cost fitting; source copy-call counts are not assumed identical to lowered DMA instruction counts", "fa_condition_study.py:quality"),
    ]
    for mechanism, status, notes, source in evidence:
        rows.append(dict(record_kind="source_evidence", mechanism=mechanism, decision=status,
                         notes=notes, source=source))
    for row in samples:
        item = dict(record_kind="sample", dataset=row["dataset"], case=row["id"], stratum=row["stratum"],
                    inputs=json.dumps(row["inputs"], sort_keys=True), effective_inputs=json.dumps(row["effective"], sort_keys=True),
                    quality_passed=not row["flags"], quality_flags=";".join(row["flags"]), repeats=row["repeats"],
                    measured_us=row["actual_us"], recorded_prediction_us=row["recorded_prediction_us"],
                    api_corrected_us=row["before_us"], source_read_corrected_us=row["corrected_us"],
                    api_ape_percent=ape(row["before_us"], row["actual_us"]),
                    corrected_ape_percent=ape(row["corrected_us"], row["actual_us"]))
        for name in QUANTITIES:
            item["source_" + name] = row["source_quantities"][name]
            item["measured_" + name] = row["measured_quantities"][name]
        item["corrected_gm_to_ub_kb"] = row["source_quantities"]["gm_to_ub_kb"] + row["added_read_kb"]
        for pipe in transfer.micro.PIPES:
            item[pipe + "_predicted_active_sum_us"] = row["pipe_predicted"][pipe]
            item[pipe + "_measured_active_sum_us"] = row["pipe_targets"][pipe]
            item[pipe + "_critical_path_us"] = row["critical"][pipe]
        item.update({"cv_" + name: value for name, value in row["cv"].items()})
        item.update({"diagnostic_oracle_" + name + "_us": value for name, value in row["diagnostic_oracle"].items()})
        rows.append(item)
    rows.extend(dict(record_kind="failed_collection", **row) for row in failed)
    latest = [row for row in samples if row["dataset"] == paths[-1].name]
    qualified = [row for row in latest if not row["flags"]]
    cohorts = {"latest_all": latest, "latest_quality_passed": qualified,
               "previous_quality_passed": [r for r in samples if r["dataset"] != paths[-1].name and not r["flags"]]}
    cohorts.update({"stratum/" + name: [r for r in latest if r["stratum"] == name] for name in sorted({r["stratum"] for r in latest})})
    def summary(cohort, mechanism, values, **kwargs):
        stats = conditions.stats(values)
        rows.append(dict(record_kind="summary", dataset=cohort, mechanism=mechanism, **stats, **kwargs))
        return stats
    for cohort, selected in cohorts.items():
        for variant in ("before_us", "corrected_us"):
            summary(cohort, variant, [ape(r[variant], r["actual_us"]) for r in selected])
        if cohort.startswith("stratum/"):
            continue
        for pipe in transfer.micro.PIPES:
            summary(cohort, "pipe/" + pipe, [ape(r["pipe_predicted"][pipe], r["pipe_targets"][pipe]) for r in selected],
                    notes="whole-kernel activity; scalar busy includes waits and is not a useful-issue target" if "scalar" in pipe else "whole-kernel engine activity; not independently measured stage time")
        for name in QUANTITIES:
            summary(cohort, "quantity/" + name, [ape(r["source_quantities"][name], r["measured_quantities"][name]) for r in selected])
        summary(cohort, "quantity/corrected_gm_to_ub_kb", [ape(r["source_quantities"]["gm_to_ub_kb"] + r["added_read_kb"], r["measured_quantities"]["gm_to_ub_kb"]) for r in selected])
    reference = conditions.stats([ape(r["corrected_us"], r["actual_us"]) for r in qualified])
    cv_metrics = {}
    for key in qualified[0]["cv"]:
        values = [ape(r["cv"][key], r["actual_us"]) for r in qualified if r["cv"][key] is not None]
        cv_metrics[key] = summary("latest_quality_passed", "cv/" + key, values,
                                  notes="retrospective grouped validation; candidate coefficient changes not active")
    for changed in [(pipe,) for pipe in REFIT_PIPES] + [REFIT_PIPES]:
        name = ",".join(changed)
        worse = [axis + "/" + metric for axis in AXES for metric in ("median", "p90", "above20", "above30")
                 if cv_metrics[axis + "/" + name][metric] > reference[metric] + 1e-9]
        rows.append(dict(record_kind="decision", mechanism="refit/" + name, decision="not_promoted",
                         notes=json.dumps({"worse_than_frozen_source_correction": worse,
                                           "prospective_validation_required": True, "coefficients_activated": False})))
    for pipe in qualified[0]["diagnostic_oracle"]:
        summary("latest_quality_passed", "diagnostic_oracle/" + pipe,
                [ape(r["diagnostic_oracle"][pipe], r["actual_us"]) for r in qualified],
                notes="uses measured activity to diagnose sensitivity; unavailable to input-only inference")
    historical = ROOT / "output/fa_input_latency_v1/diagnostics.csv"
    if historical.is_file():
        inventory = collections.defaultdict(list)
        for row in csv.DictReader(historical.open()):
            if row["status"] == "ok":
                inventory[row["family"]].append(row)
        for family, part in sorted(inventory.items()):
            rows.append(dict(record_kind="historical_family_inventory", mechanism=family, n=len(part),
                             notes=json.dumps({"with_counter_fields": sum(all(r.get(k) not in ("", None) for k in ("aic_cube_active_sum_us", "gm_to_l1_kb", "aiv_vec_active_sum_us", "aic_cube_instructions")) for r in part),
                                               "numerically_checked": sum(float(r.get("checked_elements") or 0) > 0 for r in part),
                                               "scope": "historical successful records; not new independent validation and not all quality-qualified"})))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("PIPELINE_AUDIT=" + str(path), flush=True)
    for row in rows:
        if row.get("record_kind") == "summary" and row.get("dataset") == "latest_quality_passed":
            print(json.dumps(row), flush=True)


def main():
    import argparse
    import resource

    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1100 * 1024 ** 2, 1100 * 1024 ** 2))
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "fa_pipeline_audit.csv")
    args = parser.parse_args()
    model = json.loads((ROOT / "fa_branch_model.json").read_text())
    samples, failed = read_samples(args.paths, model)
    evaluate(samples, model)
    write_report(args.output, samples, failed, args.paths, model)


if __name__ == "__main__":
    main()
