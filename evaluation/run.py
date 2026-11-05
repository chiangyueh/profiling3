import os
import resource

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (1100 * 1024 ** 2, 1100 * 1024 ** 2))

import csv
import gc
import json
import math
import signal
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
sys.path.insert(0, str(ROOT))

from clean.predictor import predict
from clean.runtime.inputs import INPUT_FIELDS
from clean.runtime.schedules.drop_adapter_schedule import enabled as adapter_enabled

DATA = ROOT / "calibration/measurements.csv"
PREVIOUS = ROOT / "research/reports/clean_inference_audit.csv"
PHASES = ("startup", "drop_adapter", "bmm1", "vec1", "bmm2", "vec2")


def identity(inputs):
    return tuple(str(inputs[key]) for key in INPUT_FIELDS)


def read_previous():
    previous = {}
    with PREVIOUS.open(newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            if not row.get("inputs") or not row.get("predicted_us"):
                continue
            key = identity(json.loads(row["inputs"]))
            value = float(row["predicted_us"])
            if key in previous and previous[key][0] != value:
                raise ValueError("conflicting historical predictions")
            previous[key] = value, line
    return previous


def timeout(signum, frame):
    raise TimeoutError("clean.predict CPU evaluation exceeded its time limit")


def apply_prediction(row, prediction):
    actual = row["measured_us"]
    value = float(prediction["predicted_us"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid predicted latency")
    error = (value - actual) / actual * 100
    row.update({"predicted_us": value, "signed_error_us": value - actual, "signed_error_pct": error,
                "absolute_error_pct": abs(error), "within10": abs(error) <= 10,
                "within20": abs(error) <= 20, "within30": abs(error) <= 30,
                "predicted_implementation": prediction["implementation"],
                "route_matches_record": prediction["implementation"] == row["recorded_implementation"],
                "status": "ok", "error": ""})
    row.update({"phase_" + phase + "_us": prediction["critical_path_phase_us"][phase] for phase in PHASES})
    if row["previous_clean_predicted_us"] != "":
        row["error_change_pp"] = abs(error) - row["previous_clean_absolute_error_pct"]


def retry_timeouts(rows):
    retried = []
    handler = signal.signal(signal.SIGALRM, timeout)
    try:
        for row in rows:
            if row["status"] != "failed" or not row["error"].startswith("TimeoutError:"):
                continue
            retried.append(row["case"])
            row["first_pass_error"] = row["error"]
            row["cpu_attempts"] += 1
            print("CPU TIMEOUT RETRY case=%s limit=180s" % row["case"], flush=True)
            try:
                signal.alarm(180)
                apply_prediction(row, predict({key: row[key] for key in INPUT_FIELDS}))
            except Exception as error:
                row.update({"status": "failed", "error": type(error).__name__ + ": " + str(error)})
            finally:
                signal.alarm(0)
                gc.collect()
            print("CPU TIMEOUT RETRY END case=%s status=%s" % (row["case"], row["status"]), flush=True)
    finally:
        signal.signal(signal.SIGALRM, handler)
    return retried


def save_rows(rows):
    fields = list(rows[0])
    with (FOLDER / "errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, fraction):
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lower = int(at)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (at - lower)


def summarize(kind, name, rows):
    good = [row for row in rows if row["status"] == "ok"]
    errors = [row["absolute_error_pct"] for row in good]
    paired = [row for row in good if row["previous_clean_predicted_us"] != ""]
    result = {
        "group_kind": kind, "group": name, "inputs": len(rows), "predicted": len(good),
        "failed": len(rows) - len(good),
        "route_mismatch_count": sum(not row["route_matches_record"] for row in good),
        "host_geometry_flagged_count": sum(bool(row["host_geometry_disagreements"]) for row in rows),
    }
    if errors:
        result.update({
            "median_absolute_error_pct": statistics.median(errors),
            "p90_absolute_error_pct": percentile(errors, 0.90),
            "p95_absolute_error_pct": percentile(errors, 0.95),
            "max_absolute_error_pct": max(errors),
            "median_signed_error_pct": statistics.median(row["signed_error_pct"] for row in good),
            "median_absolute_error_us": statistics.median(abs(row["signed_error_us"]) for row in good),
        })
    for threshold in (10, 20, 30, 50):
        within = sum(error <= threshold for error in errors)
        result["within%d_count" % threshold] = within
        result["within%d_pct_of_inputs" % threshold] = within / len(rows) * 100
        result["above%d_count" % threshold] = len(good) - within
        result["above%d_pct_of_predictions" % threshold] = (len(good) - within) / len(good) * 100 if good else ""
    result["paired_previous_count"] = len(paired)
    if paired:
        result.update({
            "paired_previous_median_error_pct": statistics.median(row["previous_clean_absolute_error_pct"] for row in paired),
            "paired_current_median_error_pct": statistics.median(row["absolute_error_pct"] for row in paired),
            "paired_previous_p90_error_pct": percentile([row["previous_clean_absolute_error_pct"] for row in paired], 0.90),
            "paired_current_p90_error_pct": percentile([row["absolute_error_pct"] for row in paired], 0.90),
            "paired_improved_count": sum(row["error_change_pp"] < -1e-9 for row in paired),
            "paired_worsened_count": sum(row["error_change_pp"] > 1e-9 for row in paired),
            "paired_unchanged_count": sum(abs(row["error_change_pp"]) <= 1e-9 for row in paired),
        })
    return result


def write_summary(rows, scope):
    groups = [("overall", "all", rows)]
    for field in ("recorded_implementation", "recorded_cost_class", "layout", "drop_adapter", "geometry_flag"):
        buckets = defaultdict(list)
        for row in rows:
            value = row[field] if field != "geometry_flag" else bool(row["host_geometry_disagreements"])
            buckets[str(value)].append(row)
        groups.extend((field, name, values) for name, values in sorted(buckets.items()))
    summaries = [summarize(*group) for group in groups]
    fields = list(dict.fromkeys(key for row in summaries for key in row))
    with (FOLDER / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    with (FOLDER / "summary.txt").open("w") as stream:
        stream.write("目前 clean 預測器對既有 %d 筆量測的完整重算\n" % len(rows))
        stream.write("固定現有係數，只呼叫 clean.predictor.predict(inputs)，不量測 NPU、不重新校準。\n")
        stream.write("這些是既有校準資料的對照結果，不是新 shape 的獨立泛化驗證。\n")
        stream.write("每筆輸入、實測、預測、誤差、舊版對照及失敗原因均在 errors.csv。\n")
        stream.write("絕對百分比誤差 = abs(預測 - 實測) / 實測 * 100；P90 使用線性插值。\n")
        stream.write("不刪除離群值；預測失敗另計，within10 比例的分母包含失敗輸入。\n")
        stream.write("Host 幾何標記保留，未將其解釋為誤差原因；模型階段貢獻不是獨立量測時間。\n\n")
        stream.write("route_matches_record 僅比較 implementation 名稱，不代表完整 Host tiling 相同。\n")
        stream.write("CPU 首輪限制 60 秒；只對超時案例補算一次、限制 180 秒，first_pass_error 保留首輪錯誤。\n\n")
        stream.write("分支 | 輸入 | 成功 | 失敗 | 中位誤差% | P90誤差% | 最大誤差% | <=10%比例 | >20%比例 | >30%比例\n")
        for row in summaries:
            if row["group_kind"] not in ("overall", "recorded_implementation"):
                continue
            if not row["predicted"]:
                stream.write("%s | %d | 0 | %d | 無預測結果\n" % (row["group"], row["inputs"], row["failed"]))
                continue
            stream.write("%s | %d | %d | %d | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f\n" %
                         (row["group"], row["inputs"], row["predicted"], row["failed"],
                          row["median_absolute_error_pct"], row["p90_absolute_error_pct"], row["max_absolute_error_pct"],
                          row["within10_pct_of_inputs"], row["above20_pct_of_predictions"], row["above30_pct_of_predictions"]))
        overall = summaries[0]
        stream.write("\n能對上舊版 clean 預測的相同輸入：%d 筆。\n" % overall["paired_previous_count"])
        if overall["paired_previous_count"]:
            stream.write("此配對子集的誤差中位數：舊 %.6f%% → 現在 %.6f%%；P90：舊 %.6f%% → 現在 %.6f%%。\n" %
                         (overall["paired_previous_median_error_pct"], overall["paired_current_median_error_pct"],
                          overall["paired_previous_p90_error_pct"], overall["paired_current_p90_error_pct"]))
            stream.write("改善 %d；退步 %d；相同 %d。其餘輸入沒有舊版記錄，不補造舊值。\n" %
                         (overall["paired_improved_count"], overall["paired_worsened_count"], overall["paired_unchanged_count"]))
    (FOLDER / "scope.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n")
    return summaries[0]


def main():
    model = json.loads((ROOT / "clean/model.json").read_text())
    with DATA.open(newline="") as stream:
        measurements = list(csv.DictReader(stream))
    previous = read_previous()
    scope = {
        "status": "running", "prediction_entry": "clean.predictor.predict",
        "model_file": "clean/model.json", "model_version": model["version"],
        "measurement_file": str(DATA.relative_to(ROOT)),
        "previous_prediction_file": str(PREVIOUS.relative_to(ROOT)),
        "expected_inputs": len(measurements), "npu_executed": False, "parameters_changed": False,
        "cpu_cores": 1, "memory_limit_mib": 1100, "first_pass_timeout_seconds": 60, "retry_timeout_seconds": 180,
        "evaluation_scope": "existing_calibration_measurements; not independent holdout",
    }
    (FOLDER / "scope.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n")
    rows = []
    fields = ["case", "measurement_source_line", *INPUT_FIELDS, "measured_us", "predicted_us",
              "signed_error_us", "signed_error_pct", "absolute_error_pct", "within10", "within20", "within30",
              "recorded_implementation", "predicted_implementation", "recorded_cost_class", "route_matches_record",
              "drop_adapter", "measurement_scope", "host_geometry_disagreements", "previous_clean_predicted_us",
              "previous_clean_absolute_error_pct", "error_change_pp", "previous_source_line",
              *("phase_" + phase + "_us" for phase in PHASES), "status", "error", "cpu_attempts", "first_pass_error"]
    handler = signal.signal(signal.SIGALRM, timeout)
    try:
        with (FOLDER / "errors.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            print("CPU RECALCULATION inputs=%d model=%s" % (len(measurements), model["version"]), flush=True)
            for number, measured in enumerate(measurements, 1):
                inputs = {key: measured[key] if key in ("dtype", "layout") else int(measured[key]) for key in INPUT_FIELDS}
                actual = float(measured["duration_us"])
                row = {key: "" for key in fields}
                row.update({"case": measured["case"], "measurement_source_line": number + 1, **inputs,
                            "measured_us": actual, "recorded_implementation": measured["cost_class"].split("|")[0],
                            "recorded_cost_class": measured["cost_class"], "measurement_scope": measured["measurement_scope"],
                            "host_geometry_disagreements": measured["host_geometry_disagreements"],
                            "drop_adapter": adapter_enabled(inputs), "cpu_attempts": 1})
                old = previous.get(identity(inputs))
                if old:
                    row.update({"previous_clean_predicted_us": old[0], "previous_source_line": old[1],
                                "previous_clean_absolute_error_pct": abs(old[0] - actual) / actual * 100})
                try:
                    if not math.isfinite(actual) or actual <= 0:
                        raise ValueError("invalid measured latency")
                    signal.alarm(60)
                    apply_prediction(row, predict(inputs))
                except Exception as error:
                    row.update({"status": "failed", "error": type(error).__name__ + ": " + str(error)})
                finally:
                    signal.alarm(0)
                    gc.collect()
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                if number % 25 == 0 or number == len(measurements):
                    print("PROGRESS %d/%d failures=%d case=%s" %
                          (number, len(measurements), sum(item["status"] != "ok" for item in rows), row["case"]), flush=True)
    finally:
        signal.signal(signal.SIGALRM, handler)
    retried = retry_timeouts(rows)
    save_rows(rows)
    scope.update({"status": "complete", "evaluated_inputs": len(rows),
                  "cpu_timeout_retried_cases": retried,
                  "failed_predictions": sum(row["status"] != "ok" for row in rows)})
    overall = write_summary(rows, scope)
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)
    print("PER_INPUT=" + str(FOLDER / "errors.csv"), flush=True)
    print("SUMMARY=" + str(FOLDER / "summary.txt"), flush=True)


if __name__ == "__main__":
    main()
