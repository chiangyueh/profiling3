from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

import main as fa
import fa_latency_inputs as cost


VERSION = cost.VERSION
WARMUP = 5
REPEAT = 10
ROUNDS = 3
FIELDS = (
    "version", "case", "split", "family", "inputs", "priority", "tiling_key",
    "effective_tiling", "derived_schedule", "duration_us", "samples_us",
    "round_medians_us", "sample_count", "median_ci95_low_us", "median_ci95_high_us",
    "stable", "correctness", "model_sha256", "prediction_before_measurement",
    "predicted_us", "ape_percent", "pass_10_percent", "status", "stage", "error",
)


def append(path: Path, row: dict) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: json.dumps(row[name], sort_keys=True) if isinstance(row.get(name), (dict, list)) else row.get(name, "") for name in FIELDS})


def recorded(path: Path) -> dict:
    if not path.is_file():
        return {}
    rows = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            for key in ("inputs", "samples_us", "derived_schedule", "correctness", "round_medians_us"):
                if row.get(key):
                    row[key] = json.loads(row[key])
            rows[row["case"]] = row
    return rows


def metric_role(label: str) -> str | None:
    name = re.sub(r"[^a-z0-9]", "", label.lower())
    if "l2" in name and "hit" in name and any(word in name for word in ("rate", "ratio", "percent")):
        return "l2_hit"
    bandwidth = any(word in name for word in ("bandwidth", "throughput", "bw"))
    if "l2" in name and bandwidth:
        return "l2_bw"
    if any(word in name for word in ("cube", "mac")) and any(word in name for word in ("time", "ratio", "util", "usage")):
        return "cube"
    if "mte2" in name:
        return "aiv_mte2" if any(word in name for word in ("aiv", "vector")) else "aic_mte2"
    if any(word in name for word in ("vec", "vector")) and any(word in name for word in ("time", "ratio", "util", "usage")):
        return "vector"
    if bandwidth and any(word in name for word in ("memory", "hbm", "ddr", "gm")) and not any(word in name for word in ("l0", "ub")):
        return "memory_write" if "write" in name else "memory_read"
    return None


def metric_values(path: Path) -> tuple[dict, list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = [[cell.strip() for cell in row] for row in csv.reader(source) if any(cell.strip() for cell in row)]
    if not rows:
        return {}, []
    table = re.sub(r"_\d+$", "", re.sub(r"^\d+_", "", path.stem))
    result = {}

    def add(label: str, raw: str) -> None:
        if metric_role(label) is None:
            return
        try:
            value = float(raw.rstrip("%"))
        except ValueError:
            return
        if raw.endswith("%") and "%" not in label:
            label += " (%)"
        if math.isfinite(value):
            result.setdefault(label, []).append(value)

    vertical = rows[0][0].lower().replace(" ", "") in ("metric", "metricname", "name")
    if vertical:
        header = [cell.lower() for cell in rows[0]]
        value_index = next((i for i, cell in enumerate(header) if cell in ("value", "metric value", "metricvalue")), 1)
        unit_index = next((i for i, cell in enumerate(header) if cell == "unit"), None)
        for row in rows[1:]:
            if value_index >= len(row):
                continue
            unit = f" ({row[unit_index]})" if unit_index is not None and unit_index < len(row) else ""
            add(f"{table}:{row[0]}{unit}", row[value_index])
    else:
        for row in rows[1:]:
            context = " ".join(value for field, value in zip(rows[0], row) if "type" in field.lower() and any(word in re.sub(r"[^a-z]", "", value.lower()) for word in ("aic", "aiv", "vector")))
            for field, value in zip(rows[0], row):
                add(f"{table}:{context}:{field}" if context else f"{table}:{field}", value)
    return result, [table + ":" + field for field in rows[0]]


def text_report(directory: Path) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"existing dataset is missing: {directory}")
    hardware_path = directory / "hardware.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.is_file() else {}
    rows = recorded(directory / "routes.csv")
    cases = [row for _, row in sorted(rows.items()) if row.get("split") == "calibration" and row.get("family") == "same_ab"]
    values, frequencies, issues, headers = [], [], Counter(), set()
    for row in cases:
        grouped, freq = {}, []
        for round_id in range(1, ROUNDS + 1):
            folder = directory / "cases" / row["case"] / f"round_{round_id}"
            log = folder / "profile.log"
            if log.is_file():
                text = log.read_text(errors="replace")
                freq.extend(float(value) for value in re.findall(r"^[ \t]*Current Freq:[ \t]*(\d+(?:\.\d+)?)", text, re.MULTILINE))
            else:
                issues["missing_profile_log"] += 1
            tables = list(folder.glob("*.csv"))
            if not tables:
                issues["missing_csv_rounds"] += 1
            for table in tables:
                try:
                    extracted, fields = metric_values(table)
                except (OSError, UnicodeError, csv.Error):
                    issues["unreadable_csv"] += 1
                    continue
                headers.update(field for field in fields if any(word in field.lower() for word in ("l2", "memory", "pipe", "arithmetic")))
                for label, readings in extracted.items():
                    grouped.setdefault(label, []).extend(readings)
        values.append(grouped)
        frequencies.append(freq)
    selected = {}
    roles = ("cube", "vector", "aic_mte2", "aiv_mte2", "l2_hit", "l2_bw", "memory_read", "memory_write")
    labels = {label for grouped in values for label in grouped}
    for role in roles:
        candidates = [label for label in labels if metric_role(label) == role]
        if candidates:
            def preference(label: str) -> tuple:
                name = re.sub(r"[^a-z]", "", label.lower())
                wrong_core = (role in ("cube", "aic_mte2") and "aiv" in name) or (role in ("vector", "aiv_mte2") and "aicore" in name and "aiv" not in name)
                return (-sum(label in grouped for grouped in values), wrong_core, "write" in label.lower(), not any(word in label.lower() for word in ("ratio", "utilization", "rate", "usage")), label)
            selected[role] = min(candidates, key=preference)
    shared = {name: cases[0]["inputs"][name] for name in cost.INPUT_FIELDS if cases and all(row["inputs"][name] == cases[0]["inputs"][name] for row in cases)}
    varying = [name for name in cost.INPUT_FIELDS if name not in shared]
    for left, right in (("N2", "N1"), ("S2", "S1")):
        if cases and left in varying and all(row["inputs"][left] == row["inputs"][right] for row in cases):
            shared[left] = "=" + right
            varying.remove(left)
    metadata = {
        "report": "sameab_compact_v2", "hardware": {name: hardware.get(name) for name in ("soc", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes")},
        "shared_inputs": shared, "metrics": selected, "missing_metrics": [role for role in roles if role not in selected],
        "issues": dict(issues), "case_count": len(cases), "rounds_per_case": ROUNDS, "missing_value": "-", "metric_significant_digits": 4,
    }
    if not hardware:
        metadata["issues"]["missing_hardware_json"] = 1
    if len(selected) != len(roles):
        metadata["available_headers"] = sorted(headers, key=lambda field: (metric_role(field) is not None, field))[:8]

    def render(ranges: bool) -> str:
        metadata["metric_statistics"] = "median/min/max across existing CSV rows" if ranges else "median across existing CSV rows"
        lines = [json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)]
        lines.append(",".join([*varying, "time_us", "stable", "freq_min/max", *selected]))
        for row, grouped, freq in zip(cases, values, frequencies):
            cells = [str(row["inputs"][field]) for field in varying]
            cells += [f"{float(row['duration_us']):.5g}" if row.get("duration_us") else "-", str(row.get("stable")).lower(), f"{min(freq):.4g}/{max(freq):.4g}" if freq else "-"]
            for label in selected.values():
                readings = grouped.get(label, [])
                stats = [statistics.median(readings), min(readings), max(readings)] if readings and ranges else ([statistics.median(readings)] if readings else [])
                cells.append("/".join(f"{value:.4g}" for value in stats) if stats else "-")
            lines.append(",".join(cells))
        return "\n".join(lines) + "\n"

    content = render(True)
    if len(content.encode("utf-8")) > 8192:
        content = render(False)
    if len(content.encode("utf-8")) > 8192:
        raise RuntimeError("compact report exceeds 8 KiB; existing report was not overwritten")
    target = directory / "diagnostics.txt"
    target.write_text(content, encoding="utf-8")
    print(f"TEXT_REPORT={target}")
    print(f"SAMEAB_CASES={len(cases)} METRICS={len(selected)} BYTES={target.stat().st_size} MAX_BYTES=8192")


def select_shape(case: dict) -> None:
    p = case["inputs"]
    fa.SHAPE = {
        **{key: p[key] for key in cost.INPUT_FIELDS[:7]},
        "DTYPE": p["dtype"], "LAYOUT": p["layout"], "HAS_DROP": p["has_drop_mask"],
    }


def discover(params, directory: Path, hardware_trace: Path | None = None) -> tuple[int, int]:
    fa.build_host()
    trace = directory / "host_route.tsv"
    trace.unlink(missing_ok=True)
    env = fa.run_env(params, None)
    env.update(FA_ROUTE_TRACE=str(trace), FA_DISCOVER_ONLY="1", FA_PROFILE="0", FA_SKIP_BUILD="1")
    if hardware_trace is not None:
        env["FA_HARDWARE_TRACE"] = str(hardware_trace)
    process = subprocess.run(["bash", "run.sh"], env=env, capture_output=True, text=True)
    (directory / "host.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"Host discovery failed; see {directory / 'host.log'}")
    return fa.read_route(trace)


def hardware_profile(directory: Path, npu_id: int) -> dict:
    saved = directory / "hardware.json"
    if saved.is_file():
        hardware = json.loads(saved.read_text())
        if hardware["physical_npu_id"] != npu_id:
            raise RuntimeError("this dataset belongs to a different physical NPU")
        return hardware
    trace = directory / "hardware_runtime.json"
    seed = {"inputs": dict(zip(cost.INPUT_FIELDS, (1, 1, 1, 16, 16, 64, 64, "FP16", "BNSD", 0)))}
    select_shape(seed)
    discover(fa.shape_params(), directory, trace)
    hardware = json.loads(trace.read_text())
    sdk = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")).resolve()
    name = hardware["soc"] + ".ini"
    candidates = [sdk / "data/platform_config" / name, sdk / f"{platform.machine()}-linux/data/platform_config" / name]
    candidates.extend(sdk.glob(f"*/data/platform_config/{name}"))
    ini = next((path for path in candidates if path.is_file()), None)
    if ini is None:
        raise RuntimeError(f"platform specification is missing for {hardware['soc']} under {sdk}")
    spec = configparser.ConfigParser()
    spec.read(ini)
    hardware.update(
        physical_npu_id=npu_id,
        l1_bytes=spec.getint("AICoreSpec", "l1_size"),
        l0c_bytes=spec.getint("AICoreSpec", "l0_c_size"),
        ub_bytes=spec.getint("AICoreSpec", "ub_size"),
        l2_bytes=spec.getint("SoCInfo", "l2_size"),
        platform_spec=dict(spec["AICoreSpec"]),
        platform_rates=dict(spec["AICoreMemoryRates"]),
        platform_spec_sha256=cost.sha256(dict((section, dict(spec[section])) for section in spec.sections())),
        cann_root=str(sdk),
    )
    cost.save_json(saved, hardware)
    return hardware


def quantized(indices: np.ndarray, base: float, dtype: str) -> np.ndarray:
    salt = 0x12345678 if base < 0.015 else (0x87654321 if base < 0.025 else 0x31415926)
    hashed = indices.astype(np.uint32) ^ np.uint32(salt)
    hashed ^= hashed >> 16
    hashed *= np.uint32(0x7feb352d)
    hashed ^= hashed >> 15
    hashed *= np.uint32(0x846ca68b)
    hashed ^= hashed >> 16
    values = ((hashed & 2047).astype(np.int32) - 1024).astype(np.float32) / np.float32(1024.0)
    if dtype == "FP16":
        return values.astype(np.float16).astype(np.float64)
    if dtype == "BF16":
        bits = values.view(np.uint32)
        rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint32)
        return (rounded << 16).view(np.float32).astype(np.float64)
    return values.astype(np.float64)


def tensor_indices(p: dict, b: int, h: int, rows: np.ndarray, heads: int, sequence: int, width: int) -> np.ndarray:
    columns = np.arange(width, dtype=np.int64)
    if p["layout"] == "BNSD":
        return ((b * heads + h) * sequence + rows[:, None]) * width + columns
    return ((rows[:, None] * p["B"] + b) * heads + h) * width + columns


def check_output(inputs: dict) -> dict:
    p = cost.normalize(inputs)
    path = Path("output/output.bin")
    elements = p["B"] * p["N1"] * p["S1"] * p["DV"]
    if not path.is_file() or path.stat().st_size != elements * 4:
        raise RuntimeError("output is missing or has an incorrect size")
    output = np.memmap(path, dtype=np.float32, mode="r")
    checked = 0
    max_error = 0.0
    key_positions = np.arange(p["S2"], dtype=np.int64)
    for batch in sorted({0, p["B"] - 1}):
        for head in sorted({0, p["N1"] - 1}):
            kv_head = head // (p["N1"] // p["N2"])
            keys = quantized(tensor_indices(p, batch, kv_head, key_positions, p["N2"], p["S2"], p["D"]), 0.02, p["dtype"])
            values = quantized(tensor_indices(p, batch, kv_head, key_positions, p["N2"], p["S2"], p["DV"]), 0.03, p["dtype"])
            for query in sorted({0, p["S1"] // 2, p["S1"] - 1}):
                q = quantized(tensor_indices(p, batch, head, np.array([query]), p["N1"], p["S1"], p["D"]), 0.01, p["dtype"])[0]
                score = keys @ q / math.sqrt(p["D"])
                weights = np.exp(score - np.max(score))
                weights /= np.sum(weights)
                if p["has_drop_mask"]:
                    bit_indices = ((batch * p["N1"] + head) * p["S1"] + query) * p["S2"] + key_positions
                    packed = np.where((bit_indices // 8) % 2 == 0, 0xAA, 0xD5)
                    mask = (packed >> (bit_indices % 8)) & 1
                    weights *= mask / 0.9
                golden = weights @ values
                indices = tensor_indices(p, batch, head, np.array([query]), p["N1"], p["S1"], p["DV"])[0]
                actual = np.asarray(output[indices], dtype=np.float64)
                if not np.all(np.isfinite(actual)) or not np.allclose(actual, golden, rtol=0.01, atol=1e-4):
                    raise RuntimeError(f"independent numerical spot-check failed: b={batch} h={head} q={query}")
                max_error = max(max_error, float(np.max(np.abs(actual - golden))))
                checked += p["DV"]
    return {"method": "independent FP64 attention over all K/V positions for sampled query rows", "checked_elements": checked, "max_abs_error": max_error}


def profiler_samples(path: Path, key: int) -> list[float]:
    if not path.is_file():
        raise RuntimeError(f"profiler log is missing: {path}")
    content = path.read_text(errors="replace")
    if "Profiling running finished. All task success." not in content or "Profiling data parse finished." not in content:
        raise RuntimeError(f"profiler did not report successful collection and parsing: {path}")
    expected = re.compile(rf"^FlashAttentionScore_[A-Za-z0-9]+_{key}(?:_|$)")
    times = []
    operator = None
    in_basic_info = False
    for line in content.splitlines():
        if "Operator Basic Information:" in line:
            in_basic_info = True
            operator = None
            continue
        if not in_basic_info:
            continue
        name = re.fullmatch(r"\s*Op Name:\s*(\S+)\s*", line)
        if name:
            operator = name.group(1)
            continue
        duration = re.fullmatch(r"\s*Task Duration\(us\):\s*(\S+)\s*", line)
        if duration:
            if operator is not None and expected.match(operator):
                value = float(duration.group(1))
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"invalid profiler duration: {path}")
                times.append(value)
            in_basic_info = False
    if len(times) != REPEAT:
        raise RuntimeError(f"expected {REPEAT} profiler records for key {key}, got {len(times)} in {path}")
    return times


def recover_profile(case: dict, params, key: int, priority: int, hardware: dict, summary: Path) -> dict | None:
    log = summary / "profile.log"
    trace = summary / "effective.tsv"
    if not log.is_file() or not trace.is_file():
        return None
    try:
        times = profiler_samples(log, key)
    except (RuntimeError, ValueError):
        return None
    effective = fa.read_effective(trace)
    if effective is None:
        return None
    cost.verify_schedule(case["inputs"], hardware, priority, effective)
    Path("output/output.bin").unlink(missing_ok=True)
    env = fa.run_env(params, key)
    env.update(FA_SKIP_BUILD="1", FA_PROFILE="0", FA_WARMUP="0", FA_REPEAT="1", FA_INPUT_PATTERN="1")
    env.pop("FA_LATENCY_TRACE", None)
    env.pop("FA_EFFECTIVE_TILING_TRACE", None)
    with (summary / "recovery_check.log").open("w") as output:
        process = subprocess.run(["bash", "run.sh"], env=env, stdout=output, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"saved profiler data found, but output verification failed; see {summary / 'recovery_check.log'}")
    correctness = check_output(case["inputs"])
    return {
        "samples_us": times, "effective_tiling": effective, "correctness": correctness,
        "timing_source": "saved msprof Operator Basic Information Task Duration(us)",
        "profiler_log_sha256": cost.sha256({"log": log.read_text(errors="replace")}),
        "numerical_check": "fresh untimed execution of the same operator inputs",
    }


def profile_measure(case: dict, params, key: int, priority: int, hardware: dict, directory: Path) -> dict:
    samples = []
    medians = []
    correctness = None
    effective = None
    unstable_marker = directory / "unstable.json"
    allow_recovery = not unstable_marker.is_file()
    for round_id in range(ROUNDS):
        summary = directory / f"round_{round_id + 1}"
        summary.mkdir(parents=True, exist_ok=True)
        checkpoint = summary / "measurement.json"
        if checkpoint.is_file():
            cached = json.loads(checkpoint.read_text())
            samples.extend(cached["samples_us"])
            medians.append(statistics.median(cached["samples_us"]))
            effective = cached["effective_tiling"]
            correctness = cached["correctness"]
            print(f"MEASURE CACHE HIT: case={case['name']} round={round_id + 1}/{ROUNDS}", flush=True)
            continue
        recovered = recover_profile(case, params, key, priority, hardware, summary) if allow_recovery else None
        if recovered is not None:
            cost.save_json(checkpoint, recovered)
            samples.extend(recovered["samples_us"])
            medians.append(statistics.median(recovered["samples_us"]))
            effective = recovered["effective_tiling"]
            correctness = recovered["correctness"]
            print(f"MEASURE RECOVERED: case={case['name']} round={round_id + 1}/{ROUNDS}", flush=True)
            continue
        scratch = Path(tempfile.mkdtemp(prefix="profile_", dir=directory))
        trace = summary / "effective.tsv"
        trace.unlink(missing_ok=True)
        Path("output/output.bin").unlink(missing_ok=True)
        env = fa.run_env(params, key)
        env.update(FA_SKIP_BUILD="1", FA_PROFILE="1", FA_PROFILE_OUTPUT=str(scratch), FA_EFFECTIVE_TILING_TRACE=str(trace), FA_INPUT_PATTERN="1")
        print(f"MEASURE case={case['name']} round={round_id + 1}/{ROUNDS} log={summary / 'profile.log'}", flush=True)
        try:
            with (summary / "profile.log").open("w") as log:
                process = subprocess.run(["bash", "run.sh"], env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            if process.returncode:
                raise RuntimeError(f"NPU/profile failed; see {summary / 'profile.log'}")
            times = profiler_samples(summary / "profile.log", key)
            effective = fa.read_effective(trace)
            if effective is None:
                raise RuntimeError("effective Host tiling trace was not produced")
            cost.verify_schedule(case["inputs"], hardware, priority, effective)
            correctness = check_output(case["inputs"])
            samples.extend(times)
            medians.append(statistics.median(times))
            cost.save_json(checkpoint, {"samples_us": times, "effective_tiling": effective, "correctness": correctness})
        finally:
            for index, table in enumerate(scratch.rglob("*.csv")):
                shutil.copy2(table, summary / f"{index}_{table.name}")
            shutil.rmtree(scratch)
    duration = statistics.median(samples)
    rng = np.random.default_rng(1729)
    bootstrap = np.median(rng.choice(samples, size=(2000, len(samples))), axis=1)
    low, high = (float(x) for x in np.percentile(bootstrap, [2.5, 97.5]))
    stable = max(abs(value / duration - 1) for value in medians) <= 0.03 and (high - low) / (2 * duration) <= 0.03
    if not stable:
        cost.save_json(unstable_marker, {"round_medians_us": medians, "median_ci95_low_us": low, "median_ci95_high_us": high})
        for round_id in range(ROUNDS):
            (directory / f"round_{round_id + 1}" / "measurement.json").unlink(missing_ok=True)
    else:
        unstable_marker.unlink(missing_ok=True)
    return {
        "effective_tiling": effective, "duration_us": duration, "samples_us": samples,
        "round_medians_us": medians, "sample_count": len(samples),
        "median_ci95_low_us": low, "median_ci95_high_us": high,
        "stable": stable, "correctness": correctness,
    }


def is_complete(row: dict | None) -> bool:
    return bool(row and row.get("status") == "ok" and str(row.get("stable")).lower() == "true")


def collect(case: dict, hardware: dict, directory: Path, records: dict, csv_path: Path, ensured: set, frozen: dict | None = None, precomputed: dict | None = None) -> None:
    name = case["name"]
    if is_complete(records.get(name)):
        print(f"SKIP case={name} status=ok", flush=True)
        return
    row = {
        "version": VERSION, "case": name, "split": case["split"], "family": case["family"],
        "inputs": case["inputs"], "derived_schedule": cost.derive(case["inputs"], hardware),
        "prediction_before_measurement": bool(frozen),
    }
    stage = "predict" if frozen else "discover"
    folder = directory / "cases" / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"CASE split={case['split']} START: {name}", flush=True)
    try:
        if frozen is not None:
            row.update(predicted_us=precomputed["predicted_us"], model_sha256=cost.sha256(frozen))
            print(f"PREDICTION BEFORE NPU: {name} predicted_us={row['predicted_us']:.6f}", flush=True)
        select_shape(case)
        params = fa.shape_params()
        stage = "discover"
        priority, key = discover(params, folder)
        row.update(priority=priority, tiling_key=key)
        if priority != case["expected"]:
            raise RuntimeError(f"official route={priority}, input-derived route={case['expected']}")
        stage = "build"
        identity = (key, case["inputs"]["dtype"])
        if identity not in ensured:
            fa.build(key)
            ensured.add(identity)
        stage = "measure"
        row.update(profile_measure(case, params, key, priority, hardware, folder))
        if frozen is not None:
            error = abs(row["predicted_us"] / row["duration_us"] - 1)
            row.update(ape_percent=error * 100, pass_10_percent=error <= cost.RELATIVE_ERROR_LIMIT + 1e-12 and row["stable"])
        row.update(status="ok", stage="complete")
        print(f"CASE OK: {name} actual_us={row['duration_us']:.6f} stable={row['stable']} error_percent={row.get('ape_percent', '')} pass_10_percent={row.get('pass_10_percent', '')}", flush=True)
    except Exception as error:
        row.update(status="failed", stage=stage, error=str(error))
        print(f"CASE FAILED: {name} stage={stage} error={error}", flush=True)
    append(csv_path, row)
    records[name] = row


def validation_report(plan: list[dict], records: dict, frozen: dict, directory: Path) -> dict:
    families = {}
    passed = True
    for family in ("same_ab", "s1", "b", "b_drop"):
        cases = [case for case in plan if case["family"] == family and case["split"] == "holdout"]
        results = []
        for case in cases:
            row = records.get(case["name"], {})
            success = is_complete(row) and str(row.get("pass_10_percent")).lower() == "true"
            passed &= success
            results.append({
                "case": case["name"], "pass": success, "actual_us": row.get("duration_us"),
                "predicted_us": row.get("predicted_us"), "ape_percent": row.get("ape_percent"),
                "stable": row.get("stable"), "error": row.get("error", ""),
            })
        errors = [float(row["ape_percent"]) for row in results if row["ape_percent"] not in ("", None)]
        families[family] = {
            "required_shapes": len(cases), "passed_shapes": sum(row["pass"] for row in results),
            "mape_percent": statistics.mean(errors) if errors else None,
            "max_ape_percent": max(errors) if errors else None, "cases": results,
        }
        print(f"VALIDATION {family} passed={families[family]['passed_shapes']}/{len(cases)} max_ape_percent={families[family]['max_ape_percent']}", flush=True)
    calibration_ok = all(is_complete(records.get(case["name"])) for case in plan if case["split"] == "calibration")
    report = {
        "version": VERSION, "pass": bool(passed and calibration_ok),
        "criterion": "every required blind shape must pass; absolute relative error <= 10 percent; missing, failed or unstable measurements fail",
        "model_sha256": cost.sha256(frozen), "calibration_complete": calibration_ok,
        "families": families,
        "calibration_grouped_cv": frozen.get("calibration_grouped_cv"),
    }
    cv = frozen.get("calibration_grouped_cv", {})
    report["pass"] = report["pass"] and bool(cv) and all(item["pass"] for item in cv.values())
    cost.save_json(directory / "validation_report.json", report)
    print(f"ACCURACY_GATE={'PASS' if report['pass'] else 'FAIL'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    if args.report_only:
        text_report(root / "output" / VERSION)
        return
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(allowed[:2]))
    os.environ["FA_BUILD_JOBS"] = "1"
    fa.NPU_ID = args.id
    fa.PROFILE_WARMUP = WARMUP
    fa.PROFILE_REPEAT = REPEAT
    directory = root / "output" / VERSION
    directory.mkdir(parents=True, exist_ok=True)
    hardware = hardware_profile(directory, args.id)
    plan = cost.design(hardware)
    design_check = cost.validate_design(plan, hardware)
    protocol = {
        "version": VERSION, "hardware": hardware, "plan": plan, "rounds": ROUNDS, "repeat": REPEAT,
        "target": "msprof OpBasicInfo.csv Task Duration(us)", "error_limit": cost.RELATIVE_ERROR_LIMIT,
        "launcher_warmup": WARMUP, "profiler_warmup": 10,
        "numerical_rtol": 0.01, "numerical_atol": 1e-4, "stability_limit": 0.03,
        "input_pattern": "uint32 hash with independent Q/K/V salts; FP16-representable values in [-1,1)",
        "cost_source_sha256": cost.sha256({"source": Path(cost.__file__).read_text()}),
        "launcher_sha256": cost.sha256({"source": Path("fa_launcher.cpp").read_text()}),
    }
    signature = cost.sha256(protocol)
    manifest = directory / "manifest.json"
    measurement_protocol = {name: value for name, value in protocol.items() if name != "cost_source_sha256"}
    if manifest.is_file():
        saved = {name: value for name, value in json.loads(manifest.read_text()).items() if name != "cost_source_sha256"}
        if cost.sha256(saved) != cost.sha256(measurement_protocol):
            raise RuntimeError("the existing input-latency dataset uses a different measurement protocol")
    if not manifest.is_file():
        cost.save_json(manifest, protocol)
    print(f"PLAN shapes={len(plan)} counts={dict(Counter((case['split'], case['family']) for case in plan))}", flush=True)
    print(f"DESIGN_CHECK={design_check}", flush=True)
    print(f"DATASET={directory}", flush=True)
    csv_path = directory / "routes.csv"
    records = recorded(csv_path)
    model_path = directory / "model.json"
    frozen = json.loads(model_path.read_text()) if model_path.is_file() else None
    if frozen is not None and (frozen.get("plan_sha256") != signature or frozen.get("fit_method") != cost.FIT_METHOD):
        raise RuntimeError("the existing frozen model uses different model code; measurements are preserved, but its blind set must not be reused to validate a replacement")
    ensured = set()
    for case in plan:
        if case["split"] == "calibration":
            collect(case, hardware, directory, records, csv_path, ensured)
    incomplete = [case["name"] for case in plan if case["split"] == "calibration" and not is_complete(records.get(case["name"]))]
    diagnostic = cost.fit(list(records.values()), hardware, signature) if frozen is None else frozen
    expected_groups = {cost.group_key(case["inputs"], hardware) for case in plan if case["split"] == "calibration"}
    grouped_cv = diagnostic["calibration_grouped_cv"]
    calibration_model_pass = set(grouped_cv) == expected_groups and all(value["pass"] for value in grouped_cv.values())
    cost.save_json(directory / "calibration_report.json", {
        "version": VERSION, "fit_method": cost.FIT_METHOD, "pass": calibration_model_pass and not incomplete,
        "calibration_complete": not incomplete, "incomplete_cases": incomplete,
        "model_source_sha256": protocol["cost_source_sha256"],
        "evaluation": "calibration-only grouped cross-validation; not independent blind validation",
        "calibration_grouped_cv": grouped_cv,
    })
    for group, value in grouped_cv.items():
        print(f"CALIBRATION CV: group={group} pass={value['pass']} max_ape_percent={value['max_ape_percent']}", flush=True)
    if incomplete:
        cost.save_json(directory / "validation_report.json", {"pass": False, "stage": "calibration_incomplete", "cases": incomplete, "calibration_grouped_cv": grouped_cv})
        print(f"ACCURACY_GATE=FAIL calibration_incomplete={len(incomplete)}; successful cases are reusable", flush=True)
        raise SystemExit(1)
    if not calibration_model_pass:
        cost.save_json(directory / "validation_report.json", {"pass": False, "stage": "calibration_model_failed", "calibration_grouped_cv": grouped_cv})
        print("ACCURACY_GATE=FAIL calibration_model_failed; blind measurements were not started", flush=True)
        raise SystemExit(1)
    if frozen is None:
        frozen = diagnostic
        cost.save_json(model_path, frozen)
    predictions_path = directory / "predictions_before_measurement.json"
    if predictions_path.is_file():
        payload = json.loads(predictions_path.read_text())
        if payload["model_sha256"] != cost.sha256(frozen):
            raise RuntimeError("saved predictions belong to a different frozen model")
    else:
        payload = {
            "model_sha256": cost.sha256(frozen),
            "predictions": {case["name"]: cost.predict(case["inputs"], frozen) for case in plan if case["split"] == "holdout"},
        }
        cost.save_json(predictions_path, payload)
    for case in plan:
        if case["split"] == "holdout":
            collect(case, hardware, directory, records, csv_path, ensured, frozen, payload["predictions"][case["name"]])
    report = validation_report(plan, records, frozen, directory)
    print(f"MODEL={model_path}")
    print(f"REPORT={directory / 'validation_report.json'}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
