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
MODEL_FAMILIES = ("same_ab",)
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


PMU_RULES = {
    "aic_time_max_us": ("ArithmeticUtilization", "aic_time(us)", "max"),
    "aiv_time_max_us": ("ArithmeticUtilization", "aiv_time(us)", "max"),
    "aic_cube_ratio": ("ArithmeticUtilization", "aic_cube_ratio", "aic"),
    "aiv_vec_ratio": ("ArithmeticUtilization", "aiv_vec_ratio", "aiv"),
    "aic_mte1_ratio": ("PipeUtilization", "aic_mte1_ratio", "aic"),
    "aic_mte2_ratio": ("PipeUtilization", "aic_mte2_ratio", "aic"),
    "aic_mte3_ratio": ("PipeUtilization", "aic_mte3_ratio", "aic"),
    "aiv_mte2_ratio": ("PipeUtilization", "aiv_mte2_ratio", "aiv"),
    "aiv_mte3_ratio": ("PipeUtilization", "aiv_mte3_ratio", "aiv"),
    "aic_cube_wait_ratio": ("ResourceConflictRatio", "aic_cube_wait_ratio", "aic"),
    "aiv_vec_wait_ratio": ("ResourceConflictRatio", "aiv_vec_wait_ratio", "aiv"),
    "aic_mte2_wait_ratio": ("ResourceConflictRatio", "aic_mte2_wait_ratio", "aic"),
    "aiv_mte2_wait_ratio": ("ResourceConflictRatio", "aiv_mte2_wait_ratio", "aiv"),
    "aiv_vec_conflict_ratio": ("ResourceConflictRatio", "aiv_vec_total_cflt_ratio", "aiv"),
    "mainmem_read_kb": ("Memory", "read_main_memory_datas(KB)", "sum"),
    "mainmem_write_kb": ("Memory", "write_main_memory_datas(KB)", "sum"),
    "gm_to_l1_kb": ("Memory", "GM_to_L1_datas(KB)", "sum"),
    "l0c_to_gm_kb": ("Memory", "L0C_to_GM_datas(KB)", "sum"),
    "gm_to_ub_kb": ("Memory", "GM_to_UB_datas(KB)", "sum"),
    "ub_to_gm_kb": ("Memory", "UB_to_GM_datas(KB)", "sum"),
    "aic_l1_read_gbs": ("Memory", "aic_l1_read_bw(GB/s)", "aic"),
    "aic_l1_write_gbs": ("Memory", "aic_l1_write_bw(GB/s)", "aic"),
    "aiv_ub_read_gbs": ("MemoryUB", "aiv_ub_read_bw_vector(GB/s)", "aiv"),
    "aiv_ub_write_gbs": ("MemoryUB", "aiv_ub_write_bw_vector(GB/s)", "aiv"),
}
PMU_L2 = tuple(f"{engine}_l2_{direction}_hit_pct" for engine in ("aic", "aiv") for direction in ("read", "write"))
PMU_FIELDS = (*PMU_RULES, *PMU_L2)
REPORT_TILING = (
    "FA_S1_BASE", "FA_S2_BASE", "FA_N_RATIO", "FA_CORE_NUM", "FA_TOTAL_TASKS", "FA_SPLIT_FACTOR",
    "FA_BMM1_BASE_M", "FA_BMM1_BASE_N", "FA_BMM1_BASE_K", "FA_BMM2_BASE_M", "FA_BMM2_BASE_N", "FA_BMM2_BASE_K",
    "FA_BMM1_DEPTH_A1", "FA_BMM1_DEPTH_B1", "FA_BMM2_DEPTH_A1", "FA_BMM2_DEPTH_B1",
)


def profile_table(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = [[cell.strip() for cell in row] for row in csv.reader(source) if any(cell.strip() for cell in row)]
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("invalid profiler CSV dimensions")
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def finite(raw) -> float | None:
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def profile_counters(folder: Path, key: int) -> dict:
    log = folder / "profile.log"
    if not log.is_file():
        return {"notes": {"missing_log"}, "samples": {}, "launches": 0, "freq": []}
    text = log.read_text(errors="replace")
    pids = set(re.findall(r"^[ \t]*Pid:[ \t]*(\d+)", text, re.MULTILINE))
    notes = set()
    if "Profiling running finished. All task success." not in text or "Profiling data parse finished." not in text:
        notes.add("unsuccessful_profile")
    if len(pids) != 1:
        return {"notes": {"ambiguous_profile_pid"}, "samples": {}, "launches": 0, "freq": []}
    dates = [re.sub(r"\D", "", value) for value in re.findall(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", text)]
    if not dates:
        return {"notes": {"missing_profile_clock"}, "samples": {}, "launches": 0, "freq": []}
    lower, upper = min(dates) + "000", max(dates) + "999"
    pattern = re.compile(r"(?:\d+_)?([A-Za-z0-9]+)_(\d{17})\.csv$")
    expected = re.compile(rf"^FlashAttentionScore_[A-Za-z0-9]+_{key}(?:_|$)")
    files, basic, frequencies, boundaries = [], {}, [], set()
    for path in sorted(folder.glob("*.csv")):
        match = pattern.fullmatch(path.name)
        if not match or not lower <= match[2] <= upper:
            continue
        kind, stamp = match.groups()
        if kind != "OpBasicInfo":
            files.append((kind, stamp, path))
            continue
        try:
            records = profile_table(path)
        except (OSError, UnicodeError, ValueError, csv.Error):
            notes.add("invalid_basic_csv")
            continue
        boundaries.add(stamp)
        if any(row.get("Pid") and row["Pid"] not in pids for row in records):
            notes.add("overlapping_profile_sessions")
        records = [row for row in records if row.get("Pid") in pids and expected.match(row.get("Op Name", "")) and (finite(row.get("Task Duration(us)")) or 0) > 0]
        if len(records) == 1:
            basic[stamp] = records[0]
            frequency = finite(records[0].get("Current Freq"))
            if frequency is not None:
                frequencies.append(frequency)
    stamps = sorted(basic)
    if len(stamps) != REPEAT:
        notes.add("incomplete_launches")
    if "overlapping_profile_sessions" in notes:
        return {"notes": notes, "samples": {}, "launches": len(stamps), "freq": frequencies}
    required = {rule[0] for rule in PMU_RULES.values()} | {"L2Cache"}
    tables = {}
    for kind, stamp, path in files:
        if kind not in required:
            continue
        preceding = [start for start in boundaries if start <= stamp]
        if not preceding or max(preceding) not in basic:
            continue
        identity = (max(preceding), kind)
        try:
            records = profile_table(path)
            records = [row for row in records if row.get("block_id", "").isdigit() and row.get("sub_block_id", "").startswith(("cube", "vector"))]
            if not records:
                raise ValueError("no core rows")
            if identity in tables:
                if tables[identity] != records:
                    notes.add("duplicate_metric_table")
                    tables[identity] = None
                continue
            tables[identity] = records
        except (OSError, UnicodeError, ValueError, csv.Error):
            notes.add("invalid_" + kind)
            tables[identity] = None
    samples = {}
    for stamp in stamps:
        if any(not tables.get((stamp, kind)) for kind in required):
            notes.add("missing_metric_tables")
        for name, (kind, column, reduction) in PMU_RULES.items():
            records = tables.get((stamp, kind)) or []
            values = [(row, finite(row.get(column))) for row in records]
            values = [(row, value) for row, value in values if value is not None]
            if not values:
                notes.add("missing_metric_fields")
                continue
            if reduction == "sum":
                result = sum(value for _, value in values)
            elif reduction == "max":
                result = max(value for _, value in values)
            else:
                weighted = [(value, finite(row.get(reduction + "_time(us)"))) for row, value in values]
                weighted = [(value, weight) for value, weight in weighted if weight is not None and weight > 0]
                if not weighted:
                    notes.add("missing_metric_weights")
                    continue
                result = sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted)
            samples.setdefault(name, []).append(result)
        records = tables.get((stamp, "L2Cache")) or []
        for engine in ("aic", "aiv"):
            for direction in ("read", "write"):
                prefixes = [f"{engine}_r0_read", f"{engine}_r1_read"] if direction == "read" else [f"{engine}_write"]
                pairs = [(finite(row.get(prefix + "_cache_hit")), finite(row.get(prefix + "_cache_miss_allocate"))) for row in records for prefix in prefixes]
                pairs = [(hit, miss) for hit, miss in pairs if hit is not None and miss is not None]
                total = sum(hit + miss for hit, miss in pairs)
                if total > 0:
                    samples.setdefault(f"{engine}_l2_{direction}_hit_pct", []).append(100 * sum(hit for hit, _ in pairs) / total)
                else:
                    notes.add("missing_l2_requests")
    return {"notes": notes, "samples": samples, "launches": len(stamps), "freq": frequencies}


def text_report(directory: Path) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"existing dataset is missing: {directory}")
    hardware_path = directory / "hardware.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.is_file() else {}
    rows = recorded(directory / "routes.csv")
    cross_validation = {}
    calibration_path = directory / "calibration_report.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text())
        if calibration.get("model_source_sha256") == cost.sha256({"source": Path(cost.__file__).read_text()}):
            for group in calibration.get("calibration_grouped_cv", {}).values():
                for axis, values in group.get("axes", {}).items():
                    for case in values.get("cases", []):
                        cross_validation.setdefault(case["case"], {}).update({
                            f"cv_{axis}_ape_percent": case.get("ape_percent", ""), "calibration_cv_pass": group["pass"],
                        })
    hardware_fields = ("soc", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes")
    fields = (
        "split", "family", *cost.INPUT_FIELDS, *hardware_fields, "priority", "tiling_key",
        "duration_us", "round1_us", "round2_us", "round3_us", "sample_count",
        "ci95_low_us", "ci95_high_us", "stable", "checked_elements", "max_abs_error",
        "status", "stage", "predicted_us", "ape_percent", "pass_10_percent",
        "model_scope", "cv_sequence_pair_ape_percent", "cv_heads_ape_percent", "cv_batch_ape_percent", "calibration_cv_pass",
        *(name[3:].lower() for name in REPORT_TILING),
        "profile_rounds", "profile_launches", "freq_min_mhz", "freq_max_mhz", "pmu_status", "pmu_notes",
        *PMU_FIELDS,
    )
    target = directory / "diagnostics.csv"
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, row in sorted(rows.items()):
            result = {field: row.get(field, "") for field in fields}
            result.update(model_scope="|".join(MODEL_FAMILIES), **cross_validation.get(name, {}))
            result.update(row["inputs"])
            result.update({field: hardware.get(field, "") for field in hardware_fields})
            result.update(ci95_low_us=row.get("median_ci95_low_us", ""), ci95_high_us=row.get("median_ci95_high_us", ""))
            for index, value in enumerate(row.get("round_medians_us") or [], 1):
                result[f"round{index}_us"] = value
            correctness = row.get("correctness") or {}
            result.update(checked_elements=correctness.get("checked_elements", ""), max_abs_error=correctness.get("max_abs_error", ""))
            effective = row.get("effective_tiling") or {}
            if isinstance(effective, str):
                effective = json.loads(effective)
            result.update({field[3:].lower(): effective.get(field, "") for field in REPORT_TILING})
            grouped, frequencies, notes = {}, [], set()
            launches, rounds = 0, 0
            if row.get("tiling_key"):
                for round_id in range(1, ROUNDS + 1):
                    counters = profile_counters(directory / "cases" / name / f"round_{round_id}", int(row["tiling_key"]))
                    notes.update(counters["notes"])
                    launches += counters["launches"]
                    rounds += int(counters["launches"] == REPEAT)
                    frequencies.extend(counters["freq"])
                    for metric, values in counters["samples"].items():
                        grouped.setdefault(metric, []).extend(values)
            else:
                notes.add("no_tiling_key")
            if not hardware:
                notes.add("missing_hardware")
            result.update(
                profile_rounds=rounds, profile_launches=launches, pmu_status="partial" if notes else "ok",
                pmu_notes=";".join(sorted(notes)),
                freq_min_mhz=min(frequencies) if frequencies else "", freq_max_mhz=max(frequencies) if frequencies else "",
            )
            result.update({metric: statistics.median(values) for metric, values in grouped.items()})
            for field, value in result.items():
                parsed = finite(value) if isinstance(value, str) and value else None
                if parsed is not None and field not in ("tiling_key", *cost.INPUT_FIELDS[:7]):
                    result[field] = f"{parsed:.6g}"
                elif isinstance(value, float):
                    result[field] = f"{value:.6g}"
            writer.writerow({field: result.get(field, "") for field in fields})
    temporary.replace(target)
    legacy = directory / "diagnostics.txt"
    if legacy.is_file():
        with legacy.open(encoding="utf-8", errors="replace") as source:
            prefix = source.read(80)
        if prefix.startswith(("FA_LATENCY_DIAGNOSTICS", '{"report":"sameab_compact_v2"')):
            legacy.unlink()
    print(f"CSV_REPORT={target}")
    print(f"SHAPES={len(rows)} BYTES={target.stat().st_size}")


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
    for family in sorted({case["family"] for case in plan}):
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
        "evaluated_families": sorted({case["family"] for case in plan}), "evaluated_groups": sorted(frozen["groups"]),
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
    full_plan = cost.design(hardware)
    design_check = cost.validate_design(full_plan, hardware)
    protocol = {
        "version": VERSION, "hardware": hardware, "plan": full_plan, "rounds": ROUNDS, "repeat": REPEAT,
        "target": "msprof OpBasicInfo.csv Task Duration(us)", "error_limit": cost.RELATIVE_ERROR_LIMIT,
        "launcher_warmup": WARMUP, "profiler_warmup": 10,
        "numerical_rtol": 0.01, "numerical_atol": 1e-4, "stability_limit": 0.03,
        "input_pattern": "uint32 hash with independent Q/K/V salts; FP16-representable values in [-1,1)",
        "cost_source_sha256": cost.sha256({"source": Path(cost.__file__).read_text()}),
        "launcher_sha256": cost.sha256({"source": Path("fa_launcher.cpp").read_text()}),
    }
    signature = cost.sha256({"protocol": protocol, "model_families": MODEL_FAMILIES})
    manifest = directory / "manifest.json"
    measurement_protocol = {name: value for name, value in protocol.items() if name not in ("cost_source_sha256", "plan")}
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        saved = {name: value for name, value in previous.items() if name not in ("cost_source_sha256", "plan")}
        if cost.sha256(saved) != cost.sha256(measurement_protocol):
            raise RuntimeError("the existing input-latency dataset uses a different measurement protocol")
        old_cases = {case["name"]: case for case in previous["plan"]}
        new_cases = {case["name"]: case for case in full_plan}
        if any(name not in new_cases or cost.sha256(case) != cost.sha256(new_cases[name]) for name, case in old_cases.items()):
            raise RuntimeError("an existing measurement shape or split has changed")
        if any(case["split"] != "calibration" for name, case in new_cases.items() if name not in old_cases):
            raise RuntimeError("an existing dataset can only be extended with calibration inputs")
    if not manifest.is_file():
        cost.save_json(manifest, protocol)
    plan = [case for case in full_plan if case["family"] in MODEL_FAMILIES]
    print(f"MODEL_FAMILIES={','.join(MODEL_FAMILIES)}", flush=True)
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
    selected_names = {case["name"] for case in plan}
    diagnostic = cost.fit([row for name, row in records.items() if name in selected_names], hardware, signature) if frozen is None else frozen
    expected_groups = {cost.group_key(case["inputs"], hardware) for case in plan if case["split"] == "calibration"}
    grouped_cv = diagnostic["calibration_grouped_cv"]
    calibration_model_pass = set(grouped_cv) == expected_groups and all(value["pass"] for value in grouped_cv.values())
    cost.save_json(directory / "calibration_report.json", {
        "version": VERSION, "fit_method": cost.FIT_METHOD, "pass": calibration_model_pass and not incomplete,
        "model_families": list(MODEL_FAMILIES),
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
        text_report(directory)
        raise SystemExit(1)
    if not calibration_model_pass:
        cost.save_json(directory / "validation_report.json", {"pass": False, "stage": "calibration_model_failed", "calibration_grouped_cv": grouped_cv})
        print("ACCURACY_GATE=FAIL calibration_model_failed; blind measurements were not started", flush=True)
        text_report(directory)
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
    text_report(directory)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
