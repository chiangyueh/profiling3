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
import signal
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

import main as fa
import fa_latency_inputs as cost
import fit_fa_latency as family_models


VERSION = cost.VERSION
WARMUP = 5
REPEAT = 10
ROUNDS = 3
MODEL_FAMILIES = ("same_ab",)
RESEARCH_FAMILIES = ("s1", "b", "b_drop")
ALL_FAMILIES = ("same_ab", "s1", "b", "b_drop", "general", "varlen")
EXTERNAL_VERSION = "fa_external_latency_v1"
RUN_BUDGET_SECONDS = 8 * 60 * 60
KEY_REPORT_FIELDS = ("key_ub0", "key_ub1", "key_block", "key_dtype", "key_layout", "key_bmm1_format", "key_bmm2_source",
                     "key_l1_reuse", "key_matmul_policy", "key_s1_template", "key_s2_template", "key_d_template", "key_decode_status")
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
for _engine, _pipes in (
    ("aic", ("cube", "mte1", "mte2", "mte3", "fixpipe", "scalar")),
    ("aiv", ("vec", "mte2", "mte3", "scalar")),
):
    for _pipe in _pipes:
        for _reduction in ("sum", "max"):
            PMU_RULES[f"{_engine}_{_pipe}_active_{_reduction}_us"] = (
                "PipeUtilization", f"{_engine}_{_pipe}_time(us)", _reduction,
            )
    PMU_RULES[f"{_engine}_time_sum_us"] = ("ArithmeticUtilization", f"{_engine}_time(us)", "sum")
    for _pipe in (("mte1", "mte2", "mte3") if _engine == "aic" else ("mte2", "mte3")):
        PMU_RULES[f"{_engine}_{_pipe}_instructions"] = ("Memory", f"{_engine}_{_pipe}_instructions", "sum")
    for _direction in ("read", "write"):
        PMU_RULES[f"{_engine}_mainmem_{_direction}_kb"] = (
            "Memory", f"{_direction}_main_memory_datas(KB)", f"sum:{_engine}",
        )
PMU_RULES["aic_cube_instructions"] = ("ArithmeticUtilization", "aic_cube_total_instr_number", "sum")
PMU_L2 = tuple(f"{engine}_l2_{direction}_hit_pct" for engine in ("aic", "aiv") for direction in ("read", "write"))
PMU_L2_COUNTS = tuple(
    f"{engine}_l2_{direction}_{outcome}_requests"
    for engine in ("aic", "aiv") for direction in ("read", "write") for outcome in ("hit", "miss")
)
PMU_FIELDS = (*PMU_RULES, *PMU_L2, *PMU_L2_COUNTS)
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


@lru_cache(maxsize=2048)
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
            elif reduction.startswith("sum:"):
                engine = reduction.split(":", 1)[1]
                core = "cube" if engine == "aic" else "vector"
                selected = [value for row, value in values if row.get("sub_block_id", "").startswith(core) and finite(row.get(engine + "_time(us)")) is not None]
                if not selected:
                    notes.add("missing_metric_fields")
                    continue
                result = sum(selected)
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
                    samples.setdefault(f"{engine}_l2_{direction}_hit_requests", []).append(sum(hit for hit, _ in pairs))
                    samples.setdefault(f"{engine}_l2_{direction}_miss_requests", []).append(sum(miss for _, miss in pairs))
                else:
                    notes.add("missing_l2_requests")
    return {"notes": notes, "samples": samples, "launches": len(stamps), "freq": frequencies}


def text_report(directory: Path, families: tuple = MODEL_FAMILIES) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"existing dataset is missing: {directory}")
    hardware_path = directory / "hardware.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.is_file() else {}
    rows = {name: row for name, row in recorded(directory / "routes.csv").items() if row.get("family") in families}
    family_errors = {}
    summary_path = directory / "family_validation_report.json"
    if summary_path.is_file():
        family_errors = {name: result.get("error", "") for name, result in json.loads(summary_path.read_text()).get("families", {}).items()}
    cross_validation = {}
    calibration_paths = [directory / "calibration_report.json", *(directory / family / "model.json" for family in families if family in RESEARCH_FAMILIES)]
    for calibration_path in calibration_paths:
        if not calibration_path.is_file():
            continue
        calibration = json.loads(calibration_path.read_text())
        matching = calibration.get("model_source_sha256") == cost.sha256({"source": Path(cost.__file__).read_text()})
        if calibration.get("version") == family_models.INPUT_FAMILY_VERSION:
            matching = calibration.get("source_sha256") == family_models.input_family_source(calibration["family"])
        if matching:
            for group in calibration.get("calibration_grouped_cv", {}).values():
                for axis, values in group.get("axes", {}).items():
                    for case in values.get("cases", []):
                        cross_validation.setdefault(case["case"], {}).update({
                            f"cv_{axis}_ape_percent": case.get("ape_percent", ""), "calibration_cv_pass": group["pass"],
                        })
    hardware_fields = ("soc", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes", "ub_bytes")
    fields = (
        "split", "family", *cost.INPUT_FIELDS, *hardware_fields, "priority", "tiling_key",
        "duration_us", "round1_us", "round2_us", "round3_us", "sample_count",
        "ci95_low_us", "ci95_high_us", "stable", "checked_elements", "max_abs_error",
        "status", "stage", "error", "family_error", "predicted_us", "ape_percent", "pass_10_percent",
        "case", "model_sha256", "prediction_before_measurement",
        "case_wall_seconds", "host_wall_seconds", "build_wall_seconds", "profile_wall_seconds", *KEY_REPORT_FIELDS,
        "model_scope", "prediction_quality", "extrapolated_inputs", "cv_sequence_pair_ape_percent", "cv_heads_ape_percent", "cv_batch_ape_percent", "calibration_cv_pass",
        *(name[3:].lower() for name in REPORT_TILING),
        "profile_rounds", "profile_launches", "freq_min_mhz", "freq_max_mhz", "pmu_status", "pmu_notes",
        *PMU_FIELDS,
    )
    target = directory / "diagnostics.csv"
    temporary = target.with_suffix(".tmp")
    complete = 0
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, row in sorted(rows.items()):
            result = {field: row.get(field, "") for field in fields}
            result["family_error"] = family_errors.get(row["family"], "")
            result.update(model_scope="|".join(families), **cross_validation.get(name, {}))
            derived = row.get("derived_schedule") or {}
            result.update({field: derived.get(field, "") for field in ("case_wall_seconds", "host_wall_seconds", "build_wall_seconds", "profile_wall_seconds")})
            if row.get("tiling_key"):
                result.update(decode_fa_key(int(row["tiling_key"])))
            result.update(prediction_quality=derived.get("prediction_quality", ""), extrapolated_inputs="|".join(derived.get("prediction_extrapolated_inputs", [])))
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
            complete += int(not notes)
            result.update({metric: statistics.median(values) for metric, values in grouped.items()})
            for field, value in result.items():
                parsed = finite(value) if isinstance(value, str) and value else None
                if field.endswith(("_requests", "_instructions")) and finite(value) is not None:
                    result[field] = f"{float(value):.15g}"
                elif parsed is not None and field not in ("tiling_key", *cost.INPUT_FIELDS[:7]):
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
    print(f"PMU_COMPLETE={complete}/{len(rows)}; missing data is marked in pmu_notes")


def select_shape(case: dict) -> None:
    p = case["inputs"]
    fa.SHAPE = {
        **{key: p[key] for key in cost.INPUT_FIELDS[:7]},
        "DTYPE": p["dtype"], "LAYOUT": p["layout"], "HAS_DROP": p["has_drop_mask"],
    }


def discover(params, directory: Path, hardware_trace: Path | None = None, timeout: int | None = None) -> tuple[int, int]:
    fa.build_host()
    trace = directory / "host_route.tsv"
    trace.unlink(missing_ok=True)
    env = fa.run_env(params, None)
    env.update(FA_ROUTE_TRACE=str(trace), FA_DISCOVER_ONLY="1", FA_PROFILE="0", FA_SKIP_BUILD="1")
    if hardware_trace is not None:
        env["FA_HARDWARE_TRACE"] = str(hardware_trace)
    process = subprocess.run(["bash", "run.sh"], env=env, capture_output=True, text=True, timeout=timeout)
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
    if p["layout"] in ("BSND", "TND"):
        return (((b * sequence + rows[:, None]) * heads + h) * width) + columns
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


def verify_case_schedule(case: dict, hardware: dict, priority: int, effective: dict) -> None:
    if case["family"] not in ("general", "varlen") and not case.get("official_schedule_only"):
        cost.verify_schedule(case["inputs"], hardware, priority, effective)
        return
    if priority != case["expected"] or effective.get("FA_BMM_PRESET") != 0:
        raise RuntimeError("official Host selected a different route or a modified Matmul preset")
    required = ("FA_CORE_NUM", "FA_TOTAL_TASKS", "FA_SPLIT_FACTOR", "FA_S1_BASE", "FA_S2_BASE")
    if any(int(effective.get(name, 0)) <= 0 for name in required):
        raise RuntimeError("official Host tiling trace is incomplete")


def run_case_process(env: dict, log, timeout: int | None = None):
    if timeout is None:
        return subprocess.run(["bash", "run.sh"], env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    process = subprocess.Popen(["bash", "run.sh"], env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        if isinstance(error, subprocess.TimeoutExpired):
            raise RuntimeError(f"NPU/profile timeout after {timeout}s; see {log.name}; continuing with the next case") from error
        raise
    return subprocess.CompletedProcess(["bash", "run.sh"], code)


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
    verify_case_schedule(case, hardware, priority, effective)
    Path("output/output.bin").unlink(missing_ok=True)
    env = fa.run_env(params, key)
    env.update(FA_SKIP_BUILD="1", FA_PROFILE="0", FA_WARMUP="0", FA_REPEAT="1", FA_INPUT_PATTERN="1")
    env.pop("FA_LATENCY_TRACE", None)
    env.pop("FA_EFFECTIVE_TILING_TRACE", None)
    with (summary / "recovery_check.log").open("w") as output:
        process = run_case_process(env, output, case.get("timeout_seconds"))
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
                process = run_case_process(env, log, case.get("timeout_seconds"))
            if process.returncode:
                raise RuntimeError(f"NPU/profile failed; see {summary / 'profile.log'}")
            times = profiler_samples(summary / "profile.log", key)
            effective = fa.read_effective(trace)
            if effective is None:
                raise RuntimeError("effective Host tiling trace was not produced")
            verify_case_schedule(case, hardware, priority, effective)
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


def collect(case: dict, hardware: dict, directory: Path, records: dict, csv_path: Path, ensured: set, frozen: dict | None = None, precomputed: dict | None = None, build_failures: dict | None = None) -> None:
    name = case["name"]
    if is_complete(records.get(name)):
        print(f"SKIP case={name} status=ok", flush=True)
        return
    row = {
        "version": VERSION, "case": name, "split": case["split"], "family": case["family"],
        "inputs": case["inputs"], "derived_schedule": dict(case.get("derived_schedule") or cost.derive(case["inputs"], hardware)),
        "prediction_before_measurement": bool(frozen),
    }
    stage = "predict" if frozen else "discover"
    started = phase_started = time.monotonic()
    folder = directory / "cases" / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"CASE split={case['split']} START: {name}", flush=True)
    try:
        if frozen is not None:
            row.update(predicted_us=precomputed["predicted_us"], model_sha256=cost.sha256(frozen))
            if precomputed.get("prediction_quality"):
                row["derived_schedule"].update(prediction_quality=precomputed["prediction_quality"], prediction_extrapolated_inputs=precomputed.get("extrapolated_inputs", []))
            print(f"PREDICTION BEFORE NPU: {name} predicted_us={row['predicted_us']:.6f}", flush=True)
        select_shape(case)
        params = fa.shape_params()
        stage = "discover"
        phase_started = time.monotonic()
        priority, key = discover(params, folder, timeout=case.get("timeout_seconds"))
        row["derived_schedule"]["host_wall_seconds"] = time.monotonic() - phase_started
        row.update(priority=priority, tiling_key=key)
        if priority != case["expected"]:
            raise RuntimeError(f"official route={priority}, input-derived route={case['expected']}")
        stage = "build"
        phase_started = time.monotonic()
        identity = (key, case["inputs"]["dtype"])
        if identity not in ensured:
            if build_failures is not None and identity in build_failures:
                raise RuntimeError(f"this run already failed to build {identity}: {build_failures[identity]}")
            try:
                fa.build(key)
            except Exception as error:
                if build_failures is not None:
                    build_failures[identity] = str(error)
                raise
            ensured.add(identity)
        row["derived_schedule"]["build_wall_seconds"] = time.monotonic() - phase_started
        stage = "measure"
        phase_started = time.monotonic()
        row.update(profile_measure(case, params, key, priority, hardware, folder))
        row["derived_schedule"]["profile_wall_seconds"] = time.monotonic() - phase_started
        if frozen is not None:
            error = abs(row["predicted_us"] / row["duration_us"] - 1)
            row.update(ape_percent=error * 100, pass_10_percent=error <= cost.RELATIVE_ERROR_LIMIT + 1e-12 and (row["stable"] or case["split"] == "external"))
        row.update(status="ok", stage="complete")
        print(f"CASE OK: {name} actual_us={row['duration_us']:.6f} stable={row['stable']} error_percent={row.get('ape_percent', '')} pass_10_percent={row.get('pass_10_percent', '')}", flush=True)
    except Exception as error:
        row.update(status="failed", stage=stage, error=str(error))
        print(f"CASE FAILED: {name} stage={stage} error={error}", flush=True)
    if case["split"] == "external":
        now = time.monotonic()
        row["derived_schedule"]["case_wall_seconds"] = now - started
        if row["status"] == "failed" and stage in ("discover", "build", "measure"):
            field = {"discover": "host", "build": "build", "measure": "profile"}[stage] + "_wall_seconds"
            row["derived_schedule"][field] = now - phase_started
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


def collect_other_families(directory: Path, hardware: dict, full_plan: list[dict], records: dict,
                          csv_path: Path, ensured: set, protocol_signature: str) -> dict:
    summaries = {}
    selected = [case for case in full_plan if case["family"] in RESEARCH_FAMILIES]
    print(f"OTHER FA PLAN: cached_success={sum(records.get(case['name'], {}).get('status') == 'ok' for case in selected)} new_holdouts={sum(case['split'] == 'holdout' and records.get(case['name'], {}).get('status') != 'ok' for case in selected)}; sequential NPU execution", flush=True)
    for family in RESEARCH_FAMILIES:
        source = family_models.input_family_source(family)
        folder = directory / family
        folder.mkdir(parents=True, exist_ok=True)
        plan = [case for case in full_plan if case["family"] == family]
        calibration = [case for case in plan if case["split"] == "calibration"]
        holdout = [case for case in plan if case["split"] == "holdout"]
        signature = cost.sha256({"protocol": protocol_signature, "family": family, "source": source})
        print(f"FAMILY START: {family}; calibration={len(calibration)} new_holdouts={len(holdout)}", flush=True)
        try:
            for case in calibration:
                existing = records.get(case["name"], {})
                if existing.get("status") == "ok":
                    print(f"MEASURE CACHE HIT: {case['name']} stable={existing.get('stable')}", flush=True)
                else:
                    collect(case, hardware, directory, records, csv_path, ensured)
            exposed = any(case["name"] in records for case in holdout)
            model_path = folder / "model.json"
            model = json.loads(model_path.read_text()) if model_path.is_file() else None
            if model is not None and (model.get("plan_sha256") != signature or model.get("source_sha256") != source):
                if exposed:
                    raise RuntimeError("family model changed after holdout exposure; a new blind set is required")
                model = None
            if model is None:
                model = family_models.fit_input_family(list(records.values()), hardware, family, signature)
                cost.save_json(model_path, model)
            predictions_path = folder / "predictions_before_measurement.json"
            payload = json.loads(predictions_path.read_text()) if predictions_path.is_file() else None
            if payload is not None and payload.get("model_sha256") != cost.sha256(model):
                if exposed:
                    raise RuntimeError("saved family predictions belong to another frozen model")
                payload = None
            if payload is None:
                payload = {"model_sha256": cost.sha256(model), "predictions": {case["name"]: family_models.predict_input_family(case["inputs"], model) for case in holdout}}
                cost.save_json(predictions_path, payload)
            cv_pass = all(value["pass"] for value in model["calibration_grouped_cv"].values())
            print(f"FAMILY MODEL: {family}; calibration_cv_pass={cv_pass}; prediction_quality={'calibration_cv_pass' if cv_pass else 'provisional'}", flush=True)
            for case in holdout:
                if records.get(case["name"], {}).get("status") == "ok":
                    print(f"MEASURE CACHE HIT: {case['name']}", flush=True)
                    continue
                collect(case, hardware, directory, records, csv_path, ensured, model, payload["predictions"][case["name"]])
            report = validation_report(plan, records, model, folder)
            completed = [records.get(case["name"], {}) for case in holdout]
            errors = [finite(row.get("ape_percent")) for row in completed]
            summaries[family] = {
                "pass": report["pass"], "calibration_cv_pass": cv_pass,
                "requested": len(holdout), "runtime_ok": sum(row.get("status") == "ok" for row in completed),
                "point_error_pass": sum(error is not None and error <= 10 for error in errors),
                "max_ape_percent": report["families"][family]["max_ape_percent"],
            }
            print(f"FAMILY END: {family}; {summaries[family]}", flush=True)
        except Exception as error:
            summaries[family] = {"pass": False, "error": str(error)}
            cost.save_json(folder / "validation_report.json", summaries[family])
            print(f"FAMILY FAILED: {family}; {error}; continuing", flush=True)
    result = {"families": summaries, "pass": all(value["pass"] for value in summaries.values()), "scope": "independent prospective family research; provisional predictions are not qualified models"}
    cost.save_json(directory / "family_validation_report.json", result)
    text_report(directory, RESEARCH_FAMILIES)
    family_models.pipeline_report(directory / "diagnostics.csv")
    return result


def external_design(hardware: dict) -> list[dict]:
    plan = []
    counts = Counter()
    seen = {cost.sha256(case["inputs"]) for case in cost.design(hardware)}
    priorities = {"same_ab": 95, "s1": 97, "b": 98, "b_drop": 98, "general": 96, "varlen": 94}

    def add(family, b, n, nk, s1, s2, d, dv, dtype="FP16", layout="BNSD", drop=0, coverage=None, official_schedule_only=False):
        inputs = cost.normalize(dict(zip(cost.INPUT_FIELDS, (b, n, nk, s1, s2, d, dv, dtype, layout, drop))))
        identity = cost.sha256(inputs)
        if identity in seen:
            raise ValueError("external plan contains an already scheduled operator input")
        seen.add(identity)
        if family in ("general", "varlen") or official_schedule_only:
            derived = {"family": family, "priority": priorities[family], "prediction_quality": "unmodeled_family"}
        else:
            derived = cost.derive(inputs, hardware)
            if derived["family"] != family:
                raise ValueError(f"external input selects {derived['family']} instead of {family}")
        name = f"{family}_{layout.lower()}_b{b}_n{n}_nk{nk}_s{s1}_s{s2}_d{d}_v{dv}_drop{drop}"
        if coverage is not None:
            name += f"_{dtype.lower()}"
        case = {"name": name, "split": "external", "family": family, "expected": priorities[family],
                "inputs": inputs, "derived_schedule": derived, "timeout_seconds": 300}
        if coverage is not None:
            case["coverage"] = coverage
        if official_schedule_only:
            case["official_schedule_only"] = True
        plan.append(case)
        counts[family] += 1

    for layout in ("BNSD", "SBH"):
        for n in (16, 24):
            for sequence in (2304, 3584, 5376, 7168):
                add("same_ab", 1, n, n, sequence, sequence, 192, 128, "BF16", layout)
        for b, n, s1, s2 in ((1, 64, 2304, 2304), (1, 64, 5376, 5376), (2, 16, 1536, 3072), (1, 24, 3072, 5120)):
            add("same_ab", b, n, n, s1, s2, 192, 128, "BF16", layout)
    for b, n, nk in ((1, 6, 2), (1, 20, 5), (2, 12, 3), (4, 8, 2)):
        for s1, s2 in ((96, 192), (192, 320), (288, 448), (384, 640), (768, 960), (1280, 1024)):
            add("s1", b, n, nk, s1, s2, 128, 128)
    small = ((3, 40, 80), (3, 80, 40), (5, 96, 80), (6, 64, 112), (7, 48, 96), (8, 112, 64))
    for b in (3, 6, 12, 24):
        for n, s1, s2 in small:
            add("b", b, n, 1, s1, s2, 64, 64)
        for n, s1, s2 in (small[0], small[2], small[3], small[5]):
            add("b_drop", b, n, 1, s1, s2, 64, 64, drop=1)
    for b, n, s1, s2 in ((32, 4, 127, 127), (48, 4, 127, 127), (24, 4, 95, 127), (12, 4, 95, 127),
                          (6, 5, 79, 65), (3, 3, 95, 49), (24, 3, 111, 81), (48, 2, 127, 97)):
        add("b_drop", b, n, 1, s1, s2, 64, 64, drop=1)
    for b, n, nk in ((1, 4, 1), (1, 12, 3), (2, 8, 2), (4, 4, 1)):
        for s1, s2 in ((96, 1536), (256, 2048), (640, 3072), (1280, 4096)):
            add("general", b, n, nk, s1, s2, 128, 128)
    for b, n, nk in ((1, 4, 1), (2, 4, 1), (2, 8, 2), (4, 4, 1)):
        for s1, s2 in ((96, 192), (160, 320), (224, 640), (384, 896)):
            add("varlen", b, n, nk, s1, s2, 128, 128, layout="TND")
    targets = dict(same_ab=128, s1=128, b=128, b_drop=128, general=64, varlen=64)

    def extend(family, *args, **kwargs):
        if counts[family] >= targets[family]:
            return
        b, n, nk, s1, s2, d, dv = args
        shape = (*args, kwargs.get("dtype", "FP16"), kwargs.get("layout", "BNSD"), kwargs.get("drop", 0))
        if cost.sha256(cost.normalize(dict(zip(cost.INPUT_FIELDS, shape)))) in seen:
            return
        try:
            add(family, *args, **kwargs)
        except ValueError as error:
            if "selects" in str(error) or "supported" in str(error) or "scheduling" in str(error):
                return
            raise

    for b, n, s1, s2 in ((1, 8, 128, 512), (1, 8, 256, 1536), (2, 8, 640, 2304), (1, 16, 384, 3072)):
        extend("same_ab", b, n, 1, s1, s2, 96, 96, coverage="same_ab_fp16_d96", official_schedule_only=True)
        extend("same_ab", b, n, 1, s1, max(s2, 1536), 192, 128, coverage="same_ab_fp16_d192", official_schedule_only=True)
    for s1, s2 in ((160, 272), (320, 336), (640, 400), (768, 528)):
        extend("s1", 1, 8, 2, s1, s2, 128, 128, coverage="s1_nz", official_schedule_only=True)
    for s1, s2 in ((160, 512), (320, 768), (640, 1024), (960, 512)):
        extend("s1", 1, 8, 2, s1, s2, 64, 64, coverage="s1_bmm2_l1", official_schedule_only=True)
        extend("s1", 2, 48, 6, s1, s2, 128, 128, coverage="s1_incremental_l1_reuse", official_schedule_only=True)
        extend("s1", 1, 8, 2, s1, s2, 128, 128, dtype="BF16", coverage="s1_bf16", official_schedule_only=True)
    for b, n, s1, s2 in ((2, 3, 40, 80), (4, 4, 72, 56), (8, 6, 48, 96), (16, 8, 80, 64)):
        extend("b", b, n, 1, s1, s2, 64, 64, dtype="BF16", coverage="b_bf16", official_schedule_only=True)
        extend("b", b, n, 1, s1, s2, 64, 64, layout="BSND", coverage="b_bsnd", official_schedule_only=True)
    for layout in ("SBH", "BSND"):
        for s1, s2 in ((160, 1280), (320, 1792), (512, 2304), (896, 3328)):
            extend("general", 1, 8, 2, s1, s2, 128, 128, layout=layout, coverage=f"general_{layout.lower()}")
    for s1, s2 in ((160, 1536), (320, 2048), (512, 3072), (896, 4096)):
        extend("general", 1, 8, 2, s1, s2, 64, 64, coverage="general_d64")
        extend("general", 1, 8, 2, s1, s2, 128, 128, dtype="BF16", coverage="general_bf16")
    for s1, s2 in ((160, 1041), (320, 1153), (512, 1553), (896, 2065)):
        extend("general", 1, 8, 2, s1, s2, 128, 128, coverage="general_nz")
    for dtype in ("FP16", "BF16"):
        for s1, s2 in ((160, 384), (320, 768), (768, 1536), (1280, 2048)):
            extend("varlen", 2, 4, 1, s1, s2, 128, 128, dtype=dtype, layout="TND", coverage="tnd_uniform_packed")
        for s1, s2 in ((2048, 2048), (2048, 3072), (3072, 2048), (3072, 3072)):
            extend("varlen", 4, 4, 1, s1, s2, 128, 128, dtype=dtype, layout="TND", coverage="tnd_same_ab")
    buckets = {family: defaultdict(deque) for family in targets}

    def queue(family, *args, **kwargs):
        group = (*args[:3], *args[5:], kwargs.get("dtype", "FP16"), kwargs.get("layout", "BNSD"), kwargs.get("drop", 0))
        buckets[family][group].append((args, kwargs))

    for layout in ("BNSD", "SBH"):
        for n in (4, 12, 20, 40):
            for nk in (n, 8 if n == 40 else 4):
                for s1, s2 in ((1792, 2304), (2560, 3840), (3328, 4608), (5120, 6144), (6400, 3584), (7680, 5120), (4096, 2560), (2048, 6400)):
                    queue("same_ab", 1, n, nk, s1, s2, 192, 128, dtype="BF16", layout=layout, coverage="same_ab_dense_gqa")
    for b, n, nk in ((1, 4, 1), (1, 12, 3), (1, 24, 6), (2, 12, 3), (4, 8, 2), (2, 20, 5)):
        for s1, s2 in ((80, 192), (2048, 1024), (352, 640), (1536, 896), (144, 320), (896, 768), (208, 448), (576, 512),
                       (80, 1024), (2048, 192), (352, 896), (1536, 320), (144, 768), (896, 448), (208, 640), (576, 896)):
            queue("s1", b, n, nk, s1, s2, 128, 128, coverage="s1_dense_gqa")
    for b in (2, 5, 8, 10, 20, 32, 48):
        for n, nk in ((1, 1), (2, 1), (2, 2), (3, 1), (4, 1), (4, 4), (6, 2), (8, 2)):
            for s1, s2 in ((24, 24), (120, 120), (120, 40), (40, 120), (56, 88), (88, 56), (72, 104), (104, 72)):
                queue("b", b, n, nk, s1, s2, 64, 64, coverage="b_dense_gqa")
            for s1, s2 in ((24, 25), (120, 121), (120, 40), (40, 120), (56, 57), (88, 89), (72, 104), (104, 105)):
                queue("b_drop", b, n, nk, s1, s2, 64, 64, drop=1, coverage="drop_mask_gqa")
    for b, n, nk in ((1, 6, 2), (1, 16, 4), (2, 12, 3), (4, 8, 2)):
        for s1, s2 in ((192, 1408), (384, 2176), (768, 2816), (1536, 3584), (2048, 4608), (2560, 5632), (3200, 6656), (3840, 7680)):
            queue("general", b, n, nk, s1, s2, 128, 128, coverage="general_dense_gqa")
        for s1, s2 in ((80, 160), (144, 288), (288, 576), (448, 896), (768, 1280), (1280, 1920), (1536, 2560), (2048, 3072)):
            queue("varlen", b, n, nk, s1, s2, 128, 128, layout="TND", coverage="tnd_uniform_packed")
    for family, groups in buckets.items():
        active = deque(groups.values())
        while active and counts[family] < targets[family]:
            group = active.popleft()
            args, kwargs = group.popleft()
            extend(family, *args, **kwargs)
            if group:
                active.append(group)
    counts = Counter(case["family"] for case in plan)
    if counts != Counter(targets):
        raise ValueError(f"incomplete external test plan: {dict(counts)}")
    return plan


@lru_cache(maxsize=4)
def fa_key_schema(path: str) -> tuple:
    text = Path(path).read_text().split("ASCENDC_TPL_SEL(", 1)[0]
    fields = re.findall(r"ASCENDC_TPL_UINT_DECL\(\s*(\w+)\s*,\s*ASCENDC_TPL_(\d+)_BW\s*,\s*ASCENDC_TPL_UI_LIST\s*,([^)]*)\)", text)
    if not fields:
        raise ValueError("official FA template key declarations were not found")
    return tuple((name, int(width), tuple(int(value) for value in re.findall(r"\d+", values))) for name, width, values in fields)


def decode_fa_key(key: int) -> dict:
    source = Path(os.environ.get("OPS_TRANSFORMER_ROOT", str(Path(__file__).resolve().parent.parent / "ops-transformer-official-8.5.0")))
    source /= "attention/flash_attention_score/op_kernel/arch32/flash_attention_score_template_tiling_key.h"
    try:
        decoded, shift = {}, 0
        for name, width, values in fa_key_schema(str(source)):
            index = key >> shift & ((1 << width) - 1)
            if index >= len(values):
                raise ValueError("key index is outside the official declaration")
            decoded[name] = values[index]
            shift += width
        if key >> shift:
            raise ValueError("key has bits outside the official declaration")
        axes = {0: "B", 1: "N2", 2: "G", 3: "S1", 4: "S2", 5: "D", 9: "NONE"}
        return {
            "key_ub0": axes[decoded["UB0"]], "key_ub1": axes[decoded["UB1"]], "key_block": axes[decoded["Block"]],
            "key_dtype": {0: "FP16", 1: "FP32", 2: "BF16", 3: "FP16_HIGH_PRECISION"}[decoded["DataType"]],
            "key_layout": {0: "NONE", 1: "BSH", 2: "SBH", 3: "BNSD", 4: "TND"}[decoded["Layout"]],
            "key_bmm1_format": ("ND", "NZ")[decoded["Bmm1Format"]], "key_bmm2_source": ("GM", "L1")[decoded["Bmm2Source"]],
            "key_l1_reuse": decoded["EnableL1Reuse"], "key_matmul_policy": ("NORMAL", "UNSPLITK")[decoded["MatmulPolicyType"]],
            "key_s1_template": decoded["S1TemplateType"] * 16, "key_s2_template": decoded["S2TemplateType"] * 16,
            "key_d_template": decoded["DTemplateType"] * 16, "key_decode_status": "official_v8.5.0_schema",
        }
    except (OSError, ValueError, KeyError, IndexError):
        return {"key_decode_status": "unavailable; numeric key retained"}


def external_order(plan: list[dict]) -> list[dict]:
    groups = defaultdict(deque)
    for case in plan:
        groups[case.get("coverage", case["family"])].append(case)
    active, ordered = deque(groups.values()), []
    while active:
        group = active.popleft()
        ordered.append(group.popleft())
        if group:
            active.append(group)
    return ordered


def external_report(directory: Path, folder: Path, plan: list[dict], payload: dict) -> dict:
    text_report(folder, ALL_FAMILIES)
    original = directory / "diagnostics.csv"
    with original.open(newline="") as source:
        reader = csv.DictReader(source)
        fields, rows = list(reader.fieldnames or []), [row for row in reader if row.get("split") != "external"]
    with (folder / "diagnostics.csv").open(newline="") as source:
        reader = csv.DictReader(source)
        fields.extend(name for name in reader.fieldnames or [] if name not in fields)
        additional = list(reader)
    for name in ("prediction_model_kind", "prediction_status", "prediction_error", "coverage"):
        if name not in fields:
            fields.append(name)
    for row in additional:
        saved = payload["predictions"][row["case"]]
        row.update(prediction_model_kind=saved["kind"], prediction_status=saved["status"])
        case = next(case for case in plan if case["name"] == row["case"])
        row["coverage"] = case.get("coverage", case["family"])
        if saved["status"] == "predicted" and not external_prediction_matches(row, saved):
            row.update(prediction_status="provenance_mismatch", prediction_error="record does not match the prediction frozen before NPU execution")
            row.update(predicted_us="", ape_percent="", pass_10_percent="")
        elif saved.get("error"):
            row["prediction_error"] = saved["error"]
        if saved["kind"] == "pipeline" and row["prediction_status"] == "predicted":
            prediction = saved["result"]
            row.update(pipeline_version=family_models.PIPELINE_VERSION, pipeline_predicted_us=row.get("predicted_us", ""),
                       pipeline_ape_percent=row.get("ape_percent", ""), pipeline_qualification="prospective_external",
                       pipeline_pass_10_percent=finite(row.get("ape_percent")) <= 10 if finite(row.get("ape_percent")) is not None else "",
                       pipeline_raw_elapsed_us=prediction["pipeline_us"], pipeline_critical_pair=prediction["critical_pair"],
                       pipeline_extrapolated_inputs="|".join(prediction["extrapolated_inputs"]), pipeline_adapter_extrapolated=prediction["adapter_extrapolated"])
            for name in ("active_vector_workers", "active_cube_pairs", "critical_cube_rows", "scheduled_tasks", "mma_instructions"):
                row[f"pipeline_{name}"] = prediction[name]
            for name, term in (("initialization", "initialization_us"), ("service", "pipeline_us"), ("cube_pair_setup", "active_cube_pairs"), ("adapter", "adapter_passes")):
                row[f"pipeline_{name}_us"] = prediction["cost_terms_us"][term]
            for name, value in prediction["stage_service_us"].items():
                row[f"pipeline_{name}_service_us"] = value
        if row.get("ape_percent"):
            row["pass_10_percent"] = float(row["ape_percent"]) <= 10
    rows.extend(additional)
    for row in additional:
        fields.extend(name for name in row if name not in fields)
    temporary = original.with_suffix(".external.tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format(value, ".6g") if isinstance(value, float) else value for name, value in row.items()})
    temporary.replace(original)
    (folder / "diagnostics.csv").unlink()
    latest = recorded(folder / "routes.csv")
    families = {}
    for family in ALL_FAMILIES:
        cases = [case for case in plan if case["family"] == family]
        observed = [latest.get(case["name"], {}) for case in cases]
        errors = [float(row["ape_percent"]) for case, row in zip(cases, observed) if row.get("status") == "ok" and row.get("ape_percent") not in (None, "")
                  and external_prediction_matches(row, payload["predictions"][case["name"]])]
        families[family] = {"requested": len(cases), "runtime_ok": sum(row.get("status") == "ok" for row in observed),
                            "failed": sum(row.get("status") == "failed" for row in observed),
                            "pending": sum(not row for row in observed),
                            "measured_tiling_keys": sorted({int(row["tiling_key"]) for row in observed if row.get("status") == "ok" and row.get("tiling_key")}),
                            "predicted_before_measurement": sum(payload["predictions"][case["name"]]["status"] == "predicted" for case in cases),
                            "error_evaluated": len(errors), "point_error_pass": sum(value <= 10 for value in errors),
                            "mape_percent": statistics.mean(errors) if errors else None, "max_ape_percent": max(errors, default=None)}
    variants = Counter()
    for row in additional:
        if row["status"] == "ok":
            variants[(row["family"], row["dtype"], row["layout"], row["tiling_key"], row.get("key_bmm1_format", ""), row.get("key_bmm2_source", ""), row.get("key_l1_reuse", ""))] += 1
    result = {"version": EXTERNAL_VERSION, "families": families, "frozen_predictions_sha256": cost.sha256(payload),
              "requested_coverage": dict(Counter(case.get("coverage", case["family"]) for case in plan)),
              "measured_variants": [{"family": key[0], "dtype": key[1], "layout": key[2], "tiling_key": key[3], "bmm1_format": key[4], "bmm2_source": key[5], "l1_reuse": key[6], "cases": count} for key, count in sorted(variants.items())],
              "measurements_complete": all(value["runtime_ok"] + value["failed"] == value["requested"] for value in families.values())}
    cost.save_json(folder / "summary.json", result)
    print(f"EXTERNAL SUMMARY: {families}", flush=True)
    print(f"CSV_REPORT={original}; original measurements plus external cases; no raw-log bundle", flush=True)
    return result


def external_prediction_matches(row: dict, saved: dict) -> bool:
    value = finite(row.get("predicted_us"))
    return bool(saved["status"] == "predicted" and value is not None
                and str(row.get("prediction_before_measurement")).lower() == "true"
                and row.get("model_sha256") == saved["model_sha256"]
                and math.isclose(value, saved["result"]["predicted_us"], rel_tol=1e-5, abs_tol=1e-8))


def collect_external(directory: Path, hardware: dict, signature: str, ensured: set, deadline: float | None = None) -> dict:
    run_started = time.monotonic()
    profile_counters.cache_clear()
    if deadline is None:
        deadline = run_started + RUN_BUDGET_SECONDS
    folder = directory / "external"
    folder.mkdir(parents=True, exist_ok=True)
    cost.save_json(folder / "hardware.json", hardware)
    plan = external_design(hardware)
    protocol = {"version": EXTERNAL_VERSION, "protocol_sha256": signature, "plan": plan,
                "predictor_sha256": cost.sha256({"source": Path(family_models.__file__).read_text()})}
    manifest = folder / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else None
    if previous is not None:
        if cost.sha256({key: value for key, value in previous.items() if key != "plan"}) != cost.sha256({key: value for key, value in protocol.items() if key != "plan"}):
            raise RuntimeError("the external measurement protocol or predictor changed; existing measurements are preserved")
        current = {case["name"]: case for case in plan}
        if any(case["name"] not in current or cost.sha256(case) != cost.sha256(current[case["name"]]) for case in previous["plan"]):
            raise RuntimeError("an existing external shape or its measurement settings changed")
    records = recorded(folder / "routes.csv")
    snapshot = folder / "predictions_before_measurement.json"
    if snapshot.is_file():
        payload = json.loads(snapshot.read_text())
        accepted = {cost.sha256(protocol)}
        if previous is not None:
            accepted.add(cost.sha256(previous))
        if payload["protocol_sha256"] not in accepted:
            raise RuntimeError("saved external predictions belong to another measurement plan")
        if any(name not in payload["predictions"] for name in records):
            raise RuntimeError("an exposed external shape has no saved premeasurement prediction record")
    else:
        if records:
            raise RuntimeError("external measurements exist but their frozen prediction snapshot is missing; data were retained")
        sameab = json.loads((directory / "model.json").read_text())
        pipeline = json.loads((directory / "pipeline_models.json").read_text())["families"]
        payload = {"models": {"same_ab": sameab, **pipeline}, "predictions": {}}
    models = payload["models"]
    for case in plan:
        if case["name"] in payload["predictions"]:
            continue
        family = case["family"]
        kind = "same_ab" if family == "same_ab" else "pipeline" if family in models else "unmodeled"
        item = {"kind": kind, "status": "unmodeled_variant" if case.get("official_schedule_only") else "unmodeled_family"}
        if family in models and not case.get("official_schedule_only"):
            try:
                result = cost.predict(case["inputs"], models[family]) if kind == "same_ab" else family_models.predict_pipeline(case["inputs"], models[family])
                result["prediction_quality"] = "frozen_before_measurement"
                item.update(status="predicted", result=result, model_sha256=cost.sha256(models[family]))
            except (ValueError, RuntimeError) as error:
                item.update(status="uncalibrated_cost", error=str(error))
        payload["predictions"][case["name"]] = item
    payload["protocol_sha256"] = cost.sha256(protocol)
    cost.save_json(snapshot, payload)
    cost.save_json(manifest, protocol)
    cached = sum(records.get(case["name"], {}).get("status") == "ok" for case in plan)
    print(f"EXTERNAL PLAN: total={len(plan)} cached_success={cached} remaining={len(plan) - cached} families={dict(Counter(case['family'] for case in plan))}", flush=True)
    print("EXTERNAL: sequential cases; one build per encountered key/dtype; 3 rounds x 10 timings; no new measurements enter frozen models", flush=True)
    print(f"RUN BUDGET: 8 hours including preparation; remaining_seconds={max(0, deadline - time.monotonic()):.0f}; finish the current case/build before stopping", flush=True)
    failures = {}
    budget_reached = False
    try:
        for index, case in enumerate(external_order(plan), 1):
            name = case["name"]
            if records.get(name, {}).get("status") == "ok":
                print(f"EXTERNAL {index}/{len(plan)} CACHE HIT: {name}", flush=True)
                continue
            if time.monotonic() >= deadline:
                budget_reached = True
                print("RUN BUDGET REACHED: no new cases will start; successful measurements are reusable on the next run", flush=True)
                break
            saved = payload["predictions"][name]
            case["derived_schedule"]["prediction_quality"] = saved["status"]
            print(f"EXTERNAL {index}/{len(plan)} START: {name}; prediction={saved['status']}", flush=True)
            model = payload["models"].get(case["family"]) if saved["status"] == "predicted" else None
            collect(case, hardware, folder, records, folder / "routes.csv", ensured, model, saved.get("result"), failures)
            if index % 16 == 0:
                external_report(directory, folder, plan, payload)
    finally:
        result = external_report(directory, folder, plan, payload)
        result.update(stopped_for_time_budget=budget_reached, run_budget_seconds=RUN_BUDGET_SECONDS,
                      elapsed_seconds=time.monotonic() - run_started)
        cost.save_json(folder / "summary.json", result)
    return result


def main() -> None:
    run_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    if args.report_only:
        directory = root / "output" / VERSION
        external = directory / "external"
        families = ALL_FAMILIES if (external / "manifest.json").is_file() else RESEARCH_FAMILIES if (directory / "family_validation_report.json").is_file() else MODEL_FAMILIES
        text_report(directory, families)
        if families != MODEL_FAMILIES:
            family_models.pipeline_report(directory / "diagnostics.csv")
        if (external / "predictions_before_measurement.json").is_file():
            payload = json.loads((external / "predictions_before_measurement.json").read_text())
            plan = json.loads((external / "manifest.json").read_text())["plan"]
            external_report(directory, external, plan, payload)
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
    blind_exposed = any(row.get("split") == "holdout" for row in records.values())
    if frozen is not None and (frozen.get("plan_sha256") != signature or frozen.get("fit_method") != cost.FIT_METHOD):
        if blind_exposed:
            raise RuntimeError("the existing frozen model uses different model code; measurements are preserved, but its blind set must not be reused to validate a replacement")
        frozen = None
    calibration = [case for case in plan if case["split"] == "calibration"]
    holdout = [case for case in plan if case["split"] == "holdout"]
    print(f"MEASUREMENT PLAN: calibration_cached={sum(is_complete(records.get(case['name'])) for case in calibration)}/{len(calibration)} blind_remaining={sum(not is_complete(records.get(case['name'])) for case in holdout)}", flush=True)
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
    payload = json.loads(predictions_path.read_text()) if predictions_path.is_file() else None
    if payload is not None and payload["model_sha256"] != cost.sha256(frozen):
        if blind_exposed:
            raise RuntimeError("saved predictions belong to a different frozen model")
        payload = None
    if payload is None:
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
        text_report(directory)
        raise SystemExit(1)
    collect_other_families(directory, hardware, full_plan, records, csv_path, ensured, signature)
    text_report(directory, ALL_FAMILIES)
    family_models.pipeline_report(directory / "diagnostics.csv")
    external = collect_external(directory, hardware, signature, ensured, run_started + RUN_BUDGET_SECONDS)
    if (not external["measurements_complete"] and not external["stopped_for_time_budget"]) or any(value["failed"] for value in external["families"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
