from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import main as fa
from tiling import base


VERSION = "sameab_latency_v1"
WARMUP = 3
REPEAT = 10
ROUNDS = 2
CALIBRATION_CASES = (
    "deepseek_v3_generate_s4096_bnsd_tp1",
    "deepseek_v3_pipeline_s4096_sbh_tp1",
    "deepseek_v3_pretrain_s4096_sbh_tp2",
    "deepseek_v3_tune_s4096_bnsd_tp2",
    "deepseek_v3_pretrain_s4096_sbh_tp4",
)
HOLDOUT_CASES = (
    "deepseek_v3_surrogate_s4096_bnsd_tp8",
    "deepseek_v3_bf16_s8192_sbh_tp1",
)
FIXED_CONFIGS = (
    ("tile_128_16_8_p0", 128, 16, 8, 0),
    ("tile_128_64_4_p1", 128, 64, 4, 1),
    ("tile_128_128_4_p2", 128, 128, 4, 2),
    ("tile_128_128_8_p3", 128, 128, 8, 3),
    ("tile_256_32_8_p4", 256, 32, 8, 4),
    ("tile_256_128_4_p5", 256, 128, 4, 5),
    ("tile_256_128_8_p6", 256, 128, 8, 6),
)
FIELDS = (
    "version",
    "split",
    "case",
    "record_type",
    "round",
    "direction",
    "order",
    "config_id",
    "B",
    "N1",
    "N2",
    "S1",
    "S2",
    "D",
    "DV",
    "dtype",
    "layout",
    "npu_id",
    "soc_unit",
    "ops_transformer_version",
    "priority",
    "tiling_key",
    "requested_s1_base",
    "requested_s2_base",
    "requested_n_ratio",
    "requested_bmm_preset",
    "effective_s1_base",
    "effective_s2_base",
    "effective_n_ratio",
    "effective_bmm_preset",
    "bmm1_base_m",
    "bmm1_base_n",
    "bmm1_base_k",
    "bmm2_base_m",
    "bmm2_base_n",
    "bmm2_base_k",
    "duration_us",
    "samples_us",
    "sample_count",
    "matched",
    "correct",
    "max_abs_error",
    "mean_abs_error",
    "error_ratio",
    "exit_code",
    "status",
    "error",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_payload() -> dict:
    tracked = (
        Path(__file__),
        Path("main.py"),
        Path("fa_launcher.cpp"),
        Path("patches/fa_candidate_hook.patch"),
        Path("tiling/validators/attention/AttentionValidator.py"),
        Path("tiling/validators/attention/FlashAttentionScoreValidator.py"),
    )
    payload = {
        "version": VERSION,
        "split_policy": "holdout_by_complete_shape",
        "calibration_cases": list(CALIBRATION_CASES),
        "holdout_cases": list(HOLDOUT_CASES),
        "fixed_configs": [list(item) for item in FIXED_CONFIGS],
        "warmup": WARMUP,
        "repeat": REPEAT,
        "rounds": ROUNDS,
        "npu_id": fa.NPU_ID,
        "soc_unit": os.environ.get("FA_SOC_UNIT", "ascend910b"),
        "ops_transformer_version": os.environ.get("OPS_TRANSFORMER_VERSION", "v8.5.0"),
        "source_sha256": {str(path): digest(path) for path in tracked},
    }
    payload["signature"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def prepare_output(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    expected = manifest_payload()
    manifest = root / "manifest.json"
    if manifest.is_file():
        current = json.loads(manifest.read_text())
        if current.get("signature") != expected["signature"]:
            raise RuntimeError(f"dataset manifest mismatch: {manifest}")
    else:
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest)
    return root / "calibration.csv", root / "holdout.csv"


def completed(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.is_file():
        return set()
    rows = csv.DictReader(path.open(newline=""))
    return {
        (row["case"], row["round"], row["record_type"], row["config_id"])
        for row in rows
        if row.get("status") == "ok"
    }


def append(path: Path, row: dict) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDS})


def shape_fields() -> dict:
    return {
        "B": fa.SHAPE["B"],
        "N1": fa.SHAPE["N1"],
        "N2": fa.SHAPE["N2"],
        "S1": fa.SHAPE["S1"],
        "S2": fa.SHAPE["S2"],
        "D": fa.SHAPE["D"],
        "DV": fa.SHAPE["DV"],
        "dtype": fa.SHAPE["DTYPE"],
        "layout": fa.SHAPE["LAYOUT"],
    }


def profile_samples() -> list[float]:
    profiles = list(Path(".").glob("OPPROF_*"))
    if not profiles:
        return []
    profile = max(profiles, key=lambda item: item.stat().st_mtime)
    tables = list(profile.rglob("OpBasicInfo.csv"))
    if not tables:
        return []
    table = max(tables, key=lambda item: item.stat().st_mtime)
    values = pd.to_numeric(pd.read_csv(table)["Task Duration(us)"], errors="coerce").dropna().tolist()
    return [float(value) for value in values[-min(REPEAT, len(values)):]]


def correctness() -> tuple[bool, float, float, float]:
    output_path = Path("output/output.bin")
    golden_path = Path("output/golden.bin")
    if not output_path.is_file() or not golden_path.is_file():
        return False, math.inf, math.inf, 1.0
    if output_path.stat().st_size == 0 or output_path.stat().st_size != golden_path.stat().st_size:
        return False, math.inf, math.inf, 1.0
    count = output_path.stat().st_size // np.dtype(np.float32).itemsize
    output = np.memmap(output_path, dtype=np.float32, mode="r", shape=(count,))
    golden = np.memmap(golden_path, dtype=np.float32, mode="r", shape=(count,))
    bad_count = 0
    finite_count = 0
    maximum = 0.0
    total_error = 0.0
    chunk = 1 << 20
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        actual = np.asarray(output[start:stop])
        expected = np.asarray(golden[start:stop])
        finite = np.isfinite(actual) & np.isfinite(expected)
        delta = np.abs(actual - expected)
        bad_count += int(np.count_nonzero(~finite | (delta >= 1e-9)))
        if np.any(finite):
            finite_delta = delta[finite]
            maximum = max(maximum, float(np.max(finite_delta)))
            total_error += float(np.sum(finite_delta, dtype=np.float64))
            finite_count += int(finite_delta.size)
    mean = total_error / finite_count if finite_count else math.inf
    ratio = bad_count / count
    return ratio <= 1e-4, maximum, mean, ratio


def effective_fields(effective: dict[str, int] | None) -> dict:
    if effective is None:
        return {}
    return {
        "effective_s1_base": effective["FA_S1_BASE"],
        "effective_s2_base": effective["FA_S2_BASE"],
        "effective_n_ratio": effective["FA_N_RATIO"],
        "effective_bmm_preset": effective["FA_BMM_PRESET"],
        "bmm1_base_m": effective["FA_BMM1_BASE_M"],
        "bmm1_base_n": effective["FA_BMM1_BASE_N"],
        "bmm1_base_k": effective["FA_BMM1_BASE_K"],
        "bmm2_base_m": effective["FA_BMM2_BASE_M"],
        "bmm2_base_n": effective["FA_BMM2_BASE_N"],
        "bmm2_base_k": effective["FA_BMM2_BASE_K"],
    }


def requested_fields(values: dict[str, int]) -> dict:
    return {
        "requested_s1_base": values["FA_S1_BASE"],
        "requested_s2_base": values["FA_S2_BASE"],
        "requested_n_ratio": values["FA_N_RATIO"],
        "requested_bmm_preset": values["FA_BMM_PRESET"],
    }


def make_params(shape_params: list[base.BaseParam], values: dict[str, int]) -> tuple[list[base.BaseParam], bool]:
    domains = {name: [value] for name, value in values.items()}
    validator = fa.make_validator("same_ab", domains)
    params = list(shape_params)
    params.extend(
        base.BaseParam(name=name, value=value, is_const=False, domain=[value])
        for name, value in values.items()
    )
    params = validator.get_all_params(params)
    return params, validator.is_valid(params)


def run_official(shape_params: list[base.BaseParam], key: int) -> dict:
    fa.clean_profile()
    output = Path("output/output.bin")
    golden = Path("output/golden.bin")
    trace = Path("output/official_effective.tsv")
    output.unlink(missing_ok=True)
    golden.unlink(missing_ok=True)
    trace.unlink(missing_ok=True)
    env = fa.run_env(shape_params, key)
    env["FA_SKIP_BUILD"] = "1"
    env["FA_EFFECTIVE_TILING_TRACE"] = str(trace.resolve())
    process = subprocess.run(["bash", "run.sh"], env=env, capture_output=True, text=True)
    samples = profile_samples()
    effective = fa.read_effective(trace)
    if process.returncode == 0 and output.is_file():
        output.replace(golden)
    status = "ok" if process.returncode == 0 and effective is not None and len(samples) == REPEAT and golden.is_file() else "failed"
    error = ""
    if status != "ok":
        lines = (process.stdout + process.stderr).strip().splitlines()
        error = lines[-1] if lines else "official baseline did not produce complete data"
    return {
        "effective": effective,
        "samples": samples,
        "duration": statistics.median(samples) if samples else math.inf,
        "exit_code": process.returncode,
        "status": status,
        "error": error,
    }


def run_candidate(params: list[base.BaseParam], key: int, requested: dict[str, int]) -> dict:
    fa.clean_profile()
    output = Path("output/output.bin")
    trace = Path("output/effective.tsv")
    output.unlink(missing_ok=True)
    trace.unlink(missing_ok=True)
    env = fa.run_env(params, key)
    env["FA_SKIP_BUILD"] = "1"
    env["FA_EFFECTIVE_TILING_TRACE"] = str(trace.resolve())
    process = subprocess.run(["bash", "run.sh"], env=env, capture_output=True, text=True)
    effective = fa.read_effective(trace)
    matched = effective is not None and all(effective.get(name) == value for name, value in requested.items())
    if matched:
        matched = fa.bmm_preset_matches(requested, effective)
    samples = profile_samples()
    correct, maximum, mean, ratio = correctness()
    if process.returncode != 0:
        status = "runner_failed"
    elif not matched:
        status = "tiling_mismatch"
    elif len(samples) != REPEAT:
        status = "missing_latency"
    elif not correct:
        status = "incorrect"
    else:
        status = "ok"
    error = ""
    if status != "ok":
        lines = (process.stdout + process.stderr).strip().splitlines()
        error = lines[-1] if lines else status
    return {
        "effective": effective,
        "samples": samples,
        "duration": statistics.median(samples) if samples else math.inf,
        "matched": matched,
        "correct": correct,
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "error_ratio": ratio,
        "exit_code": process.returncode,
        "status": status,
        "error": error,
    }


def configurations(official: dict[str, int]) -> list[tuple[str, dict[str, int]]]:
    values = {
        "FA_S1_BASE": official["FA_S1_BASE"],
        "FA_S2_BASE": official["FA_S2_BASE"],
        "FA_N_RATIO": official["FA_N_RATIO"],
        "FA_BMM_PRESET": 0,
    }
    result = [("official_explicit", values)]
    seen = {tuple(values[name] for name in fa.SEARCH_NAMES)}
    for name, s1, s2, ratio, preset in FIXED_CONFIGS:
        values = {
            "FA_S1_BASE": s1,
            "FA_S2_BASE": s2,
            "FA_N_RATIO": ratio,
            "FA_BMM_PRESET": preset,
        }
        signature = tuple(values[item] for item in fa.SEARCH_NAMES)
        if signature not in seen:
            result.append((name, values))
            seen.add(signature)
    return result


def base_row(split: str, case: str, priority: int, key: int, round_index: int, direction: str) -> dict:
    return {
        "version": VERSION,
        "split": split,
        "case": case,
        "round": round_index,
        "direction": direction,
        "npu_id": fa.NPU_ID,
        "soc_unit": os.environ.get("FA_SOC_UNIT", "ascend910b"),
        "ops_transformer_version": os.environ.get("OPS_TRANSFORMER_VERSION", "v8.5.0"),
        "priority": priority,
        "tiling_key": key,
        **shape_fields(),
    }


def collect_case(split: str, case: dict, path: Path, done: set[tuple[str, str, str, str]]) -> int:
    name = str(case["name"])
    fa.SHAPE = dict(zip(("B", "N1", "N2", "S1", "S2", "D", "DV", "DTYPE", "LAYOUT"), case["shape"]))
    shape_params = fa.shape_params()
    priority, key = fa.discover(shape_params)
    if priority != 95:
        raise RuntimeError(f"{name} selected route {priority}, expected SameAB route 95")
    fa.build(key)
    failures = 0
    for round_index in range(1, ROUNDS + 1):
        direction = "forward" if round_index == 1 else "reverse"
        start = run_official(shape_params, key)
        if start["status"] != "ok":
            raise RuntimeError(f"{name} official baseline failed: {start['error']}")
        official = start["effective"]
        configs = configurations(official)
        if direction == "reverse":
            configs = list(reversed(configs))
        baseline_id = f"baseline_start_{direction}"
        baseline_key = (name, str(round_index), "baseline", baseline_id)
        if baseline_key not in done:
            row = base_row(split, name, priority, key, round_index, direction)
            row.update(
                {
                    "record_type": "baseline",
                    "order": -1,
                    "config_id": baseline_id,
                    **requested_fields({item: official[item] for item in fa.SEARCH_NAMES}),
                    **effective_fields(official),
                    "duration_us": start["duration"],
                    "samples_us": json.dumps(start["samples"]),
                    "sample_count": len(start["samples"]),
                    "matched": True,
                    "correct": True,
                    "exit_code": start["exit_code"],
                    "status": start["status"],
                    "error": start["error"],
                }
            )
            append(path, row)
        for order, (config_id, values) in enumerate(configs):
            record_key = (name, str(round_index), "candidate", config_id)
            if record_key in done:
                print(f"SKIP split={split} case={name} round={round_index} config={config_id}", flush=True)
                continue
            params, valid = make_params(shape_params, values)
            row = base_row(split, name, priority, key, round_index, direction)
            row.update(
                {
                    "record_type": "candidate",
                    "order": order,
                    "config_id": config_id,
                    **requested_fields(values),
                }
            )
            if not valid:
                row.update({"status": "validator_reject", "error": "SameAB validator rejected configuration"})
                append(path, row)
                failures += 1
                print(f"REJECT split={split} case={name} round={round_index} config={config_id}", flush=True)
                continue
            measured = run_candidate(params, key, values)
            row.update(
                {
                    **effective_fields(measured["effective"]),
                    "duration_us": measured["duration"],
                    "samples_us": json.dumps(measured["samples"]),
                    "sample_count": len(measured["samples"]),
                    "matched": measured["matched"],
                    "correct": measured["correct"],
                    "max_abs_error": measured["max_abs_error"],
                    "mean_abs_error": measured["mean_abs_error"],
                    "error_ratio": measured["error_ratio"],
                    "exit_code": measured["exit_code"],
                    "status": measured["status"],
                    "error": measured["error"],
                }
            )
            append(path, row)
            if measured["status"] != "ok":
                failures += 1
            print(
                f"MEASURE split={split} case={name} round={round_index} config={config_id} "
                f"status={measured['status']} duration_us={measured['duration']}",
                flush=True,
            )
        end = run_official(shape_params, key)
        baseline_id = f"baseline_end_{direction}"
        baseline_key = (name, str(round_index), "baseline", baseline_id)
        if baseline_key not in done:
            row = base_row(split, name, priority, key, round_index, direction)
            row.update(
                {
                    "record_type": "baseline",
                    "order": len(configs),
                    "config_id": baseline_id,
                    **(
                        requested_fields({item: end["effective"][item] for item in fa.SEARCH_NAMES})
                        if end["effective"]
                        else {}
                    ),
                    **effective_fields(end["effective"]),
                    "duration_us": end["duration"],
                    "samples_us": json.dumps(end["samples"]),
                    "sample_count": len(end["samples"]),
                    "matched": end["effective"] is not None,
                    "correct": end["status"] == "ok",
                    "exit_code": end["exit_code"],
                    "status": end["status"],
                    "error": end["error"],
                }
            )
            append(path, row)
        if end["status"] != "ok":
            failures += 1
        drift = 100.0 * (end["duration"] / start["duration"] - 1.0) if start["duration"] > 0 else math.inf
        print(
            f"ROUND split={split} case={name} round={round_index} "
            f"baseline_start_us={start['duration']} baseline_end_us={end['duration']} drift_percent={drift}",
            flush=True,
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=4)
    args = parser.parse_args()
    fa.NPU_ID = args.id
    fa.PROFILE_WARMUP = WARMUP
    fa.PROFILE_REPEAT = REPEAT
    root = Path(__file__).resolve().parent
    os.chdir(root)
    calibration, holdout = prepare_output(root / "output" / VERSION)
    cases = {str(case["name"]): case for case in fa.CASES}
    failures = 0
    for split, names, path in (
        ("calibration", CALIBRATION_CASES, calibration),
        ("holdout", HOLDOUT_CASES, holdout),
    ):
        done = completed(path)
        for index, name in enumerate(names, 1):
            print(f"CASE split={split} index={index}/{len(names)} name={name}", flush=True)
            try:
                failures += collect_case(split, cases[name], path, done)
            except Exception as error:
                failures += 1
                print(f"CASE_FAILED split={split} name={name} error={error}", flush=True)
    print(f"CALIBRATION_DATA={calibration}")
    print(f"HOLDOUT_DATA={holdout}")
    print(f"FAILURES={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
