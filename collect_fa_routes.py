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

import main as fa


VERSION = "fa_working_routes_latency_v1"
WARMUP = 3
REPEAT = 10
MODEL_PATH = Path("output/sameab_latency_model_v1/model.json")
FIELDS = (
    "version",
    "split",
    "family",
    "case",
    "expected_route",
    "route_attempts",
    "priority",
    "route",
    "tiling_key",
    "B",
    "N1",
    "N2",
    "S1",
    "S2",
    "D",
    "DV",
    "dtype",
    "layout",
    "has_drop_mask",
    "q_bytes",
    "k_bytes",
    "v_bytes",
    "output_bytes",
    "bmm1_flops",
    "bmm2_flops",
    "npu_id",
    "soc_unit",
    "warmup",
    "repeat",
    "effective_tiling",
    "duration_us",
    "samples_us",
    "sample_count",
    "model_version",
    "model_sha256",
    "predicted_baseline_us",
    "predicted_ratio",
    "predicted_us",
    "prediction_abs_error_us",
    "prediction_ape_percent",
    "prediction_status",
    "prediction_error",
    "status",
    "stage",
    "error",
)
CASES = (
    {
        "name": "sameab_external_s1536_bnsd_n16",
        "family": "same_ab",
        "split": "external",
        "expected": 95,
        "shape": (1, 16, 16, 1536, 1536, 192, 128, "BF16", "BNSD", 0),
    },
    {
        "name": "sameab_external_s2048_sbh_n16",
        "family": "same_ab",
        "split": "external",
        "expected": 95,
        "shape": (1, 16, 16, 2048, 2048, 192, 128, "BF16", "SBH", 0),
    },
    {
        "name": "sameab_external_s2560_bnsd_n32",
        "family": "same_ab",
        "split": "external",
        "expected": 95,
        "shape": (1, 32, 32, 2560, 2560, 192, 128, "BF16", "BNSD", 0),
    },
    {
        "name": "sameab_external_s3072_sbh_n32",
        "family": "same_ab",
        "split": "external",
        "expected": 95,
        "shape": (1, 32, 32, 3072, 3072, 192, 128, "BF16", "SBH", 0),
    },
    {
        "name": "sameab_external_s3584_bnsd_n64",
        "family": "same_ab",
        "split": "external",
        "expected": 95,
        "shape": (1, 64, 64, 3584, 3584, 192, 128, "BF16", "BNSD", 0),
    },
    {
        "name": "sameab_external_s5120_sbh_n32",
        "family": "same_ab",
        "split": "external",
        "expected": 95,
        "shape": (1, 32, 32, 5120, 5120, 192, 128, "BF16", "SBH", 0),
    },
    {
        "name": "sameab_external_s7168_bnsd_n16",
        "family": "same_ab",
        "split": "external",
        "expected": 95,
        "shape": (1, 16, 16, 7168, 7168, 192, 128, "BF16", "BNSD", 0),
    },
    {
        "name": "sameab_external_s10240_sbh_n16",
        "family": "same_ab",
        "split": "external",
        "expected": 95,
        "shape": (1, 16, 16, 10240, 10240, 192, 128, "BF16", "SBH", 0),
    },
    {
        "name": "s1_b1_n8_s64_s256",
        "family": "s1",
        "split": "calibration",
        "expected": 97,
        "shape": (1, 8, 1, 64, 256, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "s1_b1_n8_s96_s384",
        "family": "s1",
        "split": "calibration",
        "expected": 97,
        "shape": (1, 8, 1, 96, 384, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "s1_b1_n8_s128_s512",
        "family": "s1",
        "split": "calibration",
        "expected": 97,
        "shape": (1, 8, 1, 128, 512, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "s1_b1_n8_s192_s640",
        "family": "s1",
        "split": "calibration",
        "expected": 97,
        "shape": (1, 8, 1, 192, 640, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "s1_b1_n8_s256_s768",
        "family": "s1",
        "split": "calibration",
        "expected": 97,
        "shape": (1, 8, 1, 256, 768, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "s1_b2_n8_s128_s896",
        "family": "s1",
        "split": "calibration",
        "expected": 97,
        "shape": (2, 8, 1, 128, 896, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "s1_b1_n16_n2_s384_s1024",
        "family": "s1",
        "split": "holdout",
        "expected": 97,
        "shape": (1, 16, 2, 384, 1024, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "s1_b2_n16_n2_s256_s512",
        "family": "s1",
        "split": "holdout",
        "expected": 97,
        "shape": (2, 16, 2, 256, 512, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "b_b1_n1_s16_s16",
        "family": "b",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 1, 1, 16, 16, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "b_b1_n2_s16_s32",
        "family": "b",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 2, 1, 16, 32, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "b_b1_n4_s32_s32",
        "family": "b",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 4, 1, 32, 32, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "b_b1_n4_s32_s64",
        "family": "b",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 4, 1, 32, 64, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "b_b2_n8_s64_s64",
        "family": "b",
        "split": "calibration",
        "expected": 98,
        "shape": (2, 8, 1, 64, 64, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "b_b1_n4_n2_s128_s64",
        "family": "b",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 4, 2, 128, 64, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "b_b1_n8_s64_s128",
        "family": "b",
        "split": "holdout",
        "expected": 98,
        "shape": (1, 8, 1, 64, 128, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "b_b4_n2_s128_s64",
        "family": "b",
        "split": "holdout",
        "expected": 98,
        "shape": (4, 2, 1, 128, 64, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "drop_b1_n1_s16_s33",
        "family": "b_drop",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 1, 1, 16, 33, 64, 64, "FP16", "BNSD", 1),
    },
    {
        "name": "drop_b1_n1_s31_s65",
        "family": "b_drop",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 1, 1, 31, 65, 64, 64, "FP16", "BNSD", 1),
    },
    {
        "name": "drop_b1_n2_s32_s48",
        "family": "b_drop",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 2, 1, 32, 48, 64, 64, "FP16", "BNSD", 1),
    },
    {
        "name": "drop_b1_n4_s48_s80",
        "family": "b_drop",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 4, 1, 48, 80, 64, 64, "FP16", "BNSD", 1),
    },
    {
        "name": "drop_b2_n4_s64_s96",
        "family": "b_drop",
        "split": "calibration",
        "expected": 98,
        "shape": (2, 4, 1, 64, 96, 64, 64, "FP16", "BNSD", 1),
    },
    {
        "name": "drop_b1_n4_n2_s96_s64",
        "family": "b_drop",
        "split": "calibration",
        "expected": 98,
        "shape": (1, 4, 2, 96, 64, 64, 64, "FP16", "BNSD", 1),
    },
    {
        "name": "drop_b1_n8_s64_s128",
        "family": "b_drop",
        "split": "holdout",
        "expected": 98,
        "shape": (1, 8, 1, 64, 128, 64, 64, "FP16", "BNSD", 1),
    },
    {
        "name": "drop_b4_n2_s128_s64",
        "family": "b_drop",
        "split": "holdout",
        "expected": 98,
        "shape": (4, 2, 1, 128, 64, 64, 64, "FP16", "BNSD", 1),
    },
)
ROUTE_NAMES = {94: "varlen", 95: "same_ab", 96: "general", 97: "s1", 98: "b"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sameab_model(path: Path) -> tuple[dict | None, str]:
    if not path.is_file():
        return None, ""
    model = json.loads(path.read_text())
    if model.get("model_version") != "sameab_latency_model_v1":
        raise RuntimeError(f"unsupported SameAB model: {path}")
    return model, digest(path)


def derived_fields(case: dict) -> dict:
    values = shape_row(case)
    dtype_bytes = fa.DTYPES[str(values["dtype"]).upper()][1]
    b = values["B"]
    n1 = values["N1"]
    n2 = values["N2"]
    s1 = values["S1"]
    s2 = values["S2"]
    d = values["D"]
    dv = values["DV"]
    return {
        "q_bytes": b * n1 * s1 * d * dtype_bytes,
        "k_bytes": b * n2 * s2 * d * dtype_bytes,
        "v_bytes": b * n2 * s2 * dv * dtype_bytes,
        "output_bytes": b * n1 * s1 * dv * dtype_bytes,
        "bmm1_flops": 2 * b * n1 * s1 * s2 * d,
        "bmm2_flops": 2 * b * n1 * s1 * s2 * dv,
    }


def predict_sameab(row: dict, model: dict | None, model_sha256: str) -> dict:
    if row["family"] != "same_ab":
        return {"prediction_status": "not_applicable"}
    if model is None:
        return {"prediction_status": "model_missing", "prediction_error": str(MODEL_PATH)}
    scope = model["scope"]
    if (
        row["dtype"] != scope["dtype"]
        or int(row["D"]) != int(scope["D"])
        or int(row["DV"]) != int(scope["DV"])
        or row["layout"] not in scope["layouts"]
    ):
        return {"prediction_status": "outside_model_scope"}
    coefficient = float(
        model["baseline_model"]["coefficient_us_per_flop"][row["layout"]]
    )
    baseline = coefficient * (int(row["bmm1_flops"]) + int(row["bmm2_flops"]))
    ratio = 1.0
    predicted = baseline
    actual = float(row["duration_us"])
    return {
        "model_version": model["model_version"],
        "model_sha256": model_sha256,
        "predicted_baseline_us": baseline,
        "predicted_ratio": ratio,
        "predicted_us": predicted,
        "prediction_abs_error_us": abs(predicted - actual),
        "prediction_ape_percent": abs(predicted - actual) / actual * 100.0,
        "prediction_status": "ok",
    }


def write_prediction_report(path: Path, csv_path: Path, model_path: Path) -> None:
    if not csv_path.is_file():
        return
    rows = [
        row
        for row in csv.DictReader(csv_path.open(newline=""))
        if row.get("status") == "ok" and row.get("prediction_status") == "ok"
    ]
    if not rows:
        print("SAMEAB_PREDICTIONS=0")
        return
    errors = [float(row["prediction_ape_percent"]) for row in rows]
    absolute = [float(row["prediction_abs_error_us"]) for row in rows]
    ordered = sorted(errors)
    p90_index = math.ceil(0.9 * len(ordered)) - 1
    report = {
        "model": str(model_path),
        "model_sha256": rows[0]["model_sha256"],
        "count": len(rows),
        "mape_percent": statistics.mean(errors),
        "median_ape_percent": statistics.median(errors),
        "p90_ape_percent": ordered[p90_index],
        "max_ape_percent": max(errors),
        "mae_us": statistics.mean(absolute),
        "cases": [
            {
                "case": row["case"],
                "actual_us": float(row["duration_us"]),
                "predicted_us": float(row["predicted_us"]),
                "ape_percent": float(row["prediction_ape_percent"]),
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"SAMEAB_PREDICTION_REPORT={path}")
    print(
        f"SAMEAB_PREDICTION count={len(rows)} MAPE={report['mape_percent']:.3f}% "
        f"MEDIAN_APE={report['median_ape_percent']:.3f}% "
        f"P90_APE={report['p90_ape_percent']:.3f}% "
        f"MAX_APE={report['max_ape_percent']:.3f}%"
    )


def append(path: Path, row: dict) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDS})


def completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        row["case"]
        for row in csv.DictReader(path.open(newline=""))
        if row.get("status") == "ok"
    }


def error_text(process: subprocess.CompletedProcess[str]) -> str:
    lines = (process.stdout + process.stderr).strip().splitlines()
    return lines[-1] if lines else f"exit={process.returncode}"


def route_attempts(path: Path) -> list[dict[str, int]]:
    attempts = []
    if not path.is_file():
        return attempts
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) == 4 and fields[0] == "FlashAttentionScore":
            attempts.append(
                {"priority": int(fields[1]), "status": int(fields[2]), "tiling_key": int(fields[3])}
            )
    return attempts


def discover(params) -> tuple[list[dict[str, int]], int, int]:
    fa.build_host()
    trace = Path("output/route.tsv")
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.unlink(missing_ok=True)
    env = fa.run_env(params, None)
    env["FA_ROUTE_TRACE"] = str(trace.resolve())
    env["FA_DISCOVER_ONLY"] = "1"
    env["FA_PROFILE"] = "0"
    env["FA_SKIP_BUILD"] = "1"
    process = subprocess.run(["bash", "run.sh"], env=env, capture_output=True, text=True)
    attempts = route_attempts(trace)
    selected = [item for item in attempts if item["status"] == 0]
    if process.returncode != 0 or not selected:
        raise RuntimeError(error_text(process))
    terminal = selected[-1]
    return attempts, terminal["priority"], terminal["tiling_key"]


def latency_data(path: Path) -> list[float]:
    if not path.is_file():
        return []
    try:
        return [float(line) for line in path.read_text().splitlines() if line.strip()]
    except ValueError:
        return []


def measure(params, key: int) -> tuple[dict[str, int], list[float]]:
    output = Path("output/output.bin")
    trace = Path("output/route_effective.tsv")
    latency = Path("output/route_latency.tsv")
    output.unlink(missing_ok=True)
    trace.unlink(missing_ok=True)
    latency.unlink(missing_ok=True)
    env = fa.run_env(params, key)
    env["FA_SKIP_BUILD"] = "1"
    env["FA_PROFILE"] = "0"
    env["FA_EFFECTIVE_TILING_TRACE"] = str(trace.resolve())
    env["FA_LATENCY_TRACE"] = str(latency.resolve())
    process = subprocess.run(["bash", "run.sh"], env=env, capture_output=True, text=True)
    effective = fa.read_effective(trace)
    samples = latency_data(latency)
    if process.returncode != 0:
        raise RuntimeError(error_text(process))
    if effective is None:
        raise RuntimeError("effective tiling trace was not produced")
    if len(samples) != REPEAT:
        raise RuntimeError(f"expected {REPEAT} latency samples, got {len(samples)}")
    if not output.is_file():
        raise RuntimeError("output was not produced")
    return effective, samples


def shape_row(case: dict) -> dict:
    names = ("B", "N1", "N2", "S1", "S2", "D", "DV", "dtype", "layout", "has_drop_mask")
    return dict(zip(names, case["shape"]))


def select_shape(case: dict) -> None:
    values = shape_row(case)
    fa.SHAPE = {
        "B": values["B"],
        "N1": values["N1"],
        "N2": values["N2"],
        "S1": values["S1"],
        "S2": values["S2"],
        "D": values["D"],
        "DV": values["DV"],
        "DTYPE": values["dtype"],
        "LAYOUT": values["layout"],
        "HAS_DROP": values["has_drop_mask"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=4)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    fa.NPU_ID = args.id
    fa.PROFILE_WARMUP = WARMUP
    fa.PROFILE_REPEAT = REPEAT
    output = root / "output" / VERSION / "routes.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.model.resolve()
    try:
        model, model_sha256 = load_sameab_model(model_path)
    except Exception as error:
        model = None
        model_sha256 = ""
        print(f"SAMEAB_MODEL_ERROR={error}", flush=True)
    print(f"CASES={len(CASES)} DATASET={output}", flush=True)
    print(
        f"SAMEAB_MODEL={model_path if model is not None else 'missing'}",
        flush=True,
    )
    done = completed(output)
    succeeded = 0
    failed = 0
    for index, case in enumerate(CASES, 1):
        name = str(case["name"])
        if name in done:
            succeeded += 1
            print(f"SKIP {index}/{len(CASES)} case={name} status=ok", flush=True)
            continue
        row = {
            "version": VERSION,
            "split": case["split"],
            "family": case["family"],
            "case": name,
            "expected_route": case["expected"],
            **shape_row(case),
            **derived_fields(case),
            "npu_id": args.id,
            "soc_unit": os.environ.get("FA_SOC_UNIT", "ascend910b"),
            "warmup": WARMUP,
            "repeat": REPEAT,
        }
        print(f"CASE {index}/{len(CASES)} START: {name}", flush=True)
        stage = "shape"
        try:
            select_shape(case)
            params = fa.shape_params()
            stage = "discover"
            attempts, priority, key = discover(params)
            row.update(
                {
                    "route_attempts": json.dumps(attempts, separators=(",", ":")),
                    "priority": priority,
                    "route": ROUTE_NAMES.get(priority, "unknown"),
                    "tiling_key": key,
                }
            )
            if priority != case["expected"]:
                raise RuntimeError(f"selected route {priority}, expected {case['expected']}")
            stage = "build"
            fa.build(key)
            stage = "measure"
            effective, samples = measure(params, key)
            row.update(
                {
                    "effective_tiling": json.dumps(effective, sort_keys=True, separators=(",", ":")),
                    "duration_us": statistics.median(samples),
                    "samples_us": json.dumps(samples, separators=(",", ":")),
                    "sample_count": len(samples),
                    "status": "ok",
                    "stage": "complete",
                }
            )
            row.update(predict_sameab(row, model, model_sha256))
            succeeded += 1
            message = (
                f"CASE {index}/{len(CASES)} OK: {name} route={priority} key={key} "
                f"duration_us={row['duration_us']}"
            )
            if row.get("prediction_status") == "ok":
                message += (
                    f" predicted_us={float(row['predicted_us']):.6f} "
                    f"ape={float(row['prediction_ape_percent']):.3f}%"
                )
            print(message, flush=True)
        except Exception as error:
            failed += 1
            row.update({"status": "failed", "stage": stage, "error": str(error)})
            print(f"CASE {index}/{len(CASES)} SKIPPED: {name} error={error}", flush=True)
        append(output, row)
    print(f"ROUTE_DATA={output}")
    print(f"SUCCEEDED={succeeded} FAILED={failed}")
    write_prediction_report(
        output.parent / "sameab_prediction_report.json", output, model_path
    )
    from fit_fa_latency import build_models

    build_models([output], root / "output" / "fa_latency_model_v1", model_path)


if __name__ == "__main__":
    main()
