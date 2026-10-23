from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path

import main as fa


VERSION = "fa_route_latency_v1"
WARMUP = 3
REPEAT = 10
FIELDS = (
    "version",
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
    "effective_tiling",
    "duration_us",
    "samples_us",
    "sample_count",
    "status",
    "stage",
    "error",
)
CASES = (
    {
        "name": "route_same_ab",
        "expected": 95,
        "shape": (1, 8, 1, 128, 512, 96, 96, "FP16", "BNSD", 0),
    },
    {
        "name": "route_general",
        "expected": 96,
        "shape": (1, 8, 1, 128, 1536, 128, 128, "FP32", "BNSD", 0),
    },
    {
        "name": "route_s1",
        "expected": 97,
        "shape": (1, 8, 1, 128, 512, 128, 128, "FP16", "BNSD", 0),
    },
    {
        "name": "route_b",
        "expected": 98,
        "shape": (1, 1, 1, 16, 16, 64, 64, "FP16", "BNSD", 0),
    },
    {
        "name": "route_varlen",
        "expected": 94,
        "shape": (2, 1, 1, 128, 128, 64, 64, "FP32", "TND", 0),
    },
    {
        "name": "route_drop_adapter",
        "expected": 98,
        "shape": (1, 1, 1, 31, 65, 64, 64, "FP16", "BNSD", 1),
    },
)
ROUTE_NAMES = {94: "varlen", 95: "same_ab", 96: "general", 97: "s1", 98: "b"}


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
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    fa.NPU_ID = args.id
    fa.PROFILE_WARMUP = WARMUP
    fa.PROFILE_REPEAT = REPEAT
    output = root / "output" / VERSION / "routes.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
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
            "case": name,
            "expected_route": case["expected"],
            **shape_row(case),
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
            succeeded += 1
            print(
                f"CASE {index}/{len(CASES)} OK: {name} route={priority} key={key} "
                f"duration_us={row['duration_us']}",
                flush=True,
            )
        except Exception as error:
            failed += 1
            row.update({"status": "failed", "stage": stage, "error": str(error)})
            print(f"CASE {index}/{len(CASES)} SKIPPED: {name} error={error}", flush=True)
        append(output, row)
    print(f"ROUTE_DATA={output}")
    print(f"SUCCEEDED={succeeded} FAILED={failed}")


if __name__ == "__main__":
    main()
