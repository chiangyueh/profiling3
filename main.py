# NEW BEGIN
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from tiling import base, estimator_algs, ga, limits, valids


SHAPE = {
    "B": 1,
    "N1": 128,
    "N2": 128,
    "S1": 4096,
    "S2": 4096,
    "D": 192,
    "DV": 128,
    "DTYPE": "BF16",
    "LAYOUT": "BNSD",
}
NPU_ID = 4
POPULATION_SIZE = 16
GENERATIONS = 32

PRIORITY = 95
TILING_KEY = 69864023600
DTYPES = {"FP32": (0, 4), "FP16": (1, 2), "BF16": (2, 2)}
LAYOUTS = {"BNSD": 0, "SBH": 1, "BSND": 2}


def shape_params() -> list[base.BaseParam]:
    dtype = str(SHAPE["DTYPE"]).upper()
    layout = str(SHAPE["LAYOUT"]).upper()
    if dtype not in DTYPES or layout not in LAYOUTS:
        raise ValueError("DTYPE must be FP32, FP16, or BF16; LAYOUT must be BNSD, SBH, or BSND")
    dtype_kind, dtype_bytes = DTYPES[dtype]
    values = {
        "FA_B": int(SHAPE["B"]),
        "FA_N1": int(SHAPE["N1"]),
        "FA_N2": int(SHAPE["N2"]),
        "FA_S1": int(SHAPE["S1"]),
        "FA_S2": int(SHAPE["S2"]),
        "FA_D": int(SHAPE["D"]),
        "FA_DV": int(SHAPE["DV"]),
        "FA_DTYPE_KIND": dtype_kind,
        "FA_DTYPE_BYTES": dtype_bytes,
        "FA_LAYOUT": LAYOUTS[layout],
    }
    return [base.BaseParam(name=name, value=value, is_const=True) for name, value in values.items()]


def run_env(params: list[base.BaseParam], key: int) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("FA_S1_BASE", "FA_S2_BASE", "FA_N_RATIO"):
        env.pop(name, None)
    env["FA_NPU_ID"] = str(NPU_ID)
    env["FA_TILING_KEY"] = str(key)
    for param in params:
        env[param.name] = str(param.value)
    return env


def build(key: int) -> None:
    env = dict(os.environ)
    env["FA_TILING_KEY"] = str(key)
    completed = subprocess.run(["bash", "build.sh"], env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"build failed for tiling key {key}")


def read_effective(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text().splitlines()):
        fields = line.split("\t")
        if len(fields) == 4 and fields[0] in {
            "FlashAttentionScoreTilingS1s2Bn2gs1",
            "FlashAttentionScoreTilingS1s2Bn2gs1SameAB",
        }:
            return {
                "FA_S1_BASE": int(fields[1]),
                "FA_S2_BASE": int(fields[2]),
                "FA_N_RATIO": int(fields[3]),
            }
    return None


def profile_time() -> float:
    profiles = list(Path(".").glob("OPPROF_*"))
    if not profiles:
        return float("inf")
    profile = max(profiles, key=lambda path: path.stat().st_mtime)
    table = profile / "OpBasicInfo.csv"
    if not table.is_file():
        return float("inf")
    import pandas as pd

    return float(pd.read_csv(table)["Task Duration(us)"].iloc[0])


def clean_profile() -> None:
    for path in Path(".").glob("OPPROF_*"):
        shutil.rmtree(path)


def official_baseline(params: list[base.BaseParam], key: int) -> float:
    clean_profile()
    output = Path("output/output.bin")
    golden = Path("output/golden.bin")
    output.unlink(missing_ok=True)
    golden.unlink(missing_ok=True)
    env = run_env(params, key)
    env["FA_SKIP_BUILD"] = "1"
    completed = subprocess.run(["bash", "run.sh"], env=env)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError("official FA baseline failed")
    duration = profile_time()
    output.replace(golden)
    if duration == float("inf"):
        raise RuntimeError("official FA latency was not produced")
    return duration


class FaProfileEstimator(estimator_algs.AlgoProfileEst):
    def __init__(self, *args, tiling_key: int, **kwargs) -> None:
        self.tiling_key = tiling_key
        self.last_correct = False
        super().__init__(*args, **kwargs)

    def _run_estimator(self, params: list[base.BaseParam]) -> float:
        clean_profile()
        Path("output/output.bin").unlink(missing_ok=True)
        trace = Path("output/effective.tsv")
        trace.unlink(missing_ok=True)
        env = run_env(params, self.tiling_key)
        env["FA_SKIP_BUILD"] = "1"
        env["FA_EFFECTIVE_TILING_TRACE"] = str(trace.resolve())
        requested = {
            param.name: param.value
            for param in params
            if param.name in {"FA_S1_BASE", "FA_S2_BASE", "FA_N_RATIO"}
        }
        completed = subprocess.run(
            ["bash", self.runner], env=env, capture_output=True, text=True
        )
        effective = read_effective(trace)
        matched = effective == requested
        duration = profile_time() if completed.returncode == 0 and matched else float("inf")
        print(
            f"CANDIDATE requested={requested} effective={effective} "
            f"exit={completed.returncode} duration_us={duration}"
        )
        if completed.returncode != 0:
            output = (completed.stdout + completed.stderr).strip().splitlines()
            if output:
                print(output[-1])
        return duration

    def _is_right(self, absolute_tol: float = 1e-9, error_tol: float = 1e-4) -> bool:
        self.last_correct = super()._is_right(absolute_tol, error_tol)
        print(f"CORRECT={self.last_correct}")
        return self.last_correct


class Ga(FaProfileEstimator, ga.GaAlgo):
    pass


class GaValidator(valids.attention.FlashAttentionScoreSameABValidator):
    def _make_param(
        self, name: str, value: int, is_const: bool, domain: list[int] | None = None
    ) -> ga.GaParam:
        return ga.GaParam(name=name, value=value, is_const=is_const, domain=domain or [value])


def domains() -> dict[str, list[int]]:
    return {
        "FA_S1_BASE": [64, 128, 192, 256],
        "FA_S2_BASE": [16, 32, 64, 96, 128, 160, 192, 256],
        "FA_N_RATIO": [1, 2, 4, 5, 6, 8, 12, 16],
    }


def main() -> None:
    global NPU_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=NPU_ID)
    args = parser.parse_args()
    NPU_ID = args.id
    os.chdir(Path(__file__).resolve().parent)
    params = shape_params()
    build(TILING_KEY)
    baseline = official_baseline(params, TILING_KEY)
    search_domains = domains()
    hardware = limits.AttentionLimits(
        max_cores=48,
        aic_num=24,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L1_size=512 * 1024,
        L2_size=192 * 1024**2,
        UB_size=192 * 1024,
        domains=search_domains,
        calc_type_size=2,
    )
    cache = Path("output/search_cache_ga_deepseek_v3_generate_s4096_bnsd_tp1.json")
    print(f"ROUTE=same_ab PRIORITY={PRIORITY} TILING_KEY={TILING_KEY}")
    print(f"ALGORITHM=GA PROPOSALS={2 * POPULATION_SIZE - 1 + (GENERATIONS - 1) * (POPULATION_SIZE - 1)}")
    print(f"OFFICIAL_US={baseline}")
    search = Ga(
        is_stop=lambda results: len(results) >= GENERATIONS,
        validator=GaValidator(hardware),
        input_params=params,
        pop_size=POPULATION_SIZE,
        runner="./run.sh",
        cache_path=str(cache),
        verbose=False,
        tiling_key=TILING_KEY,
    )
    results = search()
    best = min(results, key=lambda result: result.duration)
    best_values = {
        param.name: param.value
        for param in best.params
        if param.name in search_domains
    }
    print(f"BEST={best_values}")
    print(f"BEST_US={best.duration}")
    print(f"SPEEDUP={baseline / best.duration if best.duration > 0 else 0.0}")


if __name__ == "__main__":
    main()
# NEW END
