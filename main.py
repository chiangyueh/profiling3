from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from tiling import base, estimator_algs, ga, limits, valids


CASES = [
    {
        "name": "deepseek_v3_generate_s4096_bnsd_tp1",
        "shape": (1, 128, 128, 4096, 4096, 192, 128, "BF16", "BNSD"),
    },
    {
        "name": "deepseek_v3_pipeline_s4096_sbh_tp1",
        "shape": (1, 128, 128, 4096, 4096, 192, 128, "BF16", "SBH"),
    },
    {
        "name": "deepseek_v3_pretrain_s4096_sbh_tp2",
        "shape": (1, 64, 64, 4096, 4096, 192, 128, "BF16", "SBH"),
    },
    {
        "name": "deepseek_v3_tune_s4096_bnsd_tp2",
        "shape": (1, 64, 64, 4096, 4096, 192, 128, "BF16", "BNSD"),
    },
    {
        "name": "deepseek_v3_pretrain_s4096_sbh_tp4",
        "shape": (1, 32, 32, 4096, 4096, 192, 128, "BF16", "SBH"),
    },
    {
        "name": "deepseek_v3_surrogate_s4096_bnsd_tp8",
        "shape": (1, 16, 16, 4096, 4096, 192, 128, "BF16", "BNSD"),
    },
    {
        "name": "deepseek_v3_bf16_s8192_sbh_tp1",
        "shape": (1, 128, 128, 8192, 8192, 192, 128, "BF16", "SBH"),
    },
    {
        "name": "pangu_ultra_dense_s4096_b1_model_proxy",
        "shape": (1, 96, 8, 4096, 4096, 128, 128, "BF16", "BNSD"),
    },
    {
        "name": "pangu_ultra_dense_s8192_b1_model_proxy",
        "shape": (1, 96, 8, 8192, 8192, 128, 128, "BF16", "BNSD"),
    },
]
SHAPE = {}
NPU_ID = 4
POPULATION_SIZE = 16
GENERATIONS = 32
MACRO_TOP_K = 4
FINAL_TOP_K = 3
FINAL_ROUNDS = 2
PROFILE_WARMUP = 2
PROFILE_REPEAT = 5

DTYPES = {"FP32": (0, 4), "FP16": (1, 2), "BF16": (2, 2)}
LAYOUTS = {"BNSD": 0, "SBH": 1, "BSND": 2}
ROUTES = {95: "same_ab", 96: "general"}
SEARCH_NAMES = ("FA_S1_BASE", "FA_S2_BASE", "FA_N_RATIO", "FA_BMM_PRESET")
BMM_PRESETS = {
    0: "official",
    1: "bmm1_64x128",
    2: "bmm1_128x64",
    3: "bmm2_64x128",
    4: "bmm2_128x64",
    5: "bmm1_bmm2_64x128",
    6: "bmm1_bmm2_128x64",
}


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


def run_env(params: list[base.BaseParam], key: int | None) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("FA_DISCOVER_ONLY", None)
    for name in SEARCH_NAMES:
        env.pop(name, None)
    env["FA_NPU_ID"] = str(NPU_ID)
    env["FA_WARMUP"] = str(PROFILE_WARMUP)
    env["FA_REPEAT"] = str(PROFILE_REPEAT)
    if key is None:
        env.pop("FA_TILING_KEY", None)
    else:
        env["FA_TILING_KEY"] = str(key)
    for param in params:
        env[param.name] = str(param.value)
    return env


def build(key: int) -> None:
    env = dict(os.environ)
    env.pop("FA_DISCOVER_ONLY", None)
    env["FA_TILING_KEY"] = str(key)
    completed = subprocess.run(["bash", "build.sh"], env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"build failed for tiling key {key}")


def build_host() -> None:
    env = dict(os.environ)
    env["FA_DISCOVER_ONLY"] = "1"
    env.pop("FA_TILING_KEY", None)
    completed = subprocess.run(["bash", "build.sh"], env=env)
    if completed.returncode != 0:
        raise RuntimeError("Host tiling build failed")


def read_route(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise RuntimeError("Host route trace was not produced")
    for line in reversed(path.read_text().splitlines()):
        fields = line.split("\t")
        if len(fields) == 4 and fields[0] == "FlashAttentionScore" and fields[2] == "0":
            return int(fields[1]), int(fields[3])
    raise RuntimeError("official Host tiling did not select a route")


def discover(params: list[base.BaseParam]) -> tuple[int, int]:
    build_host()
    trace = Path("output/route.tsv")
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.unlink(missing_ok=True)
    env = run_env(params, None)
    env["FA_ROUTE_TRACE"] = str(trace.resolve())
    env["FA_DISCOVER_ONLY"] = "1"
    env["FA_PROFILE"] = "0"
    env["FA_SKIP_BUILD"] = "1"
    completed = subprocess.run(
        ["bash", "run.sh"], env=env, capture_output=True, text=True
    )
    try:
        return read_route(trace)
    except RuntimeError:
        output = (completed.stdout + completed.stderr).strip().splitlines()
        if output:
            print(output[-1])
        raise RuntimeError("official Host tiling discovery failed")


def read_effective(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text().splitlines()):
        fields = line.split("\t")
        if len(fields) == 11 and fields[0] in {
            "FlashAttentionScoreTilingS1s2Bn2gs1",
            "FlashAttentionScoreTilingS1s2Bn2gs1SameAB",
        }:
            return {
                "FA_S1_BASE": int(fields[1]),
                "FA_S2_BASE": int(fields[2]),
                "FA_N_RATIO": int(fields[3]),
                "FA_BMM_PRESET": int(fields[4]),
                "FA_BMM1_BASE_M": int(fields[5]),
                "FA_BMM1_BASE_N": int(fields[6]),
                "FA_BMM1_BASE_K": int(fields[7]),
                "FA_BMM2_BASE_M": int(fields[8]),
                "FA_BMM2_BASE_N": int(fields[9]),
                "FA_BMM2_BASE_K": int(fields[10]),
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

    values = pd.to_numeric(pd.read_csv(table)["Task Duration(us)"], errors="coerce").dropna().tolist()
    if not values:
        return float("inf")
    measured = values[-min(PROFILE_REPEAT, len(values)):]
    return float(statistics.median(measured))


def clean_profile() -> None:
    for path in Path(".").glob("OPPROF_*"):
        shutil.rmtree(path)


def official_baseline(params: list[base.BaseParam], key: int) -> tuple[float, dict[str, int]]:
    clean_profile()
    output = Path("output/output.bin")
    golden = Path("output/golden.bin")
    trace = Path("output/official_effective.tsv")
    output.unlink(missing_ok=True)
    golden.unlink(missing_ok=True)
    trace.unlink(missing_ok=True)
    env = run_env(params, key)
    env["FA_SKIP_BUILD"] = "1"
    env["FA_EFFECTIVE_TILING_TRACE"] = str(trace.resolve())
    completed = subprocess.run(["bash", "run.sh"], env=env)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError("official FA baseline failed")
    duration = profile_time()
    effective = read_effective(trace)
    output.replace(golden)
    if duration == float("inf"):
        raise RuntimeError("official FA latency was not produced")
    if effective is None:
        raise RuntimeError("official FA effective tiling was not produced")
    return duration, effective


def bmm_preset_matches(requested: dict[str, int], effective: dict[str, int]) -> bool:
    preset = requested["FA_BMM_PRESET"]
    if preset == 0:
        return True
    s1 = min(requested["FA_S1_BASE"], int(SHAPE["S1"]))
    s1 = (s1 + 15) // 16 * 16
    window = min(
        requested["FA_S2_BASE"] * requested["FA_N_RATIO"],
        int(SHAPE["S2"]),
    )
    window = (window + 15) // 16 * 16
    dv = (int(SHAPE["DV"]) + 15) // 16 * 16
    if preset in (1, 2, 5, 6):
        expected_m = min(64 if preset in (1, 5) else 128, s1)
        expected_n = min(64 if preset in (2, 6) else 128, window)
        if effective["FA_BMM1_BASE_M"] != expected_m or effective["FA_BMM1_BASE_N"] != expected_n:
            return False
    if preset in (3, 4, 5, 6):
        expected_m = min(64 if preset in (3, 5) else 128, s1)
        expected_n = min(64 if preset in (4, 6) else 128, dv)
        if effective["FA_BMM2_BASE_M"] != expected_m or effective["FA_BMM2_BASE_N"] != expected_n:
            return False
    return True


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
        env["ATTENTION_EFFECTIVE_TILING_TRACE"] = str(trace.resolve())
        requested = {
            param.name: param.value
            for param in params
            if param.name in SEARCH_NAMES
        }
        completed = subprocess.run(
            ["bash", self.runner], env=env, capture_output=True, text=True
        )
        effective = read_effective(trace)
        matched = effective is not None and all(
            effective.get(name) == value for name, value in requested.items()
        )
        if matched:
            matched = bmm_preset_matches(requested, effective)
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
    def __init__(self, *args, **kwargs) -> None:
        self.observations: dict[str, base.BaseResult] = {}
        super().__init__(*args, **kwargs)

    def _duration(self, params: list[base.BaseParam]) -> base.BaseResult:
        result = super()._duration(params)
        self.observations[self._key(result.params)] = result
        return result


class GaValidator(valids.attention.FlashAttentionScoreSameABValidator):
    def _make_param(
        self, name: str, value: int, is_const: bool, domain: list[int] | None = None
    ) -> ga.GaParam:
        return ga.GaParam(name=name, value=value, is_const=is_const, domain=domain or [value])


class GaGeneralValidator(valids.attention.FlashAttentionScoreGeneralValidator):
    def _make_param(
        self, name: str, value: int, is_const: bool, domain: list[int] | None = None
    ) -> ga.GaParam:
        return ga.GaParam(name=name, value=value, is_const=is_const, domain=domain or [value])


def bmm_preset_domain(route: str) -> list[int]:
    dtype_bytes = DTYPES[str(SHAPE["DTYPE"])][1]
    if dtype_bytes == 4:
        return [0]
    d = int(SHAPE["D"])
    dv = int(SHAPE["DV"])
    layout = str(SHAPE["LAYOUT"])
    unsplit_k = (
        route == "same_ab"
        and layout == "BNSD"
        and d == 192
        and int(SHAPE["S1"]) % 256 == 0
        and int(SHAPE["S2"]) % 128 == 0
    )
    bmm1 = 64 < d <= 128 or unsplit_k
    bmm2 = 64 < dv <= 128 or unsplit_k
    presets = [0]
    if bmm1:
        presets.extend((1, 2))
    if bmm2:
        presets.extend((3, 4))
    if bmm1 and bmm2:
        presets.extend((5, 6))
    return presets


def domains(
    route: str,
    official: dict[str, int],
    presets: list[int],
) -> dict[str, list[int]]:
    s1 = [64, 128, 192, 256, 384] if route == "same_ab" else [16, 32, 64, 96, 128, 160, 192, 256]
    s2 = [16, 32, 64, 96, 128, 160, 192, 256]
    ratio = [1, 2, 4, 5, 6, 8, 12, 16]
    s1.append(official["FA_S1_BASE"])
    s2.append(official["FA_S2_BASE"])
    ratio.append(official["FA_N_RATIO"])
    return {
        "FA_S1_BASE": sorted(set(s1)),
        "FA_S2_BASE": sorted(set(s2)),
        "FA_N_RATIO": sorted(set(ratio)),
        "FA_BMM_PRESET": sorted(set(presets)),
    }


def make_limits(search_domains: dict[str, list[int]]) -> limits.AttentionLimits:
    return limits.AttentionLimits(
        max_cores=48,
        aic_num=24,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L1_size=512 * 1024,
        L2_size=192 * 1024**2,
        UB_size=192 * 1024,
        domains=search_domains,
        calc_type_size=DTYPES[str(SHAPE["DTYPE"])][1],
    )


def make_validator(route: str, search_domains: dict[str, list[int]]):
    validator_type = GaValidator if route == "same_ab" else GaGeneralValidator
    return validator_type(make_limits(search_domains))


def cache_path(selected: str, tiling_key: int) -> Path:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path("fa_launcher.cpp"),
        Path("patches/fa_candidate_hook.patch"),
        Path("tiling/validators/attention/AttentionValidator.py"),
        Path("tiling/validators/attention/FlashAttentionScoreValidator.py"),
    ):
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(
            {
                "case": selected,
                "shape": SHAPE,
                "tiling_key": tiling_key,
                "npu": NPU_ID,
                "soc": os.environ.get("FA_SOC_UNIT", "ascend910b"),
                "warmup": PROFILE_WARMUP,
                "repeat": PROFILE_REPEAT,
            },
            sort_keys=True,
        ).encode()
    )
    path = Path("output/search_cache") / selected / f"{digest.hexdigest()[:20]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def search_values(result: base.BaseResult) -> dict[str, int]:
    return {param.name: param.value for param in result.params if param.name in SEARCH_NAMES}


def top_results(results: list[base.BaseResult], count: int) -> list[base.BaseResult]:
    unique: dict[tuple[int, ...], base.BaseResult] = {}
    for result in results:
        if result.duration == float("inf"):
            continue
        values = search_values(result)
        if any(name not in values for name in SEARCH_NAMES):
            continue
        key = tuple(values[name] for name in SEARCH_NAMES)
        if key not in unique or result.duration < unique[key].duration:
            unique[key] = result
    return sorted(unique.values(), key=lambda result: result.duration)[:count]


def official_result(
    params: list[base.BaseParam],
    validator,
    official: dict[str, int],
    duration: float,
) -> base.BaseResult:
    values = {name: official[name] for name in SEARCH_NAMES}
    candidate = list(params)
    for name in SEARCH_NAMES:
        domain = validator.limits.domains[name]
        candidate.append(ga.GaParam(name=name, value=values[name], is_const=False, domain=domain))
    candidate = validator.get_all_params(candidate)
    return base.BaseResult(duration=duration, params=candidate)


def run_all() -> None:
    failures = []
    for index, case in enumerate(CASES, 1):
        name = str(case["name"])
        print(f"FA AUTOTILING {index}/{len(CASES)} START: {name}", flush=True)
        env = dict(os.environ)
        env["FA_MODEL_CASE"] = name
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--id", str(NPU_ID)], env=env
        )
        status = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"FA AUTOTILING {index}/{len(CASES)} {status}: {name}", flush=True)
        if completed.returncode != 0:
            failures.append(name)
    if failures:
        print(f"FA AUTOTILING FAILED: {failures}")
        raise SystemExit(1)
    print("FA AUTOTILING COMPLETE")


def main() -> None:
    global NPU_ID, SHAPE
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=NPU_ID)
    args = parser.parse_args()
    NPU_ID = args.id
    os.chdir(Path(__file__).resolve().parent)
    selected = os.environ.get("FA_MODEL_CASE")
    if selected is None:
        run_all()
        return
    case = next((item for item in CASES if item["name"] == selected), None)
    if case is None:
        raise ValueError(f"unknown FA_MODEL_CASE: {selected}")
    shape = case["shape"]
    SHAPE = dict(zip(("B", "N1", "N2", "S1", "S2", "D", "DV", "DTYPE", "LAYOUT"), shape))
    params = shape_params()
    priority, tiling_key = discover(params)
    if priority not in ROUTES:
        raise RuntimeError(
            f"selected FA Host route {priority} is not supported; supported routes are 95 and 96"
        )
    route = ROUTES[priority]
    build(tiling_key)
    baseline, official = official_baseline(params, tiling_key)
    macro_domains = domains(route, official, [0])
    macro_validator = make_validator(route, macro_domains)
    cache = cache_path(selected, tiling_key)
    print(f"MODEL={selected} SHAPE={shape}")
    print(f"ROUTE={route} PRIORITY={priority} TILING_KEY={tiling_key}")
    print(f"OFFICIAL_TILING={official}")
    print(
        "SEARCH=HIERARCHICAL_GA "
        f"MACRO_PROPOSALS={2 * POPULATION_SIZE - 1 + (GENERATIONS - 1) * (POPULATION_SIZE - 1)} "
        f"MACRO_TOP_K={MACRO_TOP_K} BMM_PRESETS={bmm_preset_domain(route)}"
    )
    print(f"OFFICIAL_US={baseline}")
    search = Ga(
        is_stop=lambda results: len(results) >= GENERATIONS,
        validator=macro_validator,
        input_params=params,
        pop_size=POPULATION_SIZE,
        runner="./run.sh",
        cache_path=str(cache),
        verbose=False,
        tiling_key=tiling_key,
    )
    search()
    candidates = list(search.observations.values())
    candidates.append(official_result(params, macro_validator, official, baseline))
    macro_best = top_results(candidates, MACRO_TOP_K)
    if not macro_best:
        raise RuntimeError("macro tiling search produced no valid candidate")

    presets = bmm_preset_domain(route)
    for macro in macro_best:
        values = search_values(macro)
        refinement_domains = {
            "FA_S1_BASE": [values["FA_S1_BASE"]],
            "FA_S2_BASE": [values["FA_S2_BASE"]],
            "FA_N_RATIO": [values["FA_N_RATIO"]],
            "FA_BMM_PRESET": presets,
        }
        refinement_validator = make_validator(route, refinement_domains)
        refinement = FaProfileEstimator(
            is_stop=lambda results: True,
            validator=refinement_validator,
            input_params=params,
            runner="./run.sh",
            cache_path=str(cache),
            verbose=False,
            tiling_key=tiling_key,
        )
        combinations = refinement_validator.get_combinations(len(presets), params)
        for combination in combinations:
            candidates.append(refinement._duration(combination))

    finalists = top_results(candidates, FINAL_TOP_K)
    if not finalists:
        raise RuntimeError("BMM refinement produced no valid candidate")
    final_domains = domains(route, official, presets)
    final_estimator = FaProfileEstimator(
        is_stop=lambda results: True,
        validator=make_validator(route, final_domains),
        input_params=params,
        runner="./run.sh",
        cache_path=str(cache),
        verbose=False,
        tiling_key=tiling_key,
    )
    verified = []
    for finalist in finalists:
        round_baselines = []
        round_durations = []
        for round_index in range(1, FINAL_ROUNDS + 1):
            round_baseline, _ = official_baseline(params, tiling_key)
            duration = final_estimator._run_estimator(finalist.params)
            correct = duration != float("inf") and final_estimator._is_right()
            print(
                f"FINAL_VERIFY round={round_index} requested={search_values(finalist)} "
                f"official_us={round_baseline} candidate_us={duration} correct={correct}"
            )
            if not correct:
                round_durations = []
                break
            round_baselines.append(round_baseline)
            round_durations.append(duration)
        if round_durations:
            verified.append(
                (
                    statistics.median(round_baselines),
                    statistics.median(round_durations),
                    statistics.median(
                        official_us / candidate_us
                        for official_us, candidate_us in zip(round_baselines, round_durations)
                    ),
                    finalist,
                )
            )
    if not verified:
        raise RuntimeError("no finalist passed repeated correctness verification")
    final_baseline, final_duration, speedup, best = max(verified, key=lambda item: item[2])
    best_values = search_values(best)
    print(f"BEST={best_values}")
    print(f"BEST_BMM_PRESET={BMM_PRESETS[best_values['FA_BMM_PRESET']]}")
    print(f"FINAL_OFFICIAL_US={final_baseline}")
    print(f"BEST_US={final_duration}")
    print(f"SPEEDUP={speedup}")


if __name__ == "__main__":
    main()
