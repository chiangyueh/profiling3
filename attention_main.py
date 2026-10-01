from __future__ import annotations

import __future__
import argparse
from collections import Counter
from dataclasses import dataclass
import importlib.abc
import importlib.machinery
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from typing import Iterator

import numpy as np
import pandas as pd


if sys.version_info < (3, 10):
    class _PostponedAnnotationsLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            source_path = self.get_filename(fullname)
            return self.source_to_code(self.get_data(source_path), source_path)

        def source_to_code(self, data, path, *, _optimize=-1):
            return compile(
                data,
                path,
                "exec",
                flags=__future__.annotations.compiler_flag,
                dont_inherit=True,
                optimize=_optimize,
            )


    class _TilingFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "tiling" and not fullname.startswith("tiling."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                spec.loader = _PostponedAnnotationsLoader(fullname, spec.loader.path)
            return spec


    sys.meta_path.insert(0, _TilingFinder())

from tiling import base, bf, estimator_algs, limits, valids


FA = "flash_attention_score"
FAG = "flash_attention_score_grad"
FA_SEED_KEY = 1
FAG_SEED_KEY = 74804

DEFAULT_FEATURES = {
    "DTYPE_BYTES": 4,
    "LAYOUT": valids.attention.LAYOUT_BNSD,
    "DETERMINISTIC": 0,
    "SPARSE_MODE": 0,
    "IS_SPARSE": 0,
    "HAS_MASK": 0,
    "HAS_PSE": 0,
    "HAS_DROP": 0,
    "HAS_ROPE": 0,
    "HAS_SINK": 0,
    "HAS_ACTUAL_SEQ": 0,
    "HAS_START_IDX": 0,
    "EQUAL_ACTUAL_SEQ": 1,
}


@dataclass(frozen=True)
class RouteCase:
    name: str
    operator: str
    priority: int
    shape: tuple[int, int, int, int, int, int, int]
    features: dict[str, int]
    terminal_priority: int | None = None

    @property
    def grad(self) -> bool:
        return self.operator == FAG

    @property
    def seed_key(self) -> int:
        return FAG_SEED_KEY if self.grad else FA_SEED_KEY

    @property
    def expected_terminal_priority(self) -> int:
        return self.terminal_priority if self.terminal_priority is not None else self.priority


def _features(**updates: int) -> dict[str, int]:
    values = dict(DEFAULT_FEATURES)
    values.update(updates)
    return values


# One source-derived representative shape for every registered ascend910b
# terminal route. Priority 90 is an adapter: its dedicated case passes through
# the adapter and then terminates in priority 98.
ROUTE_CASES = (
    RouteCase("fa_drop_adapter", FA, 90, (1, 1, 1, 16, 33, 64, 64),
              _features(DTYPE_BYTES=2, HAS_DROP=1), terminal_priority=98),
    RouteCase("fa_varlen", FA, 94, (2, 1, 1, 128, 128, 64, 64),
              _features(LAYOUT=valids.attention.LAYOUT_TND, HAS_ACTUAL_SEQ=1)),
    RouteCase("fa_same_ab", FA, 95, (1, 8, 1, 128, 512, 96, 96),
              _features(DTYPE_BYTES=2)),
    RouteCase("fa_s1s2", FA, 96, (1, 8, 1, 128, 1536, 128, 128), _features()),
    RouteCase("fa_s1", FA, 97, (1, 8, 1, 128, 512, 128, 128),
              _features(DTYPE_BYTES=2)),
    RouteCase("fa_b", FA, 98, (1, 1, 1, 16, 16, 64, 64), _features(DTYPE_BYTES=2)),
    RouteCase("fag_deterministic_bn2", FAG, 1000, (1, 1, 1, 128, 128, 128, 128),
              _features(DETERMINISTIC=1)),
    RouteCase("fag_mla", FAG, 1001, (2, 1, 1, 128, 128, 64, 64),
              _features(DTYPE_BYTES=2, LAYOUT=valids.attention.LAYOUT_TND, HAS_ACTUAL_SEQ=1)),
    RouteCase("fag_basic_deterministic", FAG, 1002, (1, 1, 1, 1024, 512, 64, 64),
              _features(DTYPE_BYTES=2, LAYOUT=valids.attention.LAYOUT_TND,
                        DETERMINISTIC=1, HAS_ACTUAL_SEQ=1)),
    RouteCase("fag_same_ab_deterministic", FAG, 1100, (1, 1, 1, 1024, 512, 64, 64),
              _features(DTYPE_BYTES=2, DETERMINISTIC=1)),
    RouteCase("fag_unpadded", FAG, 2000, (2, 1, 1, 128, 128, 64, 64),
              _features(LAYOUT=valids.attention.LAYOUT_TND, HAS_ACTUAL_SEQ=1)),
    RouteCase("fag_b", FAG, 10000, (1, 1, 1, 16, 16, 64, 64), _features(DTYPE_BYTES=2)),
    RouteCase("fag_n2", FAG, 11000, (40, 32, 32, 16, 16, 128, 128),
              _features(DTYPE_BYTES=2)),
    RouteCase("fag_bn2", FAG, 15000, (32, 2, 1, 64, 64, 128, 128),
              _features(DTYPE_BYTES=2)),
    RouteCase("fag_same_ab", FAG, 15500, (1, 4, 1, 2048, 512, 128, 128),
              _features(DTYPE_BYTES=2)),
    RouteCase("fag_generic", FAG, 16000, (1, 8, 1, 128, 1536, 128, 128), _features()),
)

ROUTE_CLASSES = {
    "fa_drop_adapter": "FlashAttentionScoreTilingDropMask",
    "fa_varlen": "FlashAttentionVarLenScoreTiling",
    "fa_same_ab": "FlashAttentionScoreTilingS1s2Bn2gs1SameAB",
    "fa_s1s2": "FlashAttentionScoreTilingS1s2Bn2gs1",
    "fa_s1": "FlashAttentionScoreTilingS1Bn2gs1",
    "fa_b": "FlashAttentionScoreTilingB",
    "fag_deterministic_bn2": "FlashAttentionScoreGradTilingDeterministic",
    "fag_mla": "FlashAttentionScoreGraTilingMla",
    "fag_basic_deterministic": "FlashAttentionScoreGraTilingBasicDet",
    "fag_same_ab_deterministic": "FlashAttentionScoreGradTilingSameABDeterministic",
    "fag_unpadded": "FlashAttentionScoreGradTilingUnpaddedAttension",
    "fag_b": "FlashAttentionScoreGradUbngs1s2BbTiling",
    "fag_n2": "FlashAttentionScoreGradUngs1s2BbnTiling",
    "fag_bn2": "FlashAttentionScoreGradTilingS1s2Bn2",
    "fag_same_ab": "FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb",
    "fag_generic": "FlashAttentionScoreGradTilingS1s2Bn2gs1s2",
}


def get_domains(route: str) -> dict[str, list[int]]:
    if route in ("fa_drop_adapter", "fa_b"):
        return {
            "FA_S1_BASE": [16, 32],
            "FA_S2_BASE": [16, 48, 64],
            "FA_N_RATIO": [1, 2],
        }
    if route in ("fa_s1", "fa_same_ab"):
        return {
            "FA_S1_BASE": [64, 128, 256],
            "FA_S2_BASE": [64, 128, 1024],
            "FA_N_RATIO": [1, 2, 8],
        }
    if route in ("fa_s1s2", "fa_varlen"):
        return {
            "FA_S1_BASE": [64, 128, 256],
            "FA_S2_BASE": [64, 128, 2048],
            "FA_N_RATIO": [1, 4, 16],
        }
    if route in ("fag_same_ab", "fag_same_ab_deterministic"):
        return {
            "FAG_S1_CV_INNER": [256, 512, 1024],
            "FAG_S2_CV_INNER": [256, 512, 1024],
        }
    if route in ("fag_generic", "fag_unpadded"):
        return {
            "FAG_S1_INNER": list(range(16, 129, 16)),
            "FAG_S2_INNER": list(range(16, 65, 16)),
            "FAG_S1_CV_RATIO": [1, 4],
            "FAG_S2_CV_RATIO": [2, 4, 8, 16],
        }
    return {}


def get_validator(route: str, domains: dict[str, list[int]]):
    hardware = limits.AttentionLimits(
        max_cores=48,
        aic_num=24,
        UB_size=192 * 1024,
        L1_size=512 * 1024,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L2_size=192 * 1024**2,
        domains=domains,
        calc_type_size=4,
    )
    return valids.attention.create_validator(route, hardware)


def get_input_params(case: RouteCase) -> list[base.BaseParam]:
    batch, q_heads, kv_heads, s1, s2, d, dv = case.shape
    prefix = "FAG" if case.grad else "FA"
    values = {
        f"{prefix}_B": batch,
        f"{prefix}_N1": q_heads,
        f"{prefix}_N2": kv_heads,
        f"{prefix}_S1": s1,
        f"{prefix}_S2": s2,
        f"{prefix}_D": d,
        f"{prefix}_DV": dv,
        **{f"{prefix}_{name}": value for name, value in case.features.items()},
        "ATTENTION_GRAD": int(case.grad),
    }
    return [base.BaseParam(name=name, value=value, is_const=True) for name, value in values.items()]


def _case_env(case: RouteCase, input_params: list[base.BaseParam], key: int) -> dict[str, str]:
    env = dict(os.environ)
    env["ATTENTION_OPERATOR"] = case.operator
    env["ATTENTION_TILING_KEY"] = str(key)
    for param in input_params:
        env[param.name] = str(param.value)
    return env


def _run_with_timeout(command: list[str], env: dict[str, str], timeout: int) -> int:
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {timeout}s: {' '.join(command)}")
        os.killpg(process.pid, signal.SIGINT)
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124


def _run_with_timeout_capture(
    command: list[str], env: dict[str, str], timeout: int
) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    output = ""
    try:
        output, _ = process.communicate(timeout=timeout)
        rc = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        rc = 124
        print(f"TIMEOUT: {timeout}s: {' '.join(command)}")
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return rc, output


def _read_trace(path: Path) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    if not path.is_file():
        return records
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            continue
        try:
            records.append({
                "operator": fields[0],
                "priority": int(fields[1]),
                "status": int(fields[2]),
                "tiling_key": int(fields[3]),
            })
        except ValueError:
            continue
    return records


def _read_official_log_trace(operator: str, output: str) -> list[dict[str, int | str]]:
    """Recover route/key from the unmodified CANN debug messages.

    CANN logs the key inside TilingBaseClass::DumpTilingInfo immediately before
    TilingRegistryNew logs whether the template was accepted.  This fallback
    avoids rebuilding an already cached seed package solely for tracing.
    """

    records: list[dict[str, int | str]] = []
    latest_key: int | None = None
    key_pattern = re.compile(
        r"(?:tiling\s*key|tilingkey|tiling\s+is)\s*(?:is|:|=)?\s*(\d+)",
        re.IGNORECASE,
    )
    route_pattern = re.compile(
        r"(Do general op tiling success|Ignore general op tiling)\s+priority=(\d+)",
        re.IGNORECASE,
    )
    for line in output.splitlines():
        key_match = key_pattern.search(line)
        if key_match:
            latest_key = int(key_match.group(1))
        route_match = route_pattern.search(line)
        if not route_match:
            continue
        success = route_match.group(1).lower().startswith("do general")
        records.append({
            "operator": "FlashAttentionScoreGrad" if operator == FAG else "FlashAttentionScore",
            "priority": int(route_match.group(2)),
            "status": 0 if success else 1,
            "tiling_key": latest_key if success and latest_key is not None else 0,
        })
    return records


class CompileCache:
    def __init__(self) -> None:
        self.seen: set[tuple[str, int]] = set()
        self.failed: set[tuple[str, int]] = set()

    def ensure(self, operator: str, key: int) -> bool:
        identity = (operator, key)
        if identity in self.seen:
            return True
        if identity in self.failed:
            print(f"COMPILE PREVIOUSLY FAILED: operator={operator}, tiling_key={key}")
            return False
        print(f"COMPILE CHECK: operator={operator}, tiling_key={key}")
        env = dict(os.environ)
        env["ATTENTION_OPERATOR"] = operator
        env["ATTENTION_TILING_KEY"] = str(key)
        try:
            completed = subprocess.run(["bash", "./build_attention.sh"], env=env)
        except Exception as exc:
            print(f"COMPILE FAILED: operator={operator}, tiling_key={key}, error={exc!r}")
            self.failed.add(identity)
            return False
        if completed.returncode != 0:
            print(f"COMPILE FAILED: operator={operator}, tiling_key={key}, exit={completed.returncode}")
            self.failed.add(identity)
            return False
        self.seen.add(identity)
        return True


def discover_route(
    case: RouteCase,
    input_params: list[base.BaseParam],
    compile_cache: CompileCache,
    run_timeout: int,
    result_dir: Path,
) -> tuple[int | None, int | None, list[dict[str, int | str]]]:
    if not compile_cache.ensure(case.operator, case.seed_key):
        return None, None, []
    trace = result_dir / f"trace_discover_{case.name}.tsv"
    trace.unlink(missing_ok=True)
    env = _case_env(case, input_params, case.seed_key)
    env["ATTENTION_REFERENCE"] = "1"
    env["ATTENTION_DISCOVER_ONLY"] = "1"
    env["ATTENTION_ROUTE_TRACE"] = str(trace.resolve())
    # These are official CANN logging controls.  The source already logs both
    # the accepted template priority and its tiling key at debug level.
    env["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
    env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
    try:
        rc, output = _run_with_timeout_capture(
            ["bash", "./run_attention.sh", "-r", "npu"], env, run_timeout
        )
    except Exception as exc:
        print(f"DISCOVERY FAILED: route={case.name}, error={exc!r}")
        return None, None, _read_trace(trace)
    records = _read_trace(trace)
    if not records:
        records = _read_official_log_trace(case.operator, output)
    selected = [record for record in records if record["status"] == 0]
    if not selected:
        print(f"DISCOVERY FAILED: route={case.name}, exit={rc}, trace_records={len(records)}")
        return None, None, records
    terminal = selected[-1]
    if rc != 0:
        print(
            f"DISCOVERY FOUND UNBUILT KEY: route={case.name}, exit={rc}, "
            f"tiling_key={terminal['tiling_key']}"
        )
    return int(terminal["priority"]), int(terminal["tiling_key"]), records


def prepare_golden(
    case: RouteCase,
    input_params: list[base.BaseParam],
    domains: dict[str, list[int]],
    key: int,
    run_timeout: int,
    trace: Path,
) -> bool:
    env = _case_env(case, input_params, key)
    for name in domains:
        env.pop(name, None)
    env["ATTENTION_REFERENCE"] = "1"
    env.pop("ATTENTION_DISCOVER_ONLY", None)
    env["ATTENTION_ROUTE_TRACE"] = str(trace.resolve())
    output = Path("output/output.bin")
    golden = Path("output/golden.bin")
    output.unlink(missing_ok=True)
    golden.unlink(missing_ok=True)
    try:
        rc = _run_with_timeout(["bash", "./run_attention.sh", "-r", "npu"], env, run_timeout)
    except Exception as exc:
        print(f"GOLDEN FAILED: route={case.name}, tiling_key={key}, error={exc!r}")
        return False
    if rc != 0 or not output.is_file() or output.stat().st_size == 0:
        print(f"GOLDEN FAILED: route={case.name}, tiling_key={key}, exit={rc}")
        return False
    output.replace(golden)
    return True


class AttentionAuditAlgo(estimator_algs.AlgoProfileEst, bf.BruteForceAlgo):
    """Execute every candidate; the validator is a prediction, never a gate."""

    def __init__(self, *args, case: RouteCase, tiling_key: int, report_path: Path,
                 run_timeout: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.case = case
        self.tiling_key = tiling_key
        self.report_path = report_path
        self.run_timeout = run_timeout
        self.counts: Counter[str] = Counter()
        self.executed = 0

    def _is_right(
        self,
        relative_tol: float = 1e-5,
        absolute_tol: float = 1e-6,
        error_tol: float = 1e-4,
    ) -> bool:
        try:
            output = np.fromfile("output/output.bin", dtype=np.float32).reshape(-1)
            golden = np.fromfile("output/golden.bin", dtype=np.float32).reshape(-1)
        except Exception as error:
            print(f"IS_RIGHT: False, failed to read output: {error}")
            return False
        if output.size != golden.size or output.size == 0:
            print(f"IS_RIGHT: False, output size {output.size} != golden {golden.size}")
            return False
        finite = np.isfinite(output) & np.isfinite(golden)
        close = finite & np.isclose(output, golden, rtol=relative_tol, atol=absolute_tol)
        error_ratio = float((~close).sum()) / golden.size
        difference = np.abs(output[finite] - golden[finite]) if finite.any() else np.array([math.inf])
        max_absolute_error = float(difference.max(initial=0.0))
        denominator = np.maximum(np.abs(golden[finite]), absolute_tol) if finite.any() else np.array([1.0])
        max_relative_error = float((difference / denominator).max(initial=0.0))
        is_right = error_ratio <= error_tol
        print(
            f"IS_RIGHT: {is_right}, error ratio={error_ratio}, "
            f"max abs={max_absolute_error}, max rel={max_relative_error}"
        )
        return is_right

    def _get_time(self) -> float:
        try:
            profiles = list(Path(".").glob("OPPROF_*"))
            if not profiles:
                return float("inf")
            profile = max(profiles, key=lambda path: path.stat().st_mtime)
            table = pd.read_csv(profile / "OpBasicInfo.csv")
            durations = pd.to_numeric(table["Task Duration(us)"], errors="coerce").dropna()
            return float(durations.sum()) if not durations.empty else float("inf")
        except Exception:
            return float("inf")

    def _iter_combinations(self) -> Iterator[list[base.BaseParam]]:
        domains = self.validator.limits.domains
        names = list(domains)
        value_space = itertools.product(*(domains[name] for name in names)) if names else [()]
        for values in value_space:
            params = [base.BaseParam(name=p.name, value=p.value, is_const=True) for p in self.input_params]
            for name, value in zip(names, values):
                params.append(base.BaseParam(name=name, value=value, is_const=False, domain=domains[name]))
            yield self.validator.get_all_params(params)

    def _execute(self, params: list[base.BaseParam], step: int) -> tuple[int, float, bool, list[dict[str, int | str]]]:
        env = dict(os.environ)
        env["ATTENTION_OPERATOR"] = self.case.operator
        env["ATTENTION_TILING_KEY"] = str(self.tiling_key)
        trace = self.report_path.parent / f"trace_{self.case.name}_{step}.tsv"
        trace.unlink(missing_ok=True)
        env["ATTENTION_ROUTE_TRACE"] = str(trace.resolve())
        for param in params:
            env[param.name] = str(param.value)
        Path("output/output.bin").unlink(missing_ok=True)
        for profile in Path(".").glob("OPPROF_*"):
            shutil.rmtree(profile)
        rc = _run_with_timeout(["bash", self.runner, "-r", "npu"], env, self.run_timeout)
        duration = self._get_time() if rc == 0 else float("inf")
        correct = rc == 0 and math.isfinite(duration) and self._is_right()
        return rc, duration, correct, _read_trace(trace)

    def run(self, *args, **kwargs) -> base.BaseResult:
        best_duration = float("inf")
        best_params: list[base.BaseParam] = []
        total = 1
        for domain in self.validator.limits.domains.values():
            total *= len(domain)

        for step, params in enumerate(self._iter_combinations(), start=1):
            validator_error = ""
            try:
                predicted_valid = self.validator.is_valid(params)
            except Exception as exc:
                predicted_valid = False
                validator_error = f"validator: {exc!r}"
            try:
                rc, duration, runtime_pass, trace = self._execute(params, step)
                error = validator_error
            except Exception as exc:
                rc, duration, runtime_pass, trace = 1, float("inf"), False, []
                error = "; ".join(value for value in (validator_error, f"runtime: {exc!r}") if value)
            category = (
                f"validator_{'accept' if predicted_valid else 'reject'}_"
                f"runtime_{'pass' if runtime_pass else 'fail'}"
            )
            self.counts[category] += 1
            self.executed += 1
            if runtime_pass and duration < best_duration:
                best_duration = duration
                best_params = params
            selected = [record for record in trace if record["status"] == 0]
            actual_priority = selected[-1]["priority"] if selected else None
            actual_key = selected[-1]["tiling_key"] if selected else None
            record = {
                "route": self.case.name,
                "operator": self.case.operator,
                "expected_priority": self.case.expected_terminal_priority,
                "actual_priority": actual_priority,
                "expected_tiling_key": self.tiling_key,
                "actual_tiling_key": actual_key,
                "combination": step,
                "total_combinations": total,
                "validator_valid": predicted_valid,
                "runtime_exit_code": rc,
                "runtime_pass": runtime_pass,
                "duration_us": None if not math.isfinite(duration) else duration,
                "category": category,
                "error": error,
                "params": {param.name: param.value for param in params if not param.is_const},
            }
            with self.report_path.open("a") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"AUDIT route={self.case.name} comb={step}/{total} "
                f"validator={predicted_valid} runtime={runtime_pass} category={category} "
                f"duration_us={record['duration_us']}"
            )

        print(f"AUDIT COUNTS route={self.case.name}: {dict(self.counts)}")
        return base.BaseResult(best_duration, best_params)


def _select_npu() -> int:
    parser = argparse.ArgumentParser(description="Audit every FA/FAG validator route on one NPU")
    parser.add_argument("--id", required=True, type=int, help="physical NPU ID")
    args = parser.parse_args()
    if args.id < 0:
        parser.error("--id must be a non-negative integer")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.id)
    return args.id


def main() -> None:
    npu_id = _select_npu()
    run_timeout = int(os.environ.get("ATTENTION_RUN_TIMEOUT", "300"))
    result_dir = Path("results/attention_audit")
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / "validator_audit.jsonl"
    summary_path = result_dir / "summary.json"
    report_path.write_text("")
    compile_cache = CompileCache()
    summaries: list[dict[str, object]] = []

    print(f"NPU: physical {npu_id} (launcher logical 0)")
    print(f"ROUTES: {len(ROUTE_CASES)} (FA=6, FAG=10)")
    print("VALIDATOR MODE: label only; every combination executes")

    for case in ROUTE_CASES:
        domains = get_domains(case.name)
        input_params = get_input_params(case)
        route_summary: dict[str, object] = {
            "route": case.name,
            "official_class": ROUTE_CLASSES[case.name],
            "operator": case.operator,
            "expected_priority": case.priority,
            "expected_terminal_priority": case.expected_terminal_priority,
            "shape": case.shape,
            "domains": domains,
            "route_hit": False,
            "executed": 0,
            "counts": {},
        }
        print(
            f"\nROUTE START: {case.name} ({ROUTE_CLASSES[case.name]}), "
            f"shape={case.shape}, domains={domains}"
        )
        priority, key, trace = discover_route(case, input_params, compile_cache, run_timeout, result_dir)
        attempted_priorities = [int(record["priority"]) for record in trace]
        route_hit = case.priority in attempted_priorities and priority == case.expected_terminal_priority
        route_summary.update({
            "actual_terminal_priority": priority,
            "tiling_key": key,
            "attempted_priorities": attempted_priorities,
            "route_hit": route_hit,
        })
        if not route_hit or key is None:
            print(
                f"ROUTE MISMATCH: {case.name}, expected trace={case.priority}, "
                f"expected terminal={case.expected_terminal_priority}, actual terminal={priority}, "
                f"attempted={attempted_priorities}"
            )
            summaries.append(route_summary)
            continue
        print(f"ROUTE HIT: {case.name}, priority={priority}, tiling_key={key}")
        if not compile_cache.ensure(case.operator, key):
            summaries.append(route_summary)
            continue
        golden_trace = result_dir / f"trace_golden_{case.name}.tsv"
        golden_trace.unlink(missing_ok=True)
        if not prepare_golden(case, input_params, domains, key, run_timeout, golden_trace):
            summaries.append(route_summary)
            continue

        validator = get_validator(case.name, domains)
        algo = AttentionAuditAlgo(
            is_stop=lambda results: len(results) >= 1,
            validator=validator,
            input_params=input_params,
            runner="./run_attention.sh",
            cache_path=str(result_dir / "unused_runtime_cache.json"),
            verbose=True,
            case=case,
            tiling_key=key,
            report_path=report_path,
            run_timeout=run_timeout,
        )
        algo()
        route_summary["executed"] = algo.executed
        route_summary["counts"] = dict(algo.counts)
        summaries.append(route_summary)
        print(f"ROUTE END: {case.name}")

    complete = all(summary["route_hit"] and int(summary["executed"]) > 0 for summary in summaries)
    summary_document = {
        "complete": complete,
        "expected_routes": len(ROUTE_CASES),
        "hit_and_executed_routes": sum(
            bool(summary["route_hit"]) and int(summary["executed"]) > 0 for summary in summaries
        ),
        "routes": summaries,
    }
    summary_path.write_text(json.dumps(summary_document, indent=2, sort_keys=True) + "\n")
    print(f"\nAUDIT {'COMPLETE' if complete else 'INCOMPLETE'}")
    print(f"DETAILS: {report_path}")
    print(f"SUMMARY: {summary_path}")


if __name__ == "__main__":
    main()
