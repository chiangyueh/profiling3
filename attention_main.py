from __future__ import annotations

import __future__
import argparse
from collections import Counter, deque
from dataclasses import dataclass
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading

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

from tiling import base, estimator_algs, limits, pso, valids


FA = "flash_attention_score"
FAG = "flash_attention_score_grad"
BOOTSTRAP_KEYS = {
    (FA, 4): 1144284208,
    (FA, 2): 1144808752,
    (FAG, 4): 74804,
    # Official FlashAttentionScoreGraTilingMla::GetTilingKey() for FP16:
    # GET_TPL_TILING_KEY(9, 9, 9, 0, 3, 0, ...).
    (FAG, 2): 26214,
}

DEFAULT_FEATURES = {
    "DTYPE_BYTES": 4,
    # 0 keeps the legacy byte-based launcher choice; 2 selects BF16 while
    # retaining DTYPE_BYTES=2 for Host-route and validator calculations.
    "DTYPE_KIND": 0,
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
    tiling_key: int | None = None
    terminal_priority: int | None = None
    route: str | None = None

    @property
    def grad(self) -> bool:
        return self.operator == FAG

    @property
    def expected_terminal_priority(self) -> int:
        return self.terminal_priority if self.terminal_priority is not None else self.priority

    @property
    def route_name(self) -> str:
        return self.route if self.route is not None else self.name


def _features(**updates: int) -> dict[str, int]:
    values = dict(DEFAULT_FEATURES)
    values.update(updates)
    return values


ROUTE_COVERAGE_CASES = (
    RouteCase(
        "fa_s1s2",
        FA,
        96,
        (1, 8, 1, 128, 1536, 128, 128),
        _features(),
        tiling_key=1144284208,
    ),
    RouteCase(
        "fa_s1",
        FA,
        97,
        (1, 8, 1, 128, 512, 128, 128),
        _features(DTYPE_BYTES=2),
        tiling_key=1144808752,
    ),
    RouteCase(
        "fa_varlen",
        FA,
        94,
        (2, 1, 1, 128, 128, 64, 64),
        _features(LAYOUT=valids.attention.LAYOUT_TND, HAS_ACTUAL_SEQ=1),
        tiling_key=1145332784,
    ),
    RouteCase(
        "fa_drop_adapter",
        FA,
        90,
        (1, 1, 1, 31, 65, 64, 64),
        _features(DTYPE_BYTES=2, HAS_DROP=1),
        tiling_key=3258713696,
        terminal_priority=98,
    ),
    RouteCase(
        "fa_same_ab",
        FA,
        95,
        (1, 8, 1, 128, 512, 96, 96),
        _features(DTYPE_BYTES=2),
        tiling_key=1144809008,
    ),
    RouteCase(
        "fa_b",
        FA,
        98,
        (1, 1, 1, 16, 16, 64, 64),
        _features(DTYPE_BYTES=2),
        tiling_key=1111230048,
    ),
)

# Fully specified per-rank DeepSeek workloads from the supplied training
# configurations.  BF16 is used for the BF16/FP8 configurations because the
# aclnnFlashAttentionScoreV2 API used by this evaluator accepts FP32/FP16/BF16,
# not the operator's separate FP8 interface.
MODEL_FA_CASES = (
    RouteCase(
        "deepseek_v3_generate_s4096_bnsd_tp1",
        FA,
        95,
        (1, 128, 128, 4096, 4096, 192, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_BNSD),
        route="fa_same_ab",
    ),
    RouteCase(
        "deepseek_v3_pipeline_s4096_sbh_tp1",
        FA,
        95,
        (1, 128, 128, 4096, 4096, 192, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_SBH),
        route="fa_same_ab",
    ),
    RouteCase(
        "deepseek_v3_pretrain_s4096_sbh_tp2",
        FA,
        95,
        (1, 64, 64, 4096, 4096, 192, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_SBH),
        route="fa_same_ab",
    ),
    RouteCase(
        "deepseek_v3_tune_s4096_bnsd_tp2",
        FA,
        95,
        (1, 64, 64, 4096, 4096, 192, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_BNSD),
        route="fa_same_ab",
    ),
    RouteCase(
        "deepseek_v3_pretrain_s4096_sbh_tp4",
        FA,
        95,
        (1, 32, 32, 4096, 4096, 192, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_SBH),
        route="fa_same_ab",
    ),
    RouteCase(
        "deepseek_v3_surrogate_s4096_bnsd_tp8",
        FA,
        95,
        (1, 16, 16, 4096, 4096, 192, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_BNSD),
        route="fa_same_ab",
    ),
    RouteCase(
        "deepseek_v3_bf16_s8192_sbh_tp1",
        FA,
        95,
        (1, 128, 128, 8192, 8192, 192, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_SBH),
        route="fa_same_ab",
    ),
    # Pangu's public model-level shapes do not specify the actual training
    # batch/TP/CP mapping.  B=1 and unsplit Q=96/KV=8 are therefore labelled
    # explicitly as proxies rather than claimed as per-rank training shapes.
    RouteCase(
        "pangu_ultra_dense_s4096_b1_model_proxy",
        FA,
        96,
        (1, 96, 8, 4096, 4096, 128, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_BNSD),
        route="fa_s1s2",
    ),
    RouteCase(
        "pangu_ultra_dense_s8192_b1_model_proxy",
        FA,
        96,
        (1, 96, 8, 8192, 8192, 128, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_BNSD),
        route="fa_s1s2",
    ),
)

# These public Pangu model-level proxies are selectable individually through
# ATTENTION_ROUTE, but are not in the default batch because their output dumps
# alone are about 1.6 GiB and 6.4 GiB respectively.
LARGE_MODEL_FA_CASES = (
    RouteCase(
        "pangu_ultra_dense_s32768_b1_model_proxy",
        FA,
        96,
        (1, 96, 8, 32768, 32768, 128, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_BNSD),
        route="fa_s1s2",
    ),
    RouteCase(
        "pangu_ultra_dense_s131072_b1_model_proxy",
        FA,
        96,
        (1, 96, 8, 131072, 131072, 128, 128),
        _features(DTYPE_BYTES=2, DTYPE_KIND=2, LAYOUT=valids.attention.LAYOUT_BNSD),
        route="fa_s1s2",
    ),
)

ALL_FA_ROUTE_CASES = ROUTE_COVERAGE_CASES + MODEL_FA_CASES + LARGE_MODEL_FA_CASES

ACTIVE_ROUTE = os.environ.get("ATTENTION_ROUTE", "")
_FA_ROUTE_BY_NAME = {case.name: case for case in ALL_FA_ROUTE_CASES}
# Keep the default command on one verified workload.  Additional model cases
# remain available internally for later batches after this production
# validator/repair flow has been verified on the NPU.
DEFAULT_FA_ROUTE_CASES = (
    _FA_ROUTE_BY_NAME["deepseek_v3_pretrain_s4096_sbh_tp2"],
)
if ACTIVE_ROUTE and ACTIVE_ROUTE not in _FA_ROUTE_BY_NAME:
    raise ValueError(f"unknown ATTENTION_ROUTE: {ACTIVE_ROUTE}")
ROUTE_CASES = (_FA_ROUTE_BY_NAME[ACTIVE_ROUTE],) if ACTIVE_ROUTE else ()

# Match the colleague's search_det.py PSO search size.
SWARM_SIZE = 16
SEARCH_ITERATIONS = 30

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
    if route == "fa_drop_adapter":
        return {
            # Search the complete 16-element-aligned B-template tile domain. The
            # official implementation caps S1 at 256 and budgets 8192
            # S1-by-S2 elements; extending S2 through 512 deliberately samples
            # both sides of that hardware boundary so validator mistakes are
            # observable on the NPU instead of being filtered in advance.
            "FA_S1_BASE": list(range(16, 257, 16)),
            "FA_S2_BASE": list(range(16, 513, 16)),
            # FlashAttentionScoreTilingB does not consume nRatio.  Keep the
            # field fixed only because the shared FA validator expects it.
            "FA_N_RATIO": [1],
        }
    if route == "fa_b":
        return {
            "FA_S1_BASE": [16, 32],
            "FA_S2_BASE": [16, 48, 64],
            "FA_N_RATIO": [1, 2],
        }
    if route == "fa_s1":
        return {
            "FA_S1_BASE": [64, 128, 256],
            "FA_S2_BASE": [64, 128, 1024],
            "FA_N_RATIO": [1, 2, 8],
        }
    if route == "fa_same_ab":
        return {
            "FA_S1_BASE": [64, 128, 192, 256],
            "FA_S2_BASE": [16, 32, 64, 96, 128, 160, 192, 256],
            "FA_N_RATIO": [1, 2, 4, 5, 6, 8, 12, 16],
        }
    if route == "fa_s1s2":
        return {
            # A 512-point domain around the official 128x128 / ratio<=8
            # envelope. Values on both sides of every limit let the NPU audit
            # measure false accepts and false rejects instead of pre-filtering
            # the validator's boundary.
            "FA_S1_BASE": [16, 32, 64, 96, 128, 160, 192, 256],
            "FA_S2_BASE": [16, 32, 64, 96, 128, 160, 192, 256],
            "FA_N_RATIO": [1, 2, 4, 5, 6, 8, 12, 16],
        }
    if route == "fa_varlen":
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
        print(f"TIMEOUT: {timeout}s: {' '.join(command)}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return 124
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[WARNING] timed-out process did not exit after SIGKILL", flush=True)
        return 124


_DISCOVERY_LOG_LINE = re.compile(
    r"Start to dump tiling info|Do general op tiling success|Ignore general op tiling|"
    r"tiling\s*key|tilingkey|tiling\s+is|BinaryGetFunctionByEntry|funcEntry=|"
    r"Cannot find binary|\[ERROR\]",
    re.IGNORECASE,
)


def _run_with_timeout_capture(
    command: list[str], env: dict[str, str], timeout: int
) -> tuple[int, str]:
    """Drain verbose CANN output while retaining only route/key diagnostics."""

    process = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    selected_lines: deque[str] = deque(maxlen=512)

    def drain_output() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if _DISCOVERY_LOG_LINE.search(line):
                    selected_lines.append(line)
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=drain_output, daemon=True)
    reader.start()
    try:
        rc = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {timeout}s: {' '.join(command)}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        rc = 124
    reader.join(timeout=2)
    output = "".join(selected_lines)
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
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


def _read_effective_tiling(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text().splitlines()):
        fields = line.split("\t")
        try:
            if len(fields) == 3 and fields[0] == "FlashAttentionScoreTilingB":
                return {
                    "FA_S1_BASE": int(fields[1]),
                    "FA_S2_BASE": int(fields[2]),
                    "FA_N_RATIO": 1,
                }
            if len(fields) == 4 and fields[0] == "FlashAttentionScoreTilingS1s2Bn2gs1":
                return {
                    "FA_S1_BASE": int(fields[1]),
                    "FA_S2_BASE": int(fields[2]),
                    "FA_N_RATIO": int(fields[3]),
                }
            if len(fields) == 4 and fields[0] == "FlashAttentionScoreTilingS1s2Bn2gs1SameAB":
                return {
                    "FA_S1_BASE": int(fields[1]),
                    "FA_S2_BASE": int(fields[2]),
                    "FA_N_RATIO": int(fields[3]),
                }
        except (IndexError, ValueError):
            continue
    return None


def _read_official_log_trace(operator: str, output: str) -> list[dict[str, int | str]]:
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
    if latest_key is None:
        entry_matches = re.findall(r"(?:funcEntry|tiling\s+key)\s*[=:]\s*(\d+)", output, re.IGNORECASE)
        if entry_matches:
            latest_key = int(entry_matches[-1])
    if latest_key is not None:
        for record in records:
            if record["status"] == 0 and record["tiling_key"] == 0:
                record["tiling_key"] = latest_key
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
) -> tuple[int | None, int | None]:
    """Ask the official Host tiling code for this shape's route and key."""

    seed_key = BOOTSTRAP_KEYS[(case.operator, case.features["DTYPE_BYTES"])]
    if not compile_cache.ensure(case.operator, seed_key):
        return None, None

    trace = result_dir / f"trace_discover_{case.name}.tsv"
    trace.unlink(missing_ok=True)
    env = _case_env(case, input_params, seed_key)
    for name in get_domains(case.route_name):
        env.pop(name, None)
    env["ATTENTION_REFERENCE"] = "1"
    env["ATTENTION_ROUTE_TRACE"] = str(trace.resolve())
    env["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
    env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"

    # Tiling is selected only when the operator is dispatched, not during
    # GetWorkspaceSize.  A bootstrap binary may therefore report an unbuilt
    # kernel after writing the trace; that nonzero exit is expected here.
    timeout = min(run_timeout, 60)
    print(
        f"DISCOVERY START: route={case.name}, bootstrap_key={seed_key}, timeout={timeout}s",
        flush=True,
    )
    try:
        rc, output = _run_with_timeout_capture(
            ["bash", "./run_attention.sh", "-r", "npu"], env, timeout
        )
    except Exception as exc:
        print(f"DISCOVERY FAILED: route={case.name}, error={exc!r}")
        return None, None

    records = _read_trace(trace)
    if not records:
        records = _read_official_log_trace(case.operator, output)
    selected = [
        record for record in records
        if record["status"] == 0 and int(record["tiling_key"]) > 0
    ]
    if not selected:
        print(f"DISCOVERY FAILED: route={case.name}, exit={rc}, no Host priority/key record")
        return None, None
    terminal = selected[-1]
    priority = int(terminal["priority"])
    key = int(terminal["tiling_key"])
    print(
        f"DISCOVERY RESULT: route={case.name}, priority={priority}, "
        f"tiling_key={key}, probe_exit={rc}"
    )
    return priority, key


def prepare_golden(
    case: RouteCase,
    input_params: list[base.BaseParam],
    domains: dict[str, list[int]],
    key: int,
    run_timeout: int,
    trace: Path,
) -> tuple[bool, float]:
    env = _case_env(case, input_params, key)
    for name in domains:
        env.pop(name, None)
    env.pop("ATTENTION_REFERENCE", None)
    env.pop("ATTENTION_DISCOVER_ONLY", None)
    env["ATTENTION_ROUTE_TRACE"] = str(trace.resolve())
    output = Path("output/output.bin")
    golden = Path("output/golden.bin")
    output.unlink(missing_ok=True)
    golden.unlink(missing_ok=True)
    for profile in Path(".").glob("OPPROF_*"):
        shutil.rmtree(profile)
    try:
        rc = _run_with_timeout(["bash", "./run_attention.sh", "-r", "npu"], env, run_timeout)
    except Exception as exc:
        print(f"GOLDEN FAILED: route={case.name}, tiling_key={key}, error={exc!r}")
        return False, float("inf")
    if rc != 0 or not output.is_file() or output.stat().st_size == 0:
        print(f"GOLDEN FAILED: route={case.name}, tiling_key={key}, exit={rc}")
        return False, float("inf")
    duration = _latest_profile_duration()
    if not math.isfinite(duration):
        print(f"GOLDEN FAILED: route={case.name}, official latency is unavailable")
        return False, float("inf")
    output.replace(golden)
    print(f"OFFICIAL BASELINE: route={case.name} duration_us={duration}")
    return True, duration


def _latest_profile_duration() -> float:
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


class AttentionAuditAlgo(estimator_algs.AlgoProfileEst, pso.PsoAlgo):
    """Run PSO with validator repair before evaluating a candidate on NPU."""

    def __init__(self, *args, case: RouteCase, tiling_key: int, report_path: Path,
                 run_timeout: int, **kwargs) -> None:
        self.case = case
        self.tiling_key = tiling_key
        self.report_path = report_path
        self.run_timeout = run_timeout
        self.counts: Counter[str] = Counter()
        self.executed = 0
        self.proposed = 0
        self.audit_validator = kwargs["validator"]
        super().__init__(*args, **kwargs)

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
        return _latest_profile_duration()

    def _execute(
        self,
        params: list[base.BaseParam],
        step: int,
    ) -> tuple[
        int,
        float,
        bool,
        list[dict[str, int | str]],
        dict[str, int] | None,
        str,
    ]:
        env = dict(os.environ)
        env["ATTENTION_OPERATOR"] = self.case.operator
        env["ATTENTION_TILING_KEY"] = str(self.tiling_key)
        trace = self.report_path.parent / f"trace_{self.case.name}_{step}.tsv"
        trace.unlink(missing_ok=True)
        env["ATTENTION_ROUTE_TRACE"] = str(trace.resolve())
        effective_trace = self.report_path.parent / f"effective_{self.case.name}_{step}.tsv"
        effective_trace.unlink(missing_ok=True)
        env["ATTENTION_EFFECTIVE_TILING_TRACE"] = str(effective_trace.resolve())
        for param in params:
            env[param.name] = str(param.value)
        Path("output/output.bin").unlink(missing_ok=True)
        for profile in Path(".").glob("OPPROF_*"):
            shutil.rmtree(profile)
        rc = _run_with_timeout(["bash", self.runner, "-r", "npu"], env, self.run_timeout)
        duration = self._get_time() if rc == 0 else float("inf")
        route_trace = _read_trace(trace)
        selected = [record for record in route_trace if record["status"] == 0]
        terminal = selected[-1] if selected else None
        effective = _read_effective_tiling(effective_trace)
        requested = {
            param.name: param.value
            for param in params
            if param.name in ("FA_S1_BASE", "FA_S2_BASE", "FA_N_RATIO")
        }
        failures: list[str] = []
        if terminal is None:
            failures.append("successful Host tiling route trace is missing")
        else:
            if int(terminal["priority"]) != self.case.expected_terminal_priority:
                failures.append(
                    f"priority {terminal['priority']} != {self.case.expected_terminal_priority}"
                )
            if int(terminal["tiling_key"]) != self.tiling_key:
                failures.append(f"tiling key {terminal['tiling_key']} != {self.tiling_key}")
        if effective is None:
            failures.append("effective FA tiling trace is missing")
        else:
            for name, value in requested.items():
                if effective.get(name) != value:
                    failures.append(f"effective {name}={effective.get(name)} != requested {value}")
        injection_matches = not failures
        print(
            f"EFFECTIVE TILING route={self.case.name} requested={requested} "
            f"effective={effective} match={injection_matches}"
        )
        numerically_correct = rc == 0 and math.isfinite(duration) and self._is_right()
        runtime_pass = numerically_correct and injection_matches
        return rc, duration, runtime_pass, route_trace, effective, "; ".join(failures)

    def _duration(self, params: list[base.BaseParam]) -> base.BaseResult:
        params = self.audit_validator.get_all_params(params)
        self.proposed += 1

        requested = {
            param.name: param.value
            for param in params
            if param.name in self.audit_validator.limits.domains
        }
        validator_error = ""
        try:
            candidate_valid = self.audit_validator.is_valid(params)
        except Exception as exc:
            candidate_valid = False
            validator_error = f"validator: {exc!r}"

        repaired = False
        if not candidate_valid and not validator_error:
            try:
                repaired_params = self.audit_validator.repair(params)
                if self.audit_validator.is_valid(repaired_params):
                    params = repaired_params
                    repaired = True
                    repaired_values = {
                        param.name: param.value
                        for param in params
                        if param.name in self.audit_validator.limits.domains
                    }
                    print(
                        f"VALIDATOR REPAIR route={self.case.name} "
                        f"requested={requested} repaired={repaired_values}"
                    )
                else:
                    validator_error = "validator repair did not produce a valid configuration"
            except Exception as exc:
                validator_error = f"validator repair: {exc!r}"

        # Match the colleague framework: an invalid configuration is repaired
        # before evaluation, and an unrepairable one never reaches the NPU.
        if validator_error:
            print(
                f"VALIDATOR REJECT route={self.case.name} requested={requested} "
                f"reason={validator_error}"
            )
            return base.BaseResult(float("inf"), params)

        cache_key = self._key(params)
        if cache_key in self._cache:
            duration = self._cache[cache_key]
            print(
                f"PSO CACHE HIT route={self.case.name} candidate={self.proposed} "
                f"duration_us={None if not math.isfinite(duration) else duration}"
            )
            return base.BaseResult(duration, params)

        self.executed += 1
        try:
            rc, duration, runtime_pass, trace, effective, execution_error = self._execute(
                params, self.executed
            )
            error = execution_error
        except Exception as exc:
            rc, duration, runtime_pass, trace = 1, float("inf"), False, []
            effective = None
            error = f"runtime: {exc!r}"

        category = (
            f"validator_{'repair' if repaired else 'accept'}_"
            f"runtime_{'pass' if runtime_pass else 'fail'}"
        )
        self.counts[category] += 1
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
            "candidate": self.proposed,
            "execution": self.executed,
            "algorithm": "pso",
            "validator_valid": candidate_valid,
            "validator_repaired": repaired,
            "runtime_exit_code": rc,
            "runtime_pass": runtime_pass,
            "duration_us": None if not math.isfinite(duration) else duration,
            "category": category,
            "error": error,
            "effective_tiling": effective,
            "params": {param.name: param.value for param in params if not param.is_const},
        }
        with self.report_path.open("a") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"AUDIT route={self.case.name} candidate={self.proposed} execution={self.executed} algorithm=pso "
            f"validator={'repair' if repaired else 'accept'} runtime={runtime_pass} category={category} "
            f"duration_us={record['duration_us']}"
        )

        result_duration = duration if runtime_pass else float("inf")
        self._cache[cache_key] = result_duration
        self._save_cache()
        return base.BaseResult(result_duration, params)


def _select_npu() -> int:
    parser = argparse.ArgumentParser(description="Audit the configured FA/FAG routes on one NPU")
    parser.add_argument("--id", required=True, type=int, help="physical NPU ID")
    args = parser.parse_args()
    if args.id < 0:
        parser.error("--id must be a non-negative integer")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.id)
    return args.id


def _run_all_fa_isolated() -> None:
    """Run each verified FA route through the proven single-route process."""

    npu_id = _select_npu()
    script_dir = Path(__file__).resolve().parent
    failures: list[str] = []
    print(f"FA TEST ROUTES: {len(DEFAULT_FA_ROUTE_CASES)}, physical NPU: {npu_id}", flush=True)

    for index, case in enumerate(DEFAULT_FA_ROUTE_CASES, start=1):
        result_dir = script_dir / "results" / "attention_audit" / case.name
        summary_path = result_dir / "summary.json"
        summary_path.unlink(missing_ok=True)
        env = dict(os.environ)
        env["ATTENTION_ROUTE"] = case.name
        print(
            f"\nFA TEST {index}/{len(DEFAULT_FA_ROUTE_CASES)} START: {case.name}",
            flush=True,
        )
        completed = subprocess.run(
            [sys.executable, str(script_dir / "main.py"), f"--id={npu_id}"],
            cwd=script_dir,
            env=env,
        )

        complete = False
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
                shape = summary["shapes"][0]
                executed = int(shape["executed"])
                proposed = int(shape["proposed"])
                classified = sum(int(count) for count in shape["counts"].values())
                complete = (
                    bool(summary.get("complete"))
                    and proposed > 0
                    and executed > 0
                    and classified == executed
                )
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                complete = False
        if completed.returncode == 0 and complete:
            print(f"FA TEST {index}/{len(DEFAULT_FA_ROUTE_CASES)} PASS: {case.name}", flush=True)
        else:
            failures.append(case.name)
            print(
                f"FA TEST {index}/{len(DEFAULT_FA_ROUTE_CASES)} FAIL: {case.name} "
                f"(exit={completed.returncode})",
                flush=True,
            )

    if failures:
        print(f"\nFA TEST INCOMPLETE: {', '.join(failures)}", flush=True)
        raise SystemExit(1)
    print("\nFA TEST COMPLETE", flush=True)


def main() -> None:
    if not ACTIVE_ROUTE:
        _run_all_fa_isolated()
        return

    npu_id = _select_npu()
    run_timeout = int(os.environ.get("ATTENTION_RUN_TIMEOUT", "300"))
    result_dir = Path("results/attention_audit") / ACTIVE_ROUTE
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / "validator_audit.jsonl"
    summary_path = result_dir / "summary.json"
    report_path.write_text("")
    compile_cache = CompileCache()
    summaries: list[dict[str, object]] = []

    print(f"NPU: physical {npu_id} (launcher logical 0)")
    fa_count = sum(case.operator == FA for case in ROUTE_CASES)
    fag_count = sum(case.operator == FAG for case in ROUTE_CASES)
    print(f"ROUTES: {len(ROUTE_CASES)} (FA={fa_count}, FAG={fag_count})")
    print(
        f"SEARCH: colleague PSO, swarm={SWARM_SIZE}, iterations={SEARCH_ITERATIONS}; "
        f"{SWARM_SIZE * (SEARCH_ITERATIONS + 1)} proposals including swarm initialization; "
        "validator repairs rejected candidates before NPU; unique repaired candidates execute "
        "and duplicates reuse cache"
    )

    for case in ROUTE_CASES:
        domains = get_domains(case.route_name)
        input_params = get_input_params(case)
        route_summary: dict[str, object] = {
            "route": case.name,
            "route_family": case.route_name,
            "official_class": ROUTE_CLASSES[case.route_name],
            "operator": case.operator,
            "expected_priority": case.priority,
            "expected_terminal_priority": case.expected_terminal_priority,
            "shape": case.shape,
            "domains": domains,
            "tiling_key": None,
            "baseline_pass": False,
            "official_duration_us": None,
            "executed": 0,
            "proposed": 0,
            "counts": {},
        }
        print(
            f"\nROUTE START: {case.name} ({ROUTE_CLASSES[case.route_name]}), "
            f"shape={case.shape}, domains={domains}"
        )
        key = case.tiling_key
        if key is None:
            actual_priority, key = discover_route(
                case, input_params, compile_cache, run_timeout, result_dir
            )
            if key is None or actual_priority != case.expected_terminal_priority:
                if key is not None:
                    print(
                        f"ROUTE MISMATCH: {case.name}, expected_priority="
                        f"{case.expected_terminal_priority}, actual_priority={actual_priority}"
                    )
                summaries.append(route_summary)
                continue
        else:
            print(
                f"ROUTE CONFIGURED: {case.name}, expected_priority={case.expected_terminal_priority}, "
                f"verified_tiling_key={key}; discovery skipped"
            )
        route_summary["tiling_key"] = key
        if not compile_cache.ensure(case.operator, key):
            summaries.append(route_summary)
            continue
        golden_trace = result_dir / f"trace_golden_{case.name}.tsv"
        golden_trace.unlink(missing_ok=True)
        baseline_pass, official_duration = prepare_golden(
            case, input_params, domains, key, run_timeout, golden_trace
        )
        if not baseline_pass:
            summaries.append(route_summary)
            continue
        route_summary["baseline_pass"] = True
        route_summary["official_duration_us"] = official_duration

        validator = get_validator(case.route_name, domains)
        algo = AttentionAuditAlgo(
            is_stop=lambda results: len(results) >= SEARCH_ITERATIONS,
            validator=validator,
            input_params=input_params,
            swarm_size=SWARM_SIZE,
            runner="./run_attention.sh",
            cache_path=str(
                result_dir
                / f"search_cache_explicit_host_full_v3_{case.name}_{'_'.join(map(str, case.shape))}.json"
            ),
            verbose=True,
            case=case,
            tiling_key=key,
            report_path=report_path,
            run_timeout=run_timeout,
        )
        search_results = algo()
        best = min(search_results, key=lambda result: result.duration)
        improvement = official_duration - best.duration
        improvement_percent = improvement / official_duration * 100.0
        print(
            f"SEARCH VS OFFICIAL: route={case.name} official_us={official_duration} "
            f"best_us={best.duration} improvement_us={improvement} "
            f"improvement_percent={improvement_percent}"
        )
        print(f"AUDIT COUNTS route={case.name}: {dict(algo.counts)}")
        route_summary["executed"] = algo.executed
        route_summary["proposed"] = algo.proposed
        route_summary["counts"] = dict(algo.counts)
        route_summary["best_duration_us"] = best.duration
        route_summary["best_params"] = {
            param.name: param.value for param in best.params if not param.is_const
        }
        route_summary["better_than_official"] = best.duration < official_duration
        route_summary["improvement_percent"] = improvement_percent
        summaries.append(route_summary)
        print(f"ROUTE END: {case.name}")

    complete = all(summary["baseline_pass"] and int(summary["executed"]) > 0 for summary in summaries)
    summary_document = {
        "complete": complete,
        "expected_shapes": len(ROUTE_CASES),
        "baseline_pass_and_executed_shapes": sum(
            bool(summary["baseline_pass"]) and int(summary["executed"]) > 0 for summary in summaries
        ),
        "shapes": summaries,
    }
    summary_path.write_text(json.dumps(summary_document, indent=2, sort_keys=True) + "\n")
    print(f"\nAUDIT {'COMPLETE' if complete else 'INCOMPLETE'}")
    print(f"DETAILS: {report_path}")
    print(f"SUMMARY: {summary_path}")


if __name__ == "__main__":
    main()
