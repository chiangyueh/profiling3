from __future__ import annotations

import __future__
import argparse
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import subprocess
import sys

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


# This route matches the fixed workload currently configured below. Route
# selection will move to the workload/autotiling registry when the exact
# DeepSeek/Pangu cases are added; it is not a user-facing launch argument.
KERNEL = "fag_generic"
TILING_KEY = 74804

# (B, N1, N2, S1, S2, D, DV). Replace this list with the collected
# DeepSeek/Pangu workload shapes before starting a search campaign.
SIZES = [
    (1, 8, 1, 128, 1536, 128, 128),
]

FEATURES = {
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
}


class PsoAlgo(estimator_algs.AlgoProfileEst, bf.BruteForceAlgo):
    def _is_right(
        self,
        relative_tol: float = 1e-5,
        absolute_tol: float = 1e-6,
        error_tol: float = 1e-4,
    ) -> bool:
        """Compare non-deterministic FAG output with the autotiling reference."""
        try:
            output = np.fromfile("output/output.bin", dtype=np.float32).reshape(-1)
            golden = np.fromfile("output/golden.bin", dtype=np.float32).reshape(-1)
        except Exception as error:
            if self.verbose:
                print(f"IS_RIGHT: False, failed to read output: {error}")
            return False
        if output.size != golden.size:
            if self.verbose:
                print(f"IS_RIGHT: False, output size {output.size} != golden {golden.size}")
            return False

        finite = np.isfinite(output) & np.isfinite(golden)
        close = finite & np.isclose(
            output,
            golden,
            rtol=relative_tol,
            atol=absolute_tol,
        )
        error_ratio = float((~close).sum()) / golden.size
        if finite.any():
            difference = np.abs(output[finite] - golden[finite])
            max_absolute_error = float(difference.max(initial=0.0))
            denominator = np.maximum(np.abs(golden[finite]), absolute_tol)
            max_relative_error = float((difference / denominator).max(initial=0.0))
        else:
            max_absolute_error = float("inf")
            max_relative_error = float("inf")
        is_right = error_ratio <= error_tol
        if self.verbose:
            print(
                f"IS_RIGHT: {is_right}, error ratio = {error_ratio}, "
                f"max abs error = {max_absolute_error}, max rel error = {max_relative_error}"
            )
        return is_right

    def _get_time(self) -> float:
        """FA/FAG may launch several tasks; score the complete operator."""
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


def get_domains(kernel: str) -> dict[str, list[int]]:
    if kernel in ("fa_general", "fa_varlen"):
        return {
            "FA_S1_BASE": [64, 128],
            "FA_S2_BASE": [64, 128],
            "FA_N_RATIO": [1, 2, 4, 5, 6, 8],
        }
    if kernel == "fag_mla":
        return {}
    if kernel == "fag_same_ab":
        return {
            "FAG_S1_CV_INNER": [256, 512, 1024],
            "FAG_S2_CV_INNER": [256, 512, 1024],
        }
    if kernel == "fag_generic":
        return {
            "FAG_S1_INNER": list(range(16, 129, 16)),
            "FAG_S2_INNER": list(range(16, 65, 16)),
            "FAG_S1_CV_RATIO": [1, 4],
            "FAG_S2_CV_RATIO": [2, 4, 8, 16],
        }
    raise ValueError(f"unsupported attention route: {kernel}")


def get_validator(kernel: str, domains: dict[str, list[int]]):
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
    return valids.attention.create_validator(kernel, hardware)


def get_input_params(kernel: str, shape: tuple[int, ...]) -> list[base.BaseParam]:
    batch, q_heads, kv_heads, s1, s2, d, dv = shape
    prefix = "FA" if kernel.startswith("fa_") else "FAG"
    values = {
        f"{prefix}_B": batch,
        f"{prefix}_N1": q_heads,
        f"{prefix}_N2": kv_heads,
        f"{prefix}_S1": s1,
        f"{prefix}_S2": s2,
        f"{prefix}_D": d,
        f"{prefix}_DV": dv,
        **{f"{prefix}_{name}": value for name, value in FEATURES.items()},
        "ATTENTION_GRAD": int(prefix == "FAG"),
    }
    return [base.BaseParam(name=name, value=value, is_const=True) for name, value in values.items()]


def prepare_golden(input_params: list[base.BaseParam], domains: dict[str, list[int]]) -> None:
    """Run official autotiling once and use its output as the shape reference."""
    env = dict(os.environ)
    for name in domains:
        env.pop(name, None)
    for param in input_params:
        env[param.name] = str(param.value)
    env["ATTENTION_REFERENCE"] = "1"

    output = Path("output/output.bin")
    golden = Path("output/golden.bin")
    output.unlink(missing_ok=True)
    completed = subprocess.run(
        ["bash", "./run_attention.sh", "-r", "npu"],
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to generate the official-autotiling reference "
            f"(exit code {completed.returncode})"
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("attention reference run produced no output/output.bin")
    output.replace(golden)


def _select_npu() -> int:
    parser = argparse.ArgumentParser(description="Run FA/FAG tiling search on one NPU")
    parser.add_argument("--id", required=True, type=int, help="physical NPU ID")
    args = parser.parse_args()
    if args.id < 0:
        parser.error("--id must be a non-negative integer")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.id)
    return args.id


def main() -> None:
    npu_id = _select_npu()
    os.environ["ATTENTION_TILING_KEY"] = str(TILING_KEY)

    domains = get_domains(KERNEL)
    validator = get_validator(KERNEL, domains)
    print(f"NPU: physical {npu_id} (launcher logical 0)")
    print(f"KERNEL: {KERNEL}")
    print(f"TILING_KEY: {TILING_KEY}")
    print(f"DOMAINS: {domains}")

    for shape in SIZES:
        input_params = get_input_params(KERNEL, shape)
        if not validator.get_combinations(1, input_params):
            raise ValueError(f"shape/features do not reach a legal {KERNEL} configuration: {shape}")
        prepare_golden(input_params, domains)

        shape_key = "_".join(str(value) for value in shape)
        algo = PsoAlgo(
            is_stop=lambda results: len(results) >= 1,
            validator=validator,
            input_params=input_params,
            runner="./run_attention.sh",
            cache_path=f"cache_{KERNEL}_{shape_key}_real_v2.json",
            verbose=True,
        )
        print(f"START: {shape}")
        algo()
        print(f"END: {shape}")
