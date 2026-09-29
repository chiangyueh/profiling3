from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

from tiling.algs.base_algs.CycleEstimator import BaseAlgoNPUCycles
from tiling.algs.base_algs.MsprofEstimators import BaseAlgoMsprof, BaseAlgoProfile
from tiling.base import BaseParam


class _AttentionRunnerMixin:
    """Runner policy for FA/FAG.

    Correctness belongs to the operator runner because FA/FAG do not use the
    MatMul sample's ``output.bin``/``golden.bin`` files.  A runner may either
    emit ``DURATION_US=<number>`` or let the inherited estimator parse msprof.
    """

    def _is_right(self, *args, **kwargs) -> bool:
        return True

    def _run_attention_command(self, params: list[BaseParam], mode: str) -> float | None:
        env = dict(os.environ)
        for param in params:
            env[param.name] = str(param.value)
        for directory in Path(".").glob("OPPROF_*"):
            if directory.is_dir():
                shutil.rmtree(directory)

        completed = subprocess.run(
            ["bash", self.runner, "-r", mode],
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            if self.verbose:
                message = completed.stderr.strip() or completed.stdout.strip()
                print(f"ATTENTION RUNNER FAILED ({completed.returncode}): {message}")
            return float("inf")
        for line in reversed(completed.stdout.splitlines()):
            if line.strip().startswith("DURATION_US="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    return float("inf")
        return None


class AttentionAlgoProfile(_AttentionRunnerMixin, BaseAlgoProfile):
    def _run_estimator(self, params: list[BaseParam]) -> float:
        direct = self._run_attention_command(params, "npu")
        return self._get_time() if direct is None else direct


class AttentionAlgoMsprof(_AttentionRunnerMixin, BaseAlgoMsprof):
    def _run_estimator(self, params: list[BaseParam]) -> float:
        direct = self._run_attention_command(params, "sim")
        return self._get_time() if direct is None else direct


class AttentionAlgoNPUCycles(_AttentionRunnerMixin, BaseAlgoNPUCycles):
    pass
