from __future__ import annotations

from tiling.base.Base import BaseAlgo, BaseParam
from pathlib import Path
import os
import shutil
import subprocess
import numpy as np

class BaseAlgoNPUCycles(BaseAlgo):
    def _run_estimator(self, params: list[BaseParam]) -> float:
        env = dict(os.environ)
        for param in params:
            env[param.name] = str(param.value)
        for d in Path(".").glob("OPPROF_*"):
            shutil.rmtree(d)
 
        cycles_runs: list[int] = []
        for _ in range(5):
            r = subprocess.run(
                ["bash", self.runner, "-r", "npu", "--cycles-only"],
                env=env, capture_output=True, text=True,
            )
            c = self._parse_cycles(r.stdout)
            if c is not None:
                cycles_runs.append(c)
 
        if not cycles_runs:
            return float("inf")
        return float(np.median(cycles_runs))
 
    def _parse_cycles(self, stdout: str) -> float:
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("CYCLES="):
                try:
                    return float(line.split("=", 1)[1])
                except ValueError:
                    return float('inf')
        return float('inf')
