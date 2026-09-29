from tiling.base.Base import BaseAlgo, BaseParam
from pathlib import Path
import os
import re
import subprocess


CYCLES_RE = re.compile(r"CYCLES_US:\s*([0-9]+\.?[0-9]*)")


def _drop_stale_output() -> None:
    output = Path("output/output.bin")
    if output.exists():
        output.unlink()


class AlgoNpuCyclesEst(BaseAlgo):
    warmup = 3
    repeat = 3
    def _run_estimator(self, params: list[BaseParam]) -> float:
        env = dict(os.environ)
        for param in params:
            env[param.name] = str(param.value)
        env["MM_CYCLES_WARMUP"] = str(self.warmup)
        env["MM_CYCLES_REPEAT"] = str(self.repeat)

        cceprint = Path("cceprint")
        if cceprint.exists():
            for f in cceprint.glob("*.cce"):
                f.unlink()
        _drop_stale_output()
        process = subprocess.run(["bash", self.runner, "-r", "npu", "--cycles-only"],
                                 env=env, capture_output=True, text=True)
        return self._get_time(process.stdout)

    def _get_time(self, stdout: str = "") -> float:
        matches = CYCLES_RE.findall(stdout)
        if not matches:
            if self.verbose:
                print("NO CYCLES_US IN OUTPUT")
            return float("inf")
        return float(matches[-1])
