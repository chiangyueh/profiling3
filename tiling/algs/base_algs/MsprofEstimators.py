from __future__ import annotations

from tiling.base.Base import BaseAlgo, BaseParam
from pathlib import Path
import os
import shutil
import subprocess
import json
import pandas as pd


def _drop_stale_output() -> None:
    output = Path("output/output.bin")
    if output.exists():
        output.unlink()


class BaseAlgoMsprof(BaseAlgo):
    def _run_estimator(self, params: list[BaseParam]) -> float:
        env = dict(os.environ)
        for param in params:
            env[param.name] = str(param.value)
        _drop_stale_output()
        for d in Path(".").glob("OPPROF_*"):
            shutil.rmtree(d)
        subprocess.run(["bash", self.runner, "-r", "sim"], env=env,
                               capture_output=True, text=True)
        subprocess.run(["bash", self.runner, "-r", "cpu"], env=env,
                       capture_output=True, text=True)

        dur = self._get_time()
        return dur

    def _get_time(self) -> float:
        try:
            opprofs = list(Path('.').glob("OPPROF_*"))
            if not opprofs:
                if self.verbose:
                    print("NOT OPPROF")
                return float("inf")
            opprof = max(opprofs, key=lambda d: d.stat().st_mtime)
            traces = list(opprof.rglob("trace.json"))
            latency = 0.0
            found = False
            for tj in traces:
                with open(tj) as f:
                    events = json.load(f).get("traceEvents", [])
                ts = [e["ts"] for e in events if "ts" in e]
                if not ts:
                    continue
                span = max(e["ts"] + e.get("dur", 0) for e in events if "ts" in e) - min(ts)
                latency = max(latency, span)
                found = True
            if not found:
                return float("inf")
        except Exception:
            latency = float("inf")
        return latency


class BaseAlgoProfile(BaseAlgo):
    def _run_estimator(self, params: list[BaseParam]) -> float:
        env = dict(os.environ)
        for param in params:
            env[param.name] = str(param.value)

        cceprint = Path("cceprint")
        if cceprint.exists():
            for f in cceprint.glob("*.cce"):
                f.unlink()
        _drop_stale_output()
        for d in Path(".").glob("OPPROF_*"):
            shutil.rmtree(d)
        subprocess.run(["bash", self.runner, "-r", "npu"], env=env,
                       capture_output=True, text=True)

        dur = self._get_time()
        return dur
    
    def _get_time(self) -> float:
        try:
            opprofs = list(Path('.').glob("OPPROF_*"))
            if not opprofs:
                return float("inf")
            opprof = max(opprofs, key=lambda d: d.stat().st_mtime)
            latency = pd.read_csv(f"{opprof}/OpBasicInfo.csv")['Task Duration(us)'].iloc[0]
        except Exception:
            latency = float("inf")
        return latency
