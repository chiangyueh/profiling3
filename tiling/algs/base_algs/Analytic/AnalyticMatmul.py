from __future__ import annotations

from tiling.base.Base import BaseAlgo, BaseParam
from .cost_model import Workload, Schedule, cost, ceil_div, ASCEND_910B3
from .parameters import TILING_ENABLE
import subprocess
import os


class BaseAlgoAnalyticMatmul(BaseAlgo):
    def _run_estimator(self, params: list[BaseParam]) -> float:
        env = dict(os.environ)
        for param in params:
            env[param.name] = str(param.value)
        subprocess.run(["bash", self.runner, "-r", "cpu"], env=env,
                       capture_output=True, text=True)

        dur = self._get_time(params)
        return dur
    
    def _get_time(self, params: list[BaseParam]) -> float:
        p = {param.name: int(param.value) for param in params}
        workload = Workload(
            m=p["MM_M"], n=p["MM_N"], k=p["MM_K"],
            dtype="fp16", trans_a=False, trans_b=False,
        )

        baseM = p.get("MM_BASE_M", (p['MM_M'] + 16 - 1) // 256); baseN = p["MM_BASE_N"]; baseK = p["MM_BASE_K"]
        stepKa = p["MM_STEP_Ka"]; stepKb = p["MM_STEP_Kb"]
        stepM = p.get("MM_STEP_M", 1); stepN = p.get("MM_STEP_N", 1)
        dbL0A = p.get("MM_DB_L0A", 2); dbL0B = p.get("MM_DB_L0B", 2); dbL0C = p.get("MM_DB_L0C", 2)
        singleCoreM = p.get("MM_SINGLE_M", baseM)
        singleCoreN = p.get("MM_SINGLE_N", baseN)

        singleCoreK = workload.k         
        depthA1 = stepM * stepKa * 2
        depthB1 = stepN * stepKb * 2
        mCnt = ceil_div(workload.m, singleCoreM)
        nCnt = ceil_div(workload.n, singleCoreN)
        usedCoreNum = max(1, min(mCnt * nCnt, ASCEND_910B3.aic_cores))
        l2MTileBlock = max(1, mCnt)
        l2NTileBlock = max(1, nCnt)
        l2MTileCnt = ceil_div(mCnt, l2MTileBlock)    
        l2NTileCnt = ceil_div(nCnt, l2NTileBlock)   

        schedule = Schedule.make(
            usedCoreNum=usedCoreNum,
            singleCoreM=singleCoreM, singleCoreN=singleCoreN, singleCoreK=singleCoreK,
            baseM=baseM, baseN=baseN, baseK=baseK,
            depthA1=depthA1, depthB1=depthB1,
            stepM=stepM, stepN=stepN, stepKa=stepKa, stepKb=stepKb,
            dbL0A=dbL0A, dbL0B=dbL0B, dbL0C=dbL0C,
            l2MTileCnt=l2MTileCnt, l2NTileCnt=l2NTileCnt,
            l2MTileBlock=l2MTileBlock, l2NTileBlock=l2NTileBlock,
            l2IterateOrder=0,
            tilingEnable=TILING_ENABLE["BASE"],
            **self._zero_rest(),
        )

        c = cost(workload, schedule)
        return c["nominal_cycles"]

    @staticmethod
    def _zero_rest() -> dict[str, int]:
        from .parameters import FIELDS
        explicit = {
            "usedCoreNum","singleCoreM","singleCoreN","singleCoreK",
            "baseM","baseN","baseK","depthA1","depthB1","stepM","stepN",
            "stepKa","stepKb","dbL0A","dbL0B","dbL0C",
            "l2MTileCnt","l2NTileCnt","l2MTileBlock","l2NTileBlock","l2IterateOrder",
            "tilingEnable",
        }
        return {f: 0 for f in FIELDS if f not in explicit}
