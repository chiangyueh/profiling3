from .data import TilingInfo, HardwareInfo
from tiling.base.Base import BaseAlgo, BaseParam
from tiling.limits import MatmulLimits
import numpy as np
from .cost_models import CostModel, CostModelBase, CostModelSingleSplitK, CostModelDetSplitK


class AlgoModel(BaseAlgo):
    def _get_model(self, tiling: TilingInfo, limits: MatmulLimits) -> CostModel:
        raise NotImplementedError
    def _run_estimator(self, params: list[BaseParam]) -> float:
        ps_dict = {param.name:param.value for param in params}
        tiling = TilingInfo(
            baseM=ps_dict['MM_BASE_M'],
            baseN=ps_dict['MM_BASE_N'],
            baseK=ps_dict['MM_BASE_K'],
            singleCoreM=ps_dict['MM_BASE_M'],
            singleCoreN=ps_dict['MM_BASE_N'],
            singleCoreK=ps_dict['MM_K'],
            mTileBlock=ps_dict['MM_M_TILE_BLOCK'],
            nTileBlock=ps_dict['MM_N_TILE_BLOCK'],
            stepM=ps_dict['MM_STEP_M'],
            stepN=ps_dict['MM_STEP_N'],
            stepKa=ps_dict['MM_STEP_Ka'],
            stepKb=ps_dict['MM_STEP_Kb'],
            dbL0A=ps_dict['MM_DB_L0A'],
            dbL0B=ps_dict['MM_DB_L0B'],
            dbL0C=ps_dict['MM_DB_L0C'], 
        )
        limits = MatmulLimits(
            domains=None, max_cores=24, 
            L0A_size=64 * 1024, L0B_size=64 * 1024, L0C_size=128 * 1024,
            L1_size=512 * 1024, L2_size=192 * 1024**2,dtype_size=2)
        cost_model = self._get_model(tiling, limits)
        traffic = cost_model(ps_dict['MM_M'], ps_dict['MM_N'], ps_dict['MM_K'])
        cycles = cost_model.time_estimate()['cycles']
        gm_to_l2 = traffic.gm_to_l2
        l2_to_l1 = traffic.l2_to_l1
        l1_to_l0 = traffic.l1_to_l0
        print(f"MAX CYCLES: {max(gm_to_l2, l2_to_l1, l1_to_l0)}")
        return max(gm_to_l2, l2_to_l1)
    
    def _is_right(self, absolute_tol = 1e-9, error_tol = 0.0001):
        return True

class BaseAlgoModel(AlgoModel):
    def _get_model(self, tiling: TilingInfo, limits: MatmulLimits) -> CostModelBase:
        return CostModelBase(tiling=tiling, limits=limits, hardware=HardwareInfo())

class SingleSplitKAlgoModel(AlgoModel):
    def _get_model(self, tiling: TilingInfo, limits: MatmulLimits) -> CostModelSingleSplitK:
        return CostModelSingleSplitK(tiling=tiling, limits=limits, hardware=HardwareInfo())

class DetSplitKAlgoModel(AlgoModel):
    def _get_model(self, tiling: TilingInfo, limits: MatmulLimits) -> CostModelDetSplitK:
        return CostModelDetSplitK(tiling=tiling, limits=limits, hardware=HardwareInfo())
