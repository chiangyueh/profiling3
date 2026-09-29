from __future__ import annotations

from tiling.base import BaseValidator, BaseParam
from tiling.limits import MatmulLimits
    
class MatmulValidator(BaseValidator):
    def __init__(self,
                 limits: MatmulLimits
                 ) -> None:
        param_funcs = {
            'base_tiles': (self._base_tiles_is_valid,
                           self._repair_base_tiles),
            'l1_div_stepK': (self._div_stepK_is_valid,
                             self._repair_div_stepK),
            'l1_size': (self._l1_size_is_valid,
                        self._repair_l1_size),
                }
        super().__init__(limits, param_funcs)
        self.limits: MatmulLimits
        
    
    def _make_param(self, name: str, value: int, is_const: bool, domain: list[int] | None = None) -> BaseParam:
        return BaseParam(name=name, value=value, is_const=is_const, domain=domain or [value])
                    
    def _base_tiles_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool]:
        used_param_names = ['MM_BASE_M', 'MM_BASE_N', 'MM_BASE_K',
                            'MM_DB_L0A', 'MM_DB_L0B', 'MM_DB_L0C']
        baseM, baseN, baseK, dbL0A, dbL0B, dbL0C = [params[name].value for name in used_param_names]
        is_valid_L0 = (baseM * baseK * self.limits.dtype_size * dbL0A <= self.limits.L0A_size,
                       baseN * baseK * self.limits.dtype_size * dbL0B <= self.limits.L0B_size,
                       baseM * baseN * 4 * dbL0C <= self.limits.L0C_size)
        return is_valid_L0

    def _l1_size_is_valid(self, params: dict[str, BaseParam]) -> bool:
        used_param_names = ['MM_BASE_M', 'MM_BASE_N', 'MM_BASE_K',
                            'MM_STEP_M', 'MM_STEP_N','MM_STEP_Ka', 'MM_STEP_Kb',
                            'MM_DB_L0A', 'MM_DB_L0B']
        baseM, baseN, baseK, stepM, stepN, stepKa, stepKb, dbL0A, dbL0B = [params[name].value
                                                                           for name in used_param_names]
        depthA1 = stepM * stepKa * dbL0A
        depthB1 = stepN * stepKb * dbL0B
        return (baseM * depthA1 + baseN * depthB1) * baseK * self.limits.dtype_size <= self.limits.L1_size
    
    def _div_stepK_is_valid(self, params: dict[str, BaseParam]) -> bool:
        used_param_names = ['MM_STEP_Ka', 'MM_STEP_Kb']
        stepKa, stepKb = [params[name].value for name in used_param_names]
        return stepKa % stepKb == 0 or stepKb % stepKa == 0

    def _repair_base_tiles(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_BASE_M', 'MM_BASE_N', 'MM_BASE_K',
                            'MM_DB_L0A', 'MM_DB_L0B', 'MM_DB_L0C']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)

        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values}
            return all(self._base_tiles_is_valid(new_params))

        repaired_params = self.repair_dijkstra(movable_params, is_valid)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}


    def _repair_div_stepK(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_STEP_Ka', 'MM_STEP_Kb']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)

        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values}
            return self._div_stepK_is_valid(new_params)

        repaired_params = self.repair_dijkstra(movable_params, is_valid)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}

    def _repair_l1_size(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        
        used_param_names = ['MM_BASE_M', 'MM_BASE_N', 'MM_BASE_K',
                            'MM_STEP_Ka', 'MM_STEP_Kb']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)
        add_ctx = {name: params[name] for name in ['MM_DB_L0A', 'MM_DB_L0B','MM_DB_L0C']}

        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values, **add_ctx}
            l1 = self._l1_size_is_valid(new_params)
            l0 = all(self._base_tiles_is_valid(new_params))
            div = self._div_stepK_is_valid(new_params)
            return l1 and l0 and div

        repaired_params = self.repair_dijkstra(movable_params, is_valid)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}
        
class MatmulBaseKernelValidator(MatmulValidator):
    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        ps_dict = {p.name: p for p in params}

        M = ps_dict['MM_M'].value
        N = ps_dict['MM_N'].value
        K = ps_dict['MM_K'].value
        baseM = ps_dict['MM_BASE_M'].value
        baseN = ps_dict['MM_BASE_N'].value
        stepKa = ps_dict['MM_STEP_Ka'].value
        stepKb = ps_dict['MM_STEP_Kb'].value
        mTileBlock = ps_dict['MM_M_TILE_BLOCK'].value
        nTileBlock = ps_dict['MM_N_TILE_BLOCK'].value

        derived = {
            'MM_CORE_NUM': min(baseM * baseN, self.limits.max_cores),
            'MM_SINGLE_M': baseM,
            'MM_SINGLE_N': baseN,
            'MM_SINGLE_K': K,
            'MM_STEP_M': 1,
            'MM_STEP_N': 1,
            'MM_DEPTH_A1': stepKa * 2,
            'MM_DEPTH_B1': stepKb * 2,
            'MM_M_TILE_CNT_L2': self._ceil_div(self._ceil_div(M, baseM), mTileBlock),
            'MM_N_TILE_CNT_L2': self._ceil_div(self._ceil_div(N, baseN), nTileBlock),
        }
        return [self._make_param(name, value, True) for name, value in derived.items()]
    
    def _l2_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool]:
        used_param_names = ['MM_M', 'MM_N', 'MM_K',
                            'MM_BASE_M', 'MM_BASE_N',
                            'MM_M_TILE_BLOCK', 'MM_N_TILE_BLOCK']
        M, N, K, baseM, baseN, mTileBlock, nTileBlock = [params[name].value for name in used_param_names]
        
        mTotalCnt = (M + baseM - 1) // baseM
        nTotalCnt = (N + baseN - 1) // baseN

        a_stripes = mTileBlock * baseM * K * self.limits.dtype_size
        b_stripes = nTileBlock * baseN * K * self.limits.dtype_size 
            
        return (mTileBlock <= mTotalCnt, nTileBlock <= nTotalCnt,
                a_stripes + b_stripes <= self.limits.L2_size)
        
    def _repair_l2(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_BASE_M', 'MM_BASE_N',
                            'MM_M_TILE_BLOCK', 'MM_N_TILE_BLOCK']
        add_param_names = ['MM_M', 'MM_N', 'MM_K', 'MM_BASE_K',
                            'MM_STEP_Ka', 'MM_STEP_Kb',
                            'MM_DB_L0A', 'MM_DB_L0B', 'MM_DB_L0C']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)
        add_ctx = {name: params[name] for name in add_param_names}
    
        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values, **add_ctx}
            l2 = all(self._l2_is_valid(new_params))
            l1 = self._l1_size_is_valid(new_params)
            l0 = all(self._base_tiles_is_valid(new_params))
            return l2 and l1 and l0
    
        repaired_params = self.repair_dijkstra(movable_params, is_valid)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}
