from tiling.base import BaseParam
from .MatmulValidator import MatmulValidator
from tiling.limits import MatmulLimits
from typing import Callable


class MatmulBaseKernelValidator(MatmulValidator):
    def __init__(self,
                 limits: MatmulLimits,
                 param_funcs: dict[
                     str, list[tuple[Callable[[dict[str, BaseParam]], bool | tuple[bool]],
                                     Callable[[dict[str, BaseParam]], dict[BaseParam]]]]] | None = None
                 ) -> None:
        super().__init__(limits, param_funcs)
        self.param_funcs.update({
            "l2": (self._l2_is_valid,
                   self._repair_l2)
        })

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        ps_dict = {p.name: p for p in params}
        K = ps_dict['MM_K'].value
        baseM = ps_dict['MM_BASE_M'].value
        baseN = ps_dict['MM_BASE_N'].value
        stepKa = ps_dict['MM_STEP_Ka'].value
        stepKb = ps_dict['MM_STEP_Kb'].value

        # MM_CORE_NUM не выводим: хост берёт min(mCnt * nCnt, aicNum) без перебора,
        # но это его эвристика, а не требование ядра — на части тайлингов выгоднее
        # взять меньше ядер, поэтому параметр остаётся свободным.
        derived = {
            'MM_SINGLE_M': baseM,
            'MM_SINGLE_N': baseN,
            'MM_SINGLE_K': K,
            'MM_STEP_M': 1,
            'MM_STEP_N': 1,
            'MM_DEPTH_A1': stepKa * 2,
            'MM_DEPTH_B1': stepKb * 2,
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
                           'MM_STEP_M', 'MM_STEP_N',
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

        repaired_params = self.repair_dijkstra(movable_params, is_valid, context=params)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}