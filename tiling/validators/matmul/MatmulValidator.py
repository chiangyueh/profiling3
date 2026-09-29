from tiling.base import BaseValidator, BaseParam
from tiling.limits import MatmulLimits
from typing import Callable


class MatmulValidator(BaseValidator):
    def __init__(self,
                 limits: MatmulLimits,
                 param_funcs: dict[
                     str, list[tuple[Callable[[dict[str, BaseParam]], bool | tuple[bool]],
                                     Callable[[dict[str, BaseParam]], dict[BaseParam]]]]] | None = None
                 ) -> None:
        param_funcs = dict(param_funcs or {})
        param_funcs.update({
            'base_tiles': (self._base_tiles_is_valid,
                           self._repair_base_tiles),
            'l1_div_stepK': (self._div_stepK_is_valid,
                             self._repair_div_stepK),
            'l1_size': (self._l1_size_is_valid,
                        self._repair_l1_size),
                })
        super().__init__(limits, param_funcs)
        self.limits: MatmulLimits

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
                            'MM_DEPTH_A1', 'MM_DEPTH_B1']
        baseM, baseN, baseK, depthA1, depthB1 = [params[name].value for name in used_param_names]
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

        repaired_params = self.repair_dijkstra(movable_params, is_valid, context=params)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}

    def _repair_div_stepK(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_STEP_Ka', 'MM_STEP_Kb']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)

        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values}
            return self._div_stepK_is_valid(new_params)

        repaired_params = self.repair_dijkstra(movable_params, is_valid, context=params)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}

    def _repair_l1_size(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_BASE_M', 'MM_BASE_N', 'MM_BASE_K',
                            'MM_STEP_M', 'MM_STEP_N', 'MM_STEP_Ka', 'MM_STEP_Kb']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)
        add_ctx = {name: params[name] for name in ['MM_DB_L0A', 'MM_DB_L0B', 'MM_DB_L0C']}

        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values, **add_ctx}
            l1 = self._l1_size_is_valid(new_params)
            l0 = all(self._base_tiles_is_valid(new_params))
            div = self._div_stepK_is_valid(new_params)
            return l1 and l0 and div

        repaired_params = self.repair_dijkstra(movable_params, is_valid, context=params)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}