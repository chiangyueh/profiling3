from __future__ import annotations

from tiling.base import BaseParam
from tiling.limits import AttentionLimits

from .AttentionValidator import AttentionValidator, LAYOUT_BNSD


FA_ROUTE_SAME_AB = 95
FA_ROUTE_GENERAL = 96


def same_ab_shape(params: dict[str, BaseParam]) -> bool:
    dtype_bytes = params["FA_DTYPE_BYTES"].value
    s2 = params["FA_S2"].value
    d = params["FA_D"].value
    return dtype_bytes != 4 and (
        ((d % 16 != 0 or d == 96) and s2 >= 512)
        or (s2 > 1024 and 128 < d < 196)
        or (d == 64 and s2 % 64 != 0 and 2048 < s2 < 18432)
    )


class FlashAttentionScoreValidator(AttentionValidator):
    tile_names = ("FA_S1_BASE", "FA_S2_BASE", "FA_N_RATIO")

    def __init__(self, limits: AttentionLimits) -> None:
        super().__init__(
            limits,
            {
                "route": (self._route_is_valid, self._repair_route),
                "tile": (self._tile_is_valid, self._repair_tile),
            },
        )

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        raise NotImplementedError

    @staticmethod
    def _repair_route(params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return {}

    def _tile_is_valid(self, params: dict[str, BaseParam]) -> bool:
        required = self.tile_names + ("FA_S1", "FA_S2")
        if any(name not in params for name in required):
            return False
        s1_base, s2_base, ratio = (params[name].value for name in self.tile_names)
        if s1_base <= 0 or s2_base <= 0 or ratio <= 0:
            return False
        s2_outer = self._ceil_div(params["FA_S2"].value, s2_base)
        return (
            s1_base % 16 == 0
            and s2_base % 16 == 0
            and ratio <= max(1, s2_outer)
            and self._route_tile_is_valid(params)
        )

    def _route_tile_is_valid(self, params: dict[str, BaseParam]) -> bool:
        raise NotImplementedError

    def _repair_tile(self, params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return self._repair_names(params, self.tile_names, self._tile_is_valid)

    def _dense_derived(self, values: dict[str, int]) -> dict[str, int]:
        if any(name not in values for name in self.tile_names):
            return {}
        s1_base = values["FA_S1_BASE"]
        s2_base = values["FA_S2_BASE"]
        ratio = values["FA_N_RATIO"]
        if s1_base <= 0 or s2_base <= 0 or ratio <= 0:
            return {}
        s1_outer = self._ceil_div(values["FA_S1"], s1_base)
        raw_s2_outer = self._ceil_div(values["FA_S2"], s2_base)
        effective_ratio = min(ratio, raw_s2_outer)
        total = values["FA_B"] * values["FA_N1"] * s1_outer
        core_num = min(total, self.limits.max_cores)
        return {
            "FA_S1_OUTER": s1_outer,
            "FA_S1_TAIL": values["FA_S1"] % s1_base or s1_base,
            "FA_S2_OUTER": self._ceil_div(raw_s2_outer, effective_ratio),
            "FA_S2_TAIL": values["FA_S2"] % s2_base or s2_base,
            "FA_S2_WINDOW": s2_base * effective_ratio,
            "FA_TOTAL_TASKS": total,
            "FA_CORE_NUM": core_num,
            "FA_SPLIT_FACTOR": self._ceil_div(total, core_num),
        }


class FlashAttentionScoreGeneralValidator(FlashAttentionScoreValidator):
    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if not all(self._shape_is_valid(params)):
            return False
        return (
            (params["FA_D"].value != params["FA_DV"].value or params["FA_S2"].value > 1024)
            and not same_ab_shape(params)
        )

    def _route_tile_is_valid(self, params: dict[str, BaseParam]) -> bool:
        s1_base = params["FA_S1_BASE"].value
        s2_window = params["FA_S2_BASE"].value * params["FA_N_RATIO"].value
        d = params["FA_D"].value
        fixed_n_valid = not (
            64 < d <= 128
            and params["FA_DTYPE_BYTES"].value != 4
            and s2_window < 128
        )
        fixed_ub = 2 * 16 * 1024 + 3 * 8 * 1024 * self.limits.calc_type_size
        bmm1_nz = params["FA_S2"].value % 64 != 0 and d != 64
        fixed_ub += 35 * 1024 if bmm1_nz else 8 * 1024 * self.limits.calc_type_size
        softmax_ub = 5 * s1_base * 4 * 8
        return (
            s2_window <= 1024
            and fixed_n_valid
            and fixed_ub + softmax_ub <= self.limits.UB_size
        )

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if derived:
            derived.update(self._dense_derived(values))
            derived["FA_ROUTE"] = FA_ROUTE_GENERAL
        return self._derived_params(derived)


class FlashAttentionScoreSameABValidator(FlashAttentionScoreValidator):
    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        return all(self._shape_is_valid(params)) and same_ab_shape(params)

    def _route_tile_is_valid(self, params: dict[str, BaseParam]) -> bool:
        s1 = params["FA_S1"].value
        s2 = params["FA_S2"].value
        d = params["FA_D"].value
        dv = params["FA_DV"].value
        layout = params["FA_LAYOUT"].value
        s1_base = params["FA_S1_BASE"].value
        s2_base = params["FA_S2_BASE"].value
        ratio = params["FA_N_RATIO"].value
        if s1_base > min(self._align_up(s1, 16), 384):
            return False
        nominal_window = s2_base * ratio
        full_window = min(s2, nominal_window)
        tail_window = s2 % nominal_window or full_window
        unsplit_k = (
            layout == LAYOUT_BNSD
            and d == 192
            and s1 % 256 == 0
            and s2 % 128 == 0
        )
        if unsplit_k:
            tail_s1 = s1 % s1_base or s1_base
            return (
                s1_base in (128, 256)
                and tail_s1 % 128 == 0
                and full_window <= 1024
                and full_window % s1_base == 0
                and tail_window % s1_base == 0
            )
        fixed_m = min(128, self._align_up(s1, 16))
        return (
            full_window <= 1024
            and (s2 % 64 != 0 or full_window % 64 == 0)
            and (not (64 < d <= 128 or 64 < dv <= 128) or s1_base >= fixed_m)
        )

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if derived:
            derived.update(self._dense_derived(values))
            derived["FA_ROUTE"] = FA_ROUTE_SAME_AB
        return self._derived_params(derived)
