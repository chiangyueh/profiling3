from __future__ import annotations

from tiling.base import BaseParam
from tiling.limits import AttentionLimits

from .AttentionValidator import AttentionValidator, LAYOUT_TND


FA_ROUTE_DROP_ADAPTER = 90
FA_ROUTE_VARLEN = 94
FA_ROUTE_SAME_AB = 95
FA_ROUTE_GENERAL = 96
FA_ROUTE_S1 = 97
FA_ROUTE_B = 98


class _FlashAttentionScoreTilingValidator(AttentionValidator):
    tile_names = ("FA_S1_BASE", "FA_S2_BASE", "FA_N_RATIO")

    def __init__(self, limits: AttentionLimits) -> None:
        super().__init__(
            limits,
            "FA",
            {
                "route": (self._route_is_valid, self._repair_route),
                "tile_geometry": (self._tile_is_valid, self._repair_tile),
                "local_memory": (self._local_memory_is_valid, self._repair_tile),
                "parallel_window": (self._parallel_window_is_valid, self._repair_tile),
            },
        )

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        raise NotImplementedError

    @staticmethod
    def _repair_route(params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return {}

    def _tile_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if any(name not in params for name in self.tile_names):
            return (False,)
        s1_base, s2_base, n_ratio = (params[name].value for name in self.tile_names)
        if s1_base <= 0 or s2_base <= 0 or n_ratio <= 0:
            return (False,)
        s2_align = self._align_up(params["FA_S2"].value, 16)
        tail = s2_align % s2_base or s2_base
        has_drop = self._value(params, "FA_HAS_DROP", 0) == 1
        return (
            s1_base > 0 and s1_base % 16 == 0,
            s2_base > 0 and s2_base % 16 == 0,
            n_ratio > 0,
            not has_drop or (s2_base > 32 and tail > 32),
        )

    def _local_memory_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if any(name not in params for name in self.tile_names[:2]):
            return (False,)
        # SetCoreParams and the kernels operate on the real tail when a
        # requested base is larger than the aligned sequence length.
        s1_base = min(
            params["FA_S1_BASE"].value,
            self._align_up(params["FA_S1"].value, 16),
        )
        s2_base = min(
            params["FA_S2_BASE"].value,
            self._align_up(params["FA_S2"].value, 16),
        )
        # flash_attention_score_tiling_general.cpp uses a two-times FP32 API
        # buffer for the split-S1/S2 families. L0C holds one FP32 score tile.
        api_ub = 2 * s1_base * s2_base * 4
        score_l0c = s1_base * s2_base * 4
        return (
            api_ub <= self.limits.UB_size,
            score_l0c <= self.limits.L0C_size,
        )

    def _parallel_window_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if any(name not in params for name in self.tile_names):
            return False
        if params["FA_S2_BASE"].value <= 0:
            return False
        # The official SetCoreParams clamps a requested ratio to the number
        # of available S1/S2 outer blocks before storing it in tiling data.
        return params["FA_N_RATIO"].value > 0

    def _all_tile_constraints(self, params: dict[str, BaseParam]) -> bool:
        return (
            all(self._tile_is_valid(params))
            and all(self._local_memory_is_valid(params))
            and self._parallel_window_is_valid(params)
        )

    def _repair_tile(self, params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return self._repair_names(params, self.tile_names, self._all_tile_constraints)

    def _dense_derived(self, values: dict[str, int]) -> dict[str, int]:
        if any(name not in values for name in self.tile_names):
            return {}
        s1 = values["FA_S1"]
        s2 = values["FA_S2"]
        s1_base = values["FA_S1_BASE"]
        s2_base = values["FA_S2_BASE"]
        if s1_base <= 0 or s2_base <= 0 or values["FA_N_RATIO"] <= 0:
            return {}
        s1_outer = self._ceil_div(s1, s1_base)
        s2_outer_raw = self._ceil_div(s2, s2_base)
        n_ratio = min(values["FA_N_RATIO"], s2_outer_raw)
        total = values["FA_B"] * values["FA_N2"] * (values["FA_N1"] // values["FA_N2"]) * s1_outer
        core_num = min(total, self.limits.max_cores)
        return {
            "FA_S1_OUTER": s1_outer,
            "FA_S1_TAIL": s1 % s1_base or s1_base,
            "FA_S2_OUTER": self._ceil_div(s2_outer_raw, n_ratio),
            "FA_S2_TAIL": s2 % s2_base or s2_base,
            "FA_S2_WINDOW": s2_base * n_ratio,
            "FA_D_BASE": min(128, self._align_up(values["FA_D"], 16)),
            "FA_TOTAL_TASKS": total,
            "FA_CORE_NUM": core_num,
            "FA_SPLIT_FACTOR": self._ceil_div(total, core_num),
        }


class FlashAttentionScoreGeneralValidator(_FlashAttentionScoreTilingValidator):
    """Validator for priority-96 ``flash_attention_score_s1s2_bn2gs1``."""

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if not all(self._shape_is_valid(params)):
            return False
        dtype_bytes = params["FA_DTYPE_BYTES"].value
        layout = params["FA_LAYOUT"].value
        s2 = params["FA_S2"].value
        d = params["FA_D"].value
        dv = params["FA_DV"].value

        same_ab_preempts = dtype_bytes != 4 and (
            ((d % 16 != 0 or d == 96) and s2 >= 512)
            or (s2 > 1024 and 128 < d < 196)
            or (d == 64 and s2 % 64 != 0 and 2048 < s2 < 18432)
        )
        return layout != LAYOUT_TND and (d != dv or s2 > 1024) and not same_ab_preempts

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if not derived:
            return []
        derived.update(self._dense_derived(values))
        derived["FA_ROUTE"] = FA_ROUTE_GENERAL
        return self._derived_params(derived)


def _same_ab_shape(params: dict[str, BaseParam]) -> bool:
    dtype_bytes = params["FA_DTYPE_BYTES"].value
    s2 = params["FA_S2"].value
    d = params["FA_D"].value
    return dtype_bytes != 4 and (
        ((d % 16 != 0 or d == 96) and s2 >= 512)
        or (s2 > 1024 and 128 < d < 196)
        or (d == 64 and s2 % 64 != 0 and 2048 < s2 < 18432)
    )


class FlashAttentionScoreSameABValidator(_FlashAttentionScoreTilingValidator):
    """Validator for priority-95 ``flash_attention_score_s1s2_*_sab``."""

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        return (
            all(self._shape_is_valid(params))
            and params["FA_LAYOUT"].value != LAYOUT_TND
            and _same_ab_shape(params)
        )

    def _local_memory_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if any(name not in params for name in self.tile_names[:2]):
            return (False,)
        # SameAB CalcUBSize records the API scratch size but does not reject
        # the route by that value. Its BMM1 fixed split is capped at 128x128,
        # so the common full score-tile estimate is not applicable here.
        base_m = min(params["FA_S1_BASE"].value, params["FA_S1"].value, 128)
        base_n = min(params["FA_S2_BASE"].value, params["FA_S2"].value, 128)
        return (base_m * base_n * 4 <= self.limits.L0C_size,)

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if derived:
            derived.update(self._dense_derived(values))
            derived["FA_ROUTE"] = FA_ROUTE_SAME_AB
        return self._derived_params(derived)


class FlashAttentionScoreS1Validator(_FlashAttentionScoreTilingValidator):
    """Validator for priority-97 ``flash_attention_score_s1_bn2gs1``."""

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if not all(self._shape_is_valid(params)) or params["FA_LAYOUT"].value == LAYOUT_TND:
            return False
        if _same_ab_shape(params) or params["FA_S2"].value > 1024:
            return False
        s1_align = self._align_up(params["FA_S1"].value, 16)
        s2_align = self._align_up(params["FA_S2"].value, 16)
        d_align = self._align_up(params["FA_D"].value, 16)
        bytes_per_element = params["FA_DTYPE_BYTES"].value
        n1 = params["FA_N1"].value
        l1_a = n1 * ((s1_align + s2_align) * d_align + s2_align) * bytes_per_element
        l1_b = n1 * (s1_align + d_align) * s2_align * bytes_per_element
        work = n1 * s1_align * s2_align * bytes_per_element
        return params["FA_S2"].value > 128 or l1_a >= self.limits.L1_size or l1_b >= self.limits.L1_size or work > 128 * 1024

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if derived:
            derived.update(self._dense_derived(values))
            derived["FA_ROUTE"] = FA_ROUTE_S1
        return self._derived_params(derived)


class FlashAttentionScoreBValidator(_FlashAttentionScoreTilingValidator):
    """Validator for priority-98 ``flash_attention_score_b``."""

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if not all(self._shape_is_valid(params)) or params["FA_LAYOUT"].value == LAYOUT_TND:
            return False
        if _same_ab_shape(params) or params["FA_S2"].value > 128:
            return False
        s1_align = self._align_up(params["FA_S1"].value, 16)
        s2_align = self._align_up(params["FA_S2"].value, 16)
        work = params["FA_N1"].value * s1_align * s2_align * params["FA_DTYPE_BYTES"].value
        return work <= 128 * 1024

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if derived:
            derived.update(self._dense_derived(values))
            derived["FA_ROUTE"] = FA_ROUTE_B
        return self._derived_params(derived)


class FlashAttentionScoreDropAdapterValidator(FlashAttentionScoreBValidator):
    """Validator for the priority-90 drop-mask adapter plus its terminal route."""

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        adapter = self._value(params, "FA_HAS_DROP", 0) == 1 and (
            params["FA_S2"].value % 8 != 0
            or (params["FA_LAYOUT"].value == LAYOUT_TND and params["FA_B"].value > 1)
        )
        return adapter and super()._route_is_valid(params)

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        derived = super().get_derived_params(params)
        return [p for p in derived if p.name != "FA_ROUTE"] + [
            self._make_param("FA_ROUTE", FA_ROUTE_DROP_ADAPTER, True)
        ]


class FlashAttentionScoreVarLenValidator(_FlashAttentionScoreTilingValidator):
    """Validator for priority-94 TND ``flash_attention_var_len_score``."""

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        return (
            all(self._shape_is_valid(params))
            and params["FA_LAYOUT"].value == LAYOUT_TND
            and self._value(params, "FA_NEEDS_DROP_ADAPTER", 0) == 0
        )

    @staticmethod
    def _is_same_ab(values: dict[str, int]) -> bool:
        batch = values["FA_B"]
        s1 = values["FA_S1"]
        s2 = values["FA_S2"]
        t1 = values.get("FA_T1", batch * s1)
        t2 = values.get("FA_T2", batch * s2)
        hit = (
            (batch >= 4 and t1 >= 8192 and t2 >= 8192 and s1 >= 512 and s2 >= 512)
            or (s1 > 5120 and s2 > 5120)
        )
        return hit and values["FA_DTYPE_BYTES"] != 4 and not (
            values["FA_N1"] == 1 and values["FA_N2"] == 1
        ) and values.get("FA_HIGH_PRECISION_INVALID_LINE", 0) == 0

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if not derived or any(name not in values for name in self.tile_names):
            return self._derived_params(derived)

        s1_base = values["FA_S1_BASE"]
        s2_base = values["FA_S2_BASE"]
        if s1_base <= 0 or s2_base <= 0 or values["FA_N_RATIO"] <= 0:
            return self._derived_params(derived)
        s2_outer_raw = self._ceil_div(values["FA_S2"], s2_base)
        n_ratio = min(values["FA_N_RATIO"], s2_outer_raw)
        s1_blocks = values.get(
            "FA_ACTUAL_S1_BLOCKS",
            values["FA_B"] * self._ceil_div(values["FA_S1"], s1_base),
        )
        total = s1_blocks * values["FA_N2"] * (values["FA_N1"] // values["FA_N2"])
        same_ab = self._is_same_ab(values)
        core_num = min(total, self.limits.aic_num) * 2 if same_ab else min(total, self.limits.max_cores)
        split_cores = core_num // 2 if same_ab else core_num
        derived.update(
            {
                "FA_ROUTE": FA_ROUTE_VARLEN,
                "FA_SAME_AB": int(same_ab),
                "FA_S1_OUTER": self._ceil_div(values["FA_S1"], s1_base),
                "FA_S1_TAIL": values["FA_S1"] % s1_base or s1_base,
                "FA_S2_OUTER": self._ceil_div(s2_outer_raw, n_ratio),
                "FA_S2_TAIL": values["FA_S2"] % s2_base or s2_base,
                "FA_S2_WINDOW": s2_base * n_ratio,
                "FA_D_BASE": min(128, self._align_up(values["FA_D"], 16)),
                "FA_TOTAL_TASKS": total,
                "FA_CORE_NUM": core_num,
                "FA_SPLIT_FACTOR": self._ceil_div(total, split_cores),
            }
        )
        return self._derived_params(derived)
