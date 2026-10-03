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
    max_s1_base: int | None = None
    max_s2_base: int | None = None
    max_n_ratio: int | None = 8

    def __init__(self, limits: AttentionLimits) -> None:
        super().__init__(
            limits,
            "FA",
            {
                "route": (self._route_is_valid, self._repair_route),
                "tile_geometry": (self._tile_is_valid, self._repair_tile),
                "parallel_window": (self._parallel_window_is_valid, self._repair_tile),
            },
        )

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        raise NotImplementedError

    @staticmethod
    def _repair_route(params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return {}

    def _tile_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        required = self.tile_names + ("FA_S1", "FA_S2")
        if any(name not in params for name in required):
            return (False,)
        s1_base, s2_base, n_ratio = (params[name].value for name in self.tile_names)
        if s1_base <= 0 or s2_base <= 0 or n_ratio <= 0:
            return (False,)
        s1_align = self._align_up(params["FA_S1"].value, 16)
        s2_align = self._align_up(params["FA_S2"].value, 16)
        tail = s2_align % s2_base or s2_base
        has_drop = self._value(params, "FA_HAS_DROP", 0) == 1
        s1_limit = min(s1_align, self.max_s1_base or s1_align)
        s2_limit = min(s2_align, self.max_s2_base or s2_align)
        return (
            s1_base % 16 == 0,
            s2_base % 16 == 0,
            s1_base <= s1_limit,
            s2_base <= s2_limit,
            not has_drop or (s2_base > 32 and tail > 32),
            *self._route_tile_is_valid(params),
        )

    def _route_tile_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        """Resource rules that differ between the official FA templates."""

        return (True,)

    def _parallel_window_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if any(name not in params for name in self.tile_names):
            return False
        s2_base = params["FA_S2_BASE"].value
        n_ratio = params["FA_N_RATIO"].value
        if s2_base <= 0 or n_ratio <= 0:
            return False
        # SetCoreParams stores min(GetNRatio(), s2OuterSize).  Values above
        # this range do not describe the effective tiling and only alias a
        # smaller ratio, so reject them before the optimizer treats them as a
        # distinct configuration.
        s2_outer = self._ceil_div(params["FA_S2"].value, s2_base)
        limit = s2_outer
        if self.max_n_ratio is not None:
            limit = min(limit, self.max_n_ratio)
        return n_ratio <= max(1, limit)

    def _all_tile_constraints(self, params: dict[str, BaseParam]) -> bool:
        return (
            all(self._tile_is_valid(params))
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

    # CalcS1S2BasicBlock selects at most 128 elements on both split axes.
    max_s1_base = 128
    max_s2_base = 128

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

    # SameAB normally uses 256 and may select the official 384 best block.
    max_s1_base = 384
    max_s2_base = 128

    def _route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        return (
            all(self._shape_is_valid(params))
            and params["FA_LAYOUT"].value != LAYOUT_TND
            and _same_ab_shape(params)
        )

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if derived:
            derived.update(self._dense_derived(values))
            derived["FA_ROUTE"] = FA_ROUTE_SAME_AB
        return self._derived_params(derived)


class FlashAttentionScoreS1Validator(_FlashAttentionScoreTilingValidator):
    """Validator for priority-97 ``flash_attention_score_s1_bn2gs1``."""

    # The basic S1 block is at most 128 and the Host may combine at most four
    # adjacent blocks. S2 is never tiled above 128 in this template.
    max_s1_base = 128 * 4
    max_s2_base = 128
    max_n_ratio = 4

    def _route_tile_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        # SetCoreParams grows the effective S1 block only while its full-S2
        # workspace stays within workspaceLimit = 128 Ki elements.
        return (
            params["FA_S1_BASE"].value
            * self._align_up(params["FA_S2"].value, 16)
            <= 128 * 1024,
        )

    def _parallel_window_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if "FA_N_RATIO" not in params:
            return False
        n_ratio = params["FA_N_RATIO"].value
        return 0 < n_ratio <= self.max_n_ratio

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

    max_s1_base = 256
    max_n_ratio = 1

    def _route_tile_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        # FlashAttentionScoreTilingB derives S1 from blockBUBSizeLimit_ (8192
        # score elements). This is a tile-local limit, independent of the
        # route's separate 128-KiB whole-shape admission check.
        return (
            params["FA_S1_BASE"].value * params["FA_S2_BASE"].value <= 8 * 1024,
        )

    def _parallel_window_is_valid(self, params: dict[str, BaseParam]) -> bool:
        # The B template has no S1/S2 N:1 combination in SetCoreParams.
        return (
            "FA_N_RATIO" in params
            and params["FA_N_RATIO"].value == self.max_n_ratio
        )

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

    # TND defaults to 128, may choose 256/512 from actual-sequence statistics,
    # and always caps the S2 block at 128.
    max_s1_base = 512
    max_s2_base = 128

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
