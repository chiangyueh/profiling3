from __future__ import annotations

from tiling.base import BaseParam
from tiling.limits import AttentionLimits

from .AttentionValidator import AttentionValidator, LAYOUT_BNSD, LAYOUT_TND


FAG_ROUTE_MLA = 1001
FAG_ROUTE_SAME_AB_DETERMINISTIC = 1100
FAG_ROUTE_SAME_AB = 15500
FAG_ROUTE_GENERIC = 16000

_GENERIC_RATIO_PAIRS = {(1, 8), (1, 16), (4, 2), (4, 4)}


class FlashAttentionScoreGradMlaValidator(AttentionValidator):
    """Validator for the fixed priority-1001 MLA Basic FAG kernel.

    This route has no free host tile: the kernel fixes its vector/Cube
    decomposition. An empty domain therefore evaluates exactly one config.
    """

    def __init__(self, limits: AttentionLimits) -> None:
        super().__init__(
            limits,
            "FAG",
            {"mla_route": (self._mla_route_is_valid, self._repair_mla_route)},
        )

    def _mla_route_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if not all(self._shape_is_valid(params)):
            return (False,)
        sparse_mode = self._value(params, "FAG_SPARSE_MODE", 0)
        has_mask = self._value(params, "FAG_HAS_MASK", 0) == 1
        total_q = self._value(params, "FAG_T1", params["FAG_B"].value * params["FAG_S1"].value)
        total_kv = self._value(params, "FAG_T2", params["FAG_B"].value * params["FAG_S2"].value)
        return (
            params["FAG_LAYOUT"].value == LAYOUT_TND,
            params["FAG_DTYPE_BYTES"].value != 4,
            self._value(params, "FAG_DETERMINISTIC", 0) == 0,
            sparse_mode in (0, 2, 3),
            not (sparse_mode == 0 and has_mask),
            not (sparse_mode in (2, 3) and not has_mask),
            self._value(params, "FAG_HAS_PSE", 0) == 0,
            self._value(params, "FAG_HAS_DROP", 0) == 0,
            self._value(params, "FAG_HAS_SINK", 0) == 0,
            self._value(params, "FAG_HAS_ACTUAL_SEQ", 0) == 1,
            self._value(params, "FAG_HAS_START_IDX", 0) == 0,
            params["FAG_D"].value == params["FAG_DV"].value,
            params["FAG_D"].value <= 128,
            params["FAG_D"].value % 16 == 0,
            total_q == total_kv,
            self._value(params, "FAG_EQUAL_ACTUAL_SEQ", 1) == 1,
            params["FAG_S1"].value <= 32768,
        )

    @staticmethod
    def _repair_mla_route(params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return {}

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if derived:
            derived.update(
                {
                    "FAG_ROUTE": FAG_ROUTE_MLA,
                    "FAG_TILING_KEY_SUFFIX": 999,
                    "FAG_CORE_NUM": self.limits.aic_num,
                    "FAG_SFMG_S1": 64,
                    "FAG_SFMG_S2": 128,
                }
            )
        return self._derived_params(derived)


class FlashAttentionScoreGradGenericValidator(AttentionValidator):
    """Validator for priority-16000 Generic S1S2 FAG tiling."""

    tile_names = (
        "FAG_S1_INNER",
        "FAG_S2_INNER",
        "FAG_S1_CV_RATIO",
        "FAG_S2_CV_RATIO",
    )

    def __init__(self, limits: AttentionLimits) -> None:
        super().__init__(
            limits,
            "FAG",
            {
                "generic_route": (self._generic_route_is_valid, self._repair_route),
                "tile_geometry": (self._tile_is_valid, self._repair_tile),
                "local_memory": (self._local_memory_is_valid, self._repair_tile),
                "cv_ratio": (self._cv_ratio_is_valid, self._repair_tile),
            },
        )

    def _generic_route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if not all(self._shape_is_valid(params)):
            return False

        # This is the source-provable Generic envelope used by the current
        # workload.  It excludes every earlier DAV_2201 route: deterministic
        # families, TND/MLA, the FP16/BF16 B/N2/SameAB families, and the
        # short-S1/S2 BN2 family.  Keeping the route check in the validator is
        # important: otherwise a candidate can be accepted even though the
        # host selects another kernel and ignores all FAG_* search parameters.
        return (
            params["FAG_DTYPE_BYTES"].value == 4
            and self._value(params, "FAG_DETERMINISTIC", 0) == 0
            and params["FAG_LAYOUT"].value != LAYOUT_TND
            and max(params["FAG_S1"].value, params["FAG_S2"].value) >= 1024
            and self._value(params, "FAG_IS_SPARSE", 0) == 0
            and self._value(params, "FAG_HAS_MASK", 0) == 0
            and self._value(params, "FAG_HAS_PSE", 0) == 0
            and self._value(params, "FAG_HAS_DROP", 0) == 0
            and self._value(params, "FAG_HAS_ROPE", 0) == 0
            and self._value(params, "FAG_HAS_SINK", 0) == 0
        )

    @staticmethod
    def _repair_route(params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return {}

    def _tile_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if any(name not in params for name in self.tile_names):
            return (False,)
        s1_inner = params["FAG_S1_INNER"].value
        s2_inner = params["FAG_S2_INNER"].value
        return (
            s1_inner > 0 and s1_inner % 16 == 0,
            s2_inner > 0 and s2_inner % 16 == 0,
            s1_inner <= min(128, self._align_up(params["FAG_S1"].value, 16)),
            s2_inner <= min(64, self._align_up(params["FAG_S2"].value, 16)),
        )

    def _local_memory_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if any(name not in params for name in self.tile_names[:2]):
            return (False,)
        s1_inner = params["FAG_S1_INNER"].value
        s2_inner = params["FAG_S2_INNER"].value
        base_mn_bytes = s1_inner * s2_inner * self.limits.calc_type_size
        used_ub = 15 * s1_inner * s2_inner + 32 * s1_inner * self.limits.calc_type_size
        # GetPlatformInfo reserves 8 KiB for MatMul, and the fuzzy split keeps
        # a 33 KiB temporary API budget. Account for both before accepting a
        # candidate tile.
        return (
            base_mn_bytes <= self.limits.L0C_size,
            used_ub + (8 + 33) * 1024 <= self.limits.UB_size,
        )

    def _is_s2_fission_anchor(self, params: dict[str, BaseParam]) -> bool:
        s1_inner = params["FAG_S1_INNER"].value
        s2_inner = params["FAG_S2_INNER"].value
        if s1_inner <= 0 or s2_inner <= 0:
            return False
        s1_outer = self._ceil_div(params["FAG_S1"].value, s1_inner)
        default_s2_cv = s2_inner * 8
        if default_s2_cv == 0 or params["FAG_S2"].value % default_s2_cv != 0:
            return False
        default_tasks = (
            params["FAG_B"].value
            * params["FAG_N2"].value
            * (params["FAG_N1"].value // params["FAG_N2"].value)
            * s1_outer
            * (params["FAG_S2"].value // default_s2_cv)
        )
        return (
            params["FAG_DTYPE_BYTES"].value == 4
            and self._value(params, "FAG_DETERMINISTIC", 0) == 0
            and params["FAG_LAYOUT"].value == LAYOUT_BNSD
            and self._value(params, "FAG_SPARSE_MODE", 0) == 0
            and self._value(params, "FAG_IS_SPARSE", 0) == 0
            and self._value(params, "FAG_HAS_MASK", 0) == 0
            and self._value(params, "FAG_HAS_PSE", 0) == 0
            and self._value(params, "FAG_HAS_DROP", 0) == 0
            and self._value(params, "FAG_HAS_ROPE", 0) == 0
            and params["FAG_D"].value == 128
            and params["FAG_DV"].value == 128
            and s2_inner == 64
            and s1_outer == 1
            and params["FAG_S1"].value == s1_inner
            and self.limits.max_cores == self.limits.aic_num * 2
            and default_tasks == self.limits.max_cores // 2
        )

    def _cv_ratio_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if any(name not in params for name in self.tile_names):
            return False
        pair = (
            params["FAG_S1_CV_RATIO"].value,
            params["FAG_S2_CV_RATIO"].value,
        )
        return pair in _GENERIC_RATIO_PAIRS or (pair == (1, 4) and self._is_s2_fission_anchor(params))

    def _all_tile_constraints(self, params: dict[str, BaseParam]) -> bool:
        return (
            all(self._tile_is_valid(params))
            and all(self._local_memory_is_valid(params))
            and self._cv_ratio_is_valid(params)
        )

    def _repair_tile(self, params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return self._repair_names(params, self.tile_names, self._all_tile_constraints)

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if not derived or any(name not in values for name in self.tile_names):
            return self._derived_params(derived)

        s1_inner = values["FAG_S1_INNER"]
        s2_inner = values["FAG_S2_INNER"]
        s1_ratio = values["FAG_S1_CV_RATIO"]
        s2_ratio = values["FAG_S2_CV_RATIO"]
        if min(s1_inner, s2_inner, s1_ratio, s2_ratio, self.limits.max_cores) <= 0:
            return self._derived_params(derived)
        s1_cv = s1_inner * s1_ratio
        s2_cv = s2_inner * s2_ratio
        s1_outer = self._ceil_div(values["FAG_S1"], s1_cv)
        s2_outer = self._ceil_div(values["FAG_S2"], s2_cv)
        dense_tasks = (
            values["FAG_B"]
            * values["FAG_N2"]
            * (values["FAG_N1"] // values["FAG_N2"])
            * s1_outer
            * s2_outer
        )
        valid_tasks = values.get("FAG_VALID_TASKS", dense_tasks)
        block_factor = self._ceil_div(valid_tasks, self.limits.max_cores)
        block_outer = self._ceil_div(valid_tasks, block_factor)
        derived.update(
            {
                "FAG_ROUTE": FAG_ROUTE_GENERIC,
                "FAG_S1_CV_INNER": s1_cv,
                "FAG_S2_CV_INNER": s2_cv,
                "FAG_S1_OUTER": s1_outer,
                "FAG_S2_OUTER": s2_outer,
                "FAG_S1_TAIL": values["FAG_S1"] % s1_inner or s1_inner,
                "FAG_S2_TAIL": values["FAG_S2"] % s2_inner or s2_inner,
                "FAG_BASE_MN": s1_inner * s2_inner,
                "FAG_TOTAL_TASKS": valid_tasks,
                "FAG_CORE_NUM": self.limits.max_cores,
                "FAG_BLOCK_FACTOR": block_factor,
                "FAG_BLOCK_OUTER": block_outer,
            }
        )
        return self._derived_params(derived)


class FlashAttentionScoreGradSameABValidator(AttentionValidator):
    """Validator for the priority-15500/1100 SameAB FAG kernels."""

    tile_names = ("FAG_S1_CV_INNER", "FAG_S2_CV_INNER")

    def __init__(self, limits: AttentionLimits) -> None:
        super().__init__(
            limits,
            "FAG",
            {
                "same_ab_route": (self._same_ab_route_is_valid, self._repair_route),
                "cv_tile": (self._cv_tile_is_valid, self._repair_tile),
                "local_memory": (self._local_memory_is_valid, self._repair_tile),
            },
        )

    def _same_ab_route_is_valid(self, params: dict[str, BaseParam]) -> bool:
        if not all(self._shape_is_valid(params)):
            return False
        deterministic = self._value(params, "FAG_DETERMINISTIC", 0) == 1
        if deterministic and self._value(params, "FAG_HAS_ROPE", 0) == 1:
            return False
        if self._value(params, "FAG_HAS_SINK", 0) == 1:
            return True
        if not deterministic and params["FAG_DTYPE_BYTES"].value == 4:
            return False
        if params["FAG_LAYOUT"].value == LAYOUT_TND and self._value(params, "FAG_MAX_WORKSPACE", 0) == 1:
            return False
        return True

    @staticmethod
    def _repair_route(params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return {}

    def _cv_tile_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if any(name not in params for name in self.tile_names):
            return (False,)
        s1_cv, s2_cv = (params[name].value for name in self.tile_names)
        return (
            s1_cv > 0 and s1_cv % 16 == 0,
            s2_cv > 0 and s2_cv % 16 == 0,
            s1_cv <= 1024,
            s2_cv <= 1024,
            s1_cv * s2_cv <= 512 * 512,
        )

    def _local_memory_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        # SameAB keeps Cube inner tiles fixed at 128x64; CV tiles only alter
        # the surrounding ownership/window.
        used_ub = 15 * 128 * 64 + 32 * 128 * self.limits.calc_type_size
        return (
            128 * 64 * self.limits.calc_type_size <= self.limits.L0C_size,
            used_ub + (8 + 33) * 1024 <= self.limits.UB_size,
        )

    def _all_tile_constraints(self, params: dict[str, BaseParam]) -> bool:
        return all(self._cv_tile_is_valid(params)) and all(self._local_memory_is_valid(params))

    def _repair_tile(self, params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return self._repair_names(params, self.tile_names, self._all_tile_constraints)

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = self._values(params)
        derived = self._common_derived(values)
        if not derived or any(name not in values for name in self.tile_names):
            return self._derived_params(derived)
        s1_cv = values["FAG_S1_CV_INNER"]
        s2_cv = values["FAG_S2_CV_INNER"]
        if min(s1_cv, s2_cv, self.limits.max_cores) <= 0:
            return self._derived_params(derived)
        s1_outer = self._ceil_div(values["FAG_S1"], s1_cv)
        s2_outer = self._ceil_div(values["FAG_S2"], s2_cv)
        tasks = (
            values["FAG_B"]
            * values["FAG_N2"]
            * (values["FAG_N1"] // values["FAG_N2"])
            * s1_outer
            * s2_outer
        )
        deterministic = values.get("FAG_DETERMINISTIC", 0) == 1
        derived.update(
            {
                "FAG_ROUTE": FAG_ROUTE_SAME_AB_DETERMINISTIC if deterministic else FAG_ROUTE_SAME_AB,
                "FAG_S1_INNER": 128,
                "FAG_S2_INNER": 64,
                "FAG_S1_OUTER": s1_outer,
                "FAG_S2_OUTER": s2_outer,
                "FAG_BASE_MN": 128 * 64,
                "FAG_TOTAL_TASKS": tasks,
                "FAG_CORE_NUM": self.limits.max_cores,
                "FAG_BLOCK_FACTOR": self._ceil_div(tasks, self.limits.max_cores),
                "FAG_BLOCK_OUTER": self.limits.max_cores,
            }
        )
        return self._derived_params(derived)
