# NEW BEGIN
from __future__ import annotations

from collections.abc import Callable

from tiling.base import BaseParam, BaseValidator
from tiling.limits import AttentionLimits


LAYOUT_BNSD = 0
LAYOUT_SBH = 1
LAYOUT_BSND = 2
SUPPORTED_LAYOUTS = (LAYOUT_BNSD, LAYOUT_SBH, LAYOUT_BSND)


class AttentionValidator(BaseValidator):
    def __init__(
        self,
        limits: AttentionLimits,
        param_funcs: dict[
            str,
            tuple[
                Callable[[dict[str, BaseParam]], bool | tuple[bool, ...]],
                Callable[[dict[str, BaseParam]], dict[str, BaseParam]],
            ],
        ],
    ) -> None:
        funcs = {"shape": (self._shape_is_valid, self._repair_shape)}
        funcs.update(param_funcs)
        super().__init__(limits, funcs)
        self.limits: AttentionLimits

    @property
    def shape_names(self) -> tuple[str, ...]:
        return (
            "FA_B",
            "FA_N1",
            "FA_N2",
            "FA_S1",
            "FA_S2",
            "FA_D",
            "FA_DV",
            "FA_DTYPE_BYTES",
            "FA_LAYOUT",
        )

    def _shape_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if any(name not in params for name in self.shape_names):
            return (False,)
        return (
            all(params[name].value > 0 for name in self.shape_names[:-1]),
            params["FA_N2"].value > 0
            and params["FA_N1"].value % params["FA_N2"].value == 0,
            params["FA_DTYPE_BYTES"].value in (2, 4),
            params["FA_LAYOUT"].value in SUPPORTED_LAYOUTS,
        )

    @staticmethod
    def _repair_shape(params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        return {}

    @staticmethod
    def _values(params: list[BaseParam]) -> dict[str, int]:
        return {param.name: param.value for param in params}

    def _common_derived(self, values: dict[str, int]) -> dict[str, int]:
        if any(name not in values for name in self.shape_names):
            return {}
        if any(values[name] <= 0 for name in self.shape_names[:-1]) or values["FA_N2"] <= 0:
            return {}
        return {
            "FA_G": values["FA_N1"] // values["FA_N2"],
            "FA_S1_ALIGN": self._align_up(values["FA_S1"], 16),
            "FA_S2_ALIGN": self._align_up(values["FA_S2"], 16),
            "FA_D_ALIGN": self._align_up(values["FA_D"], 16),
            "FA_DV_ALIGN": self._align_up(values["FA_DV"], 16),
        }

    def _derived_params(self, derived: dict[str, int]) -> list[BaseParam]:
        return [self._make_param(name, value, True) for name, value in derived.items()]

    def _repair_names(
        self,
        params: dict[str, BaseParam],
        names: tuple[str, ...],
        predicate: Callable[[dict[str, BaseParam]], bool],
    ) -> dict[str, BaseParam]:
        movable = self._movable_params(params, list(names))
        if not movable:
            return {name: params[name] for name in names if name in params}
        repaired = self.repair_dijkstra(movable, predicate, context=params)
        return {name: repaired.get(name, params[name]) for name in names if name in params}
# NEW END
