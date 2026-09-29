from __future__ import annotations

from collections.abc import Callable

from tiling.base import BaseParam, BaseValidator
from tiling.limits import AttentionLimits


LAYOUT_BNSD = 0
LAYOUT_SBH = 1
LAYOUT_BSND = 2
LAYOUT_TND = 3
SUPPORTED_LAYOUTS = (LAYOUT_BNSD, LAYOUT_SBH, LAYOUT_BSND, LAYOUT_TND)


class AttentionValidator(BaseValidator):
    """Common shape validation shared by route-specific FA/FAG validators."""

    def __init__(
        self,
        limits: AttentionLimits,
        prefix: str,
        param_funcs: dict[
            str,
            tuple[
                Callable[[dict[str, BaseParam]], bool | tuple[bool, ...]],
                Callable[[dict[str, BaseParam]], dict[str, BaseParam]],
            ],
        ]
        | None = None,
    ) -> None:
        funcs = {
            "shape": (self._shape_is_valid, self._repair_shape),
        }
        funcs.update(param_funcs or {})
        super().__init__(limits, funcs)
        self.limits: AttentionLimits
        self.prefix = prefix

    @property
    def shape_names(self) -> tuple[str, ...]:
        p = self.prefix
        return (
            f"{p}_B",
            f"{p}_N1",
            f"{p}_N2",
            f"{p}_S1",
            f"{p}_S2",
            f"{p}_D",
            f"{p}_DV",
            f"{p}_DTYPE_BYTES",
            f"{p}_LAYOUT",
        )

    def _shape_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool, ...]:
        if any(name not in params for name in self.shape_names):
            return (False,)
        p = self.prefix
        values = [params[name].value for name in self.shape_names]
        n1 = params[f"{p}_N1"].value
        n2 = params[f"{p}_N2"].value
        return (
            all(value > 0 for value in values[:-1]),
            n2 > 0 and n1 % n2 == 0,
            params[f"{p}_DTYPE_BYTES"].value in (2, 4),
            params[f"{p}_LAYOUT"].value in SUPPORTED_LAYOUTS,
        )

    @staticmethod
    def _repair_shape(params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        # Workload shape and features are constants. A validator must reject an
        # unsupported operator invocation, never silently mutate it.
        return {}

    @staticmethod
    def _values(params: list[BaseParam]) -> dict[str, int]:
        return {param.name: param.value for param in params}

    @staticmethod
    def _int(values: dict[str, int], name: str, default: int = 0) -> int:
        return values.get(name, default)

    def _common_derived(self, values: dict[str, int]) -> dict[str, int]:
        p = self.prefix
        required = self.shape_names
        if any(name not in values for name in required):
            return {}
        if any(values[name] <= 0 for name in required[:-1]) or values[f"{p}_N2"] <= 0:
            return {}
        return {
            f"{p}_G": values[f"{p}_N1"] // values[f"{p}_N2"],
            f"{p}_S1_ALIGN": self._align_up(values[f"{p}_S1"], 16),
            f"{p}_S2_ALIGN": self._align_up(values[f"{p}_S2"], 16),
            f"{p}_D_ALIGN": self._align_up(values[f"{p}_D"], 16),
            f"{p}_DV_ALIGN": self._align_up(values[f"{p}_DV"], 16),
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

        repaired = self.repair_dijkstra(
            movable,
            predicate,
            context=params,
        )
        return {name: repaired.get(name, params[name]) for name in names if name in params}
