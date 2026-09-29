from __future__ import annotations

from tiling.base import BaseParam, BaseValidator
from tiling.limits import AttentionLimits


FORWARD = "flash_attention_score"
BACKWARD = "flash_attention_score_grad"
SUPPORTED_OPERATORS = (FORWARD, BACKWARD)


class AttentionValidator(BaseValidator):
    """Validate the thin environment contract used by FA/FAG runners.

    This deliberately checks only invariants that are common to all official
    tiling routes.  Route-specific legality remains with the official host
    tiler (or the runner that applies a candidate).
    """

    def __init__(self, limits: AttentionLimits) -> None:
        if limits.operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"unsupported attention operator: {limits.operator}")
        super().__init__(
            limits,
            {"attention": (self._attention_is_valid, self._repair_attention)},
        )
        self.limits: AttentionLimits

    @property
    def _prefix(self) -> str:
        return "FA" if self.limits.operator == FORWARD else "FAG"

    def _make_param(
        self,
        name: str,
        value: int,
        is_const: bool,
        domain: list[int] | None = None,
    ) -> BaseParam:
        return BaseParam(name=name, value=value, is_const=is_const, domain=domain or [value])

    def _required_names(self) -> tuple[str, ...]:
        prefix = self._prefix
        shape = (
            f"{prefix}_B",
            f"{prefix}_N1",
            f"{prefix}_N2",
            f"{prefix}_S1",
            f"{prefix}_S2",
            f"{prefix}_D",
            f"{prefix}_DV",
        )
        if self.limits.operator == FORWARD:
            tunables = ("FA_S1_BASE", "FA_S2_BASE", "FA_D_BASE", "FA_CORE_NUM")
        else:
            tunables = (
                "FAG_S1_INNER",
                "FAG_S2_INNER",
                "FAG_S1_CV_RATIO",
                "FAG_S2_CV_RATIO",
                "FAG_CORE_NUM",
            )
        return shape + tunables

    def _attention_is_valid(self, params: dict[str, BaseParam]) -> bool:
        required = self._required_names()
        if any(name not in params for name in required):
            return False
        if any(params[name].value <= 0 for name in required):
            return False

        prefix = self._prefix
        q_heads = params[f"{prefix}_N1"].value
        kv_heads = params[f"{prefix}_N2"].value
        if q_heads % kv_heads != 0:
            return False

        s1 = params[f"{prefix}_S1"].value
        s2 = params[f"{prefix}_S2"].value
        d = max(params[f"{prefix}_D"].value, params[f"{prefix}_DV"].value)
        if self.limits.operator == FORWARD:
            within_shape = (
                params["FA_S1_BASE"].value <= s1
                and params["FA_S2_BASE"].value <= s2
                and params["FA_D_BASE"].value <= d
            )
            core_num = params["FA_CORE_NUM"].value
        else:
            within_shape = (
                params["FAG_S1_INNER"].value <= s1
                and params["FAG_S2_INNER"].value <= s2
            )
            core_num = params["FAG_CORE_NUM"].value
        return within_shape and core_num <= self.limits.max_cores

    @staticmethod
    def _move_to_largest_not_above(param: BaseParam, upper: int) -> None:
        if param.is_const or param.value <= upper:
            return
        eligible = [(index, value) for index, value in enumerate(param.domain) if 0 < value <= upper]
        if eligible:
            param.update(max(eligible, key=lambda item: item[1])[0])

    def _repair_attention(self, params: dict[str, BaseParam]) -> dict[str, BaseParam]:
        prefix = self._prefix
        s1 = params[f"{prefix}_S1"].value
        s2 = params[f"{prefix}_S2"].value
        d = max(params[f"{prefix}_D"].value, params[f"{prefix}_DV"].value)
        bounds = (
            {
                "FA_S1_BASE": s1,
                "FA_S2_BASE": s2,
                "FA_D_BASE": d,
                "FA_CORE_NUM": self.limits.max_cores,
            }
            if self.limits.operator == FORWARD
            else {
                "FAG_S1_INNER": s1,
                "FAG_S2_INNER": s2,
                "FAG_CORE_NUM": self.limits.max_cores,
            }
        )
        for name, upper in bounds.items():
            if name in params:
                self._move_to_largest_not_above(params[name], upper)
        return {name: params[name] for name in bounds if name in params}

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        values = {param.name: param.value for param in params}
        prefix = self._prefix
        required_shape = [
            f"{prefix}_N1",
            f"{prefix}_N2",
            f"{prefix}_S1",
            f"{prefix}_S2",
        ]
        if any(name not in values for name in required_shape):
            return []

        if self.limits.operator == FORWARD:
            required_tiles = ["FA_S1_BASE", "FA_S2_BASE"]
            if any(name not in values for name in required_tiles):
                return []
            derived = {
                "FA_G": values["FA_N1"] // values["FA_N2"],
                "FA_S1_OUTER": self._ceil_div(values["FA_S1"], values["FA_S1_BASE"]),
                "FA_S2_OUTER": self._ceil_div(values["FA_S2"], values["FA_S2_BASE"]),
            }
        else:
            required_tiles = ["FAG_S1_INNER", "FAG_S2_INNER"]
            if any(name not in values for name in required_tiles):
                return []
            derived = {
                "FAG_G": values["FAG_N1"] // values["FAG_N2"],
                "FAG_S1_OUTER": self._ceil_div(values["FAG_S1"], values["FAG_S1_INNER"]),
                "FAG_S2_OUTER": self._ceil_div(values["FAG_S2"], values["FAG_S2_INNER"]),
            }
        return [self._make_param(name, value, True) for name, value in derived.items()]
