from __future__ import annotations

from tiling.algs.ga.GaParam import GaParam
from tiling.validators import AttentionValidator


class GaAttentionValidator(AttentionValidator):
    def _make_param(
        self,
        name: str,
        value: int,
        is_const: bool,
        domain: list[int] | None = None,
    ) -> GaParam:
        return GaParam(name=name, value=value, is_const=is_const, domain=domain or [value])
