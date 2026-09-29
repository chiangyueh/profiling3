from __future__ import annotations

from tiling.algs.ga.GaParam import GaParam
from tiling.validators.attention import (
    FlashAttentionScoreGeneralValidator,
    FlashAttentionScoreGradGenericValidator,
    FlashAttentionScoreGradMlaValidator,
    FlashAttentionScoreGradSameABValidator,
    FlashAttentionScoreVarLenValidator,
)


class _GaParamMixin:
    def _make_param(
        self,
        name: str,
        value: int,
        is_const: bool,
        domain: list[int] | None = None,
    ) -> GaParam:
        return GaParam(name=name, value=value, is_const=is_const, domain=domain or [value])


class GaFlashAttentionScoreGeneralValidator(_GaParamMixin, FlashAttentionScoreGeneralValidator):
    pass


class GaFlashAttentionScoreVarLenValidator(_GaParamMixin, FlashAttentionScoreVarLenValidator):
    pass


class GaFlashAttentionScoreGradMlaValidator(_GaParamMixin, FlashAttentionScoreGradMlaValidator):
    pass


class GaFlashAttentionScoreGradSameABValidator(_GaParamMixin, FlashAttentionScoreGradSameABValidator):
    pass


class GaFlashAttentionScoreGradGenericValidator(_GaParamMixin, FlashAttentionScoreGradGenericValidator):
    pass
