from .AttentionValidator import (
    AttentionValidator,
    LAYOUT_BNSD,
    LAYOUT_BSND,
    LAYOUT_SBH,
    LAYOUT_TND,
)
from .FlashAttentionScoreValidator import (
    FA_ROUTE_GENERAL,
    FA_ROUTE_SAME_AB,
    FA_ROUTE_VARLEN,
    FlashAttentionScoreGeneralValidator,
    FlashAttentionScoreVarLenValidator,
)
from .FlashAttentionScoreGradValidator import (
    FAG_ROUTE_GENERIC,
    FAG_ROUTE_MLA,
    FAG_ROUTE_SAME_AB,
    FAG_ROUTE_SAME_AB_DETERMINISTIC,
    FlashAttentionScoreGradGenericValidator,
    FlashAttentionScoreGradMlaValidator,
    FlashAttentionScoreGradSameABValidator,
)


def create_validator(kernel: str, limits):
    """Create the validator matching the kernel observed in profiling."""

    validators = {
        "fa_general": FlashAttentionScoreGeneralValidator,
        "fa_varlen": FlashAttentionScoreVarLenValidator,
        "fag_mla": FlashAttentionScoreGradMlaValidator,
        "fag_same_ab": FlashAttentionScoreGradSameABValidator,
        "fag_generic": FlashAttentionScoreGradGenericValidator,
    }
    try:
        validator_type = validators[kernel]
    except KeyError as error:
        supported = ", ".join(sorted(validators))
        raise ValueError(f"unsupported attention kernel {kernel!r}; choose one of: {supported}") from error
    return validator_type(limits)


__all__ = [
    "AttentionValidator",
    "FlashAttentionScoreGeneralValidator",
    "FlashAttentionScoreVarLenValidator",
    "FlashAttentionScoreGradMlaValidator",
    "FlashAttentionScoreGradSameABValidator",
    "FlashAttentionScoreGradGenericValidator",
    "create_validator",
    "LAYOUT_BNSD",
    "LAYOUT_BSND",
    "LAYOUT_SBH",
    "LAYOUT_TND",
]
