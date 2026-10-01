from .AttentionValidator import (
    AttentionValidator,
    LAYOUT_BNSD,
    LAYOUT_BSND,
    LAYOUT_SBH,
    LAYOUT_TND,
)
from .FlashAttentionScoreValidator import (
    FA_ROUTE_B,
    FA_ROUTE_DROP_ADAPTER,
    FA_ROUTE_GENERAL,
    FA_ROUTE_SAME_AB,
    FA_ROUTE_S1,
    FA_ROUTE_VARLEN,
    FlashAttentionScoreBValidator,
    FlashAttentionScoreDropAdapterValidator,
    FlashAttentionScoreGeneralValidator,
    FlashAttentionScoreS1Validator,
    FlashAttentionScoreSameABValidator,
    FlashAttentionScoreVarLenValidator,
)
from .FlashAttentionScoreGradValidator import (
    FAG_ROUTE_B,
    FAG_ROUTE_BASIC_DETERMINISTIC,
    FAG_ROUTE_BN2,
    FAG_ROUTE_DETERMINISTIC_BN2,
    FAG_ROUTE_GENERIC,
    FAG_ROUTE_MLA,
    FAG_ROUTE_N2,
    FAG_ROUTE_SAME_AB,
    FAG_ROUTE_SAME_AB_DETERMINISTIC,
    FAG_ROUTE_UNPADDED,
    FlashAttentionScoreGradBValidator,
    FlashAttentionScoreGradBasicDeterministicValidator,
    FlashAttentionScoreGradBn2Validator,
    FlashAttentionScoreGradDeterministicBn2Validator,
    FlashAttentionScoreGradGenericValidator,
    FlashAttentionScoreGradMlaValidator,
    FlashAttentionScoreGradN2Validator,
    FlashAttentionScoreGradSameABDeterministicValidator,
    FlashAttentionScoreGradSameABValidator,
    FlashAttentionScoreGradUnpaddedValidator,
)


def create_validator(kernel: str, limits):
    """Create the validator matching the kernel observed in profiling."""

    validators = {
        "fa_drop_adapter": FlashAttentionScoreDropAdapterValidator,
        "fa_b": FlashAttentionScoreBValidator,
        "fa_general": FlashAttentionScoreGeneralValidator,
        "fa_s1s2": FlashAttentionScoreGeneralValidator,
        "fa_s1": FlashAttentionScoreS1Validator,
        "fa_same_ab": FlashAttentionScoreSameABValidator,
        "fa_varlen": FlashAttentionScoreVarLenValidator,
        "fag_deterministic_bn2": FlashAttentionScoreGradDeterministicBn2Validator,
        "fag_mla": FlashAttentionScoreGradMlaValidator,
        "fag_basic_deterministic": FlashAttentionScoreGradBasicDeterministicValidator,
        "fag_same_ab_deterministic": FlashAttentionScoreGradSameABDeterministicValidator,
        "fag_unpadded": FlashAttentionScoreGradUnpaddedValidator,
        "fag_b": FlashAttentionScoreGradBValidator,
        "fag_n2": FlashAttentionScoreGradN2Validator,
        "fag_bn2": FlashAttentionScoreGradBn2Validator,
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
    "FlashAttentionScoreSameABValidator",
    "FlashAttentionScoreS1Validator",
    "FlashAttentionScoreBValidator",
    "FlashAttentionScoreDropAdapterValidator",
    "FlashAttentionScoreVarLenValidator",
    "FlashAttentionScoreGradDeterministicBn2Validator",
    "FlashAttentionScoreGradMlaValidator",
    "FlashAttentionScoreGradBasicDeterministicValidator",
    "FlashAttentionScoreGradSameABDeterministicValidator",
    "FlashAttentionScoreGradUnpaddedValidator",
    "FlashAttentionScoreGradBValidator",
    "FlashAttentionScoreGradN2Validator",
    "FlashAttentionScoreGradBn2Validator",
    "FlashAttentionScoreGradSameABValidator",
    "FlashAttentionScoreGradGenericValidator",
    "create_validator",
    "LAYOUT_BNSD",
    "LAYOUT_BSND",
    "LAYOUT_SBH",
    "LAYOUT_TND",
]
