from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AscendLimits:
    max_cores: int
    L0A_size: int
    L0B_size: int
    L0C_size: int
    L1_size: int
    L2_size: int
    UB_size: int


@dataclass
class OpLimits(AscendLimits):
    domains: dict[str, list[int]]


@dataclass
class AttentionLimits(OpLimits):
    """DAV_2201 resources required by FA/FAG route validators.

    ``max_cores`` is the AIV count; ``aic_num`` is separate because SameAB
    kernels schedule paired AIV/AIC work from the AIC count.
    """

    aic_num: int
    calc_type_size: int = 4
    
