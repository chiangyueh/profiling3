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
class MatmulLimits(OpLimits):
    dtype_size: int


@dataclass
class AttentionLimits(OpLimits):
    """DAV_2201 resources used by FA/FAG route validators."""

    aic_num: int
    calc_type_size: int = 4
