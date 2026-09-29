from __future__ import annotations

from dataclasses import dataclass

@dataclass
class OpLimits:
    domains: dict[str, list[int]]

@dataclass
class AscendLimits(OpLimits):
    max_cores: int
    L0A_size: int
    L0B_size: int
    L0C_size: int
    L1_size: int
    L2_size: int
    
@dataclass
class MatmulLimits(AscendLimits):
    dtype_size: int


@dataclass
class AttentionLimits(OpLimits):
    """Small operator-specific envelope shared by FA and FAG searches."""

    operator: str
    max_cores: int
    
