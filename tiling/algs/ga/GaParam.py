from __future__ import annotations

from tiling.base.Base import BaseParam
from dataclasses import field, dataclass
import math

@dataclass
class GaParam(BaseParam):

    domain: list = field(default_factory=list)
    index: int = 0

    def __post_init__(self):
        if self.is_const and not self.domain:
            self.domain = [self.value]
        self.index = self.domain.index(self.value)

    def _bits(self) -> int:
        return max(1, math.ceil(math.log2(len(self.domain)))) if len(self.domain) > 1 else 1

    def _encode(self, index: int) -> str:
        gray = index ^ (index >> 1)
        return f"{gray:0{self._bits()}b}"

    def _decode(self, gray: str) -> int:
        gray_int = int(gray, 2)
        decimal = 0
        while gray_int > 0:
            decimal ^= gray_int
            gray_int >>= 1
        return decimal

    @property
    def encoded_index(self) -> str:
        return self._encode(self.index)

    def update(self, index: int | str) -> None:
        if isinstance(index, str):
            index = self._decode(index)
        index = min(index, len(self.domain) - 1) 
        super().update(index)
