from __future__ import annotations

from tiling.validators import MatmulBaseKernelValidator
from .GaParam import GaParam


class GaMatmulBaseKernelValidator(MatmulBaseKernelValidator):
    def _make_param(self, name: str, value: int, is_const: bool, domain: list[int] | None = None) -> GaParam:
        return GaParam(name=name, value=value, is_const=is_const, domain=domain or [value])
        
