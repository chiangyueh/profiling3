from __future__ import annotations

from typing import Callable
from .GaParam import GaParam
from tiling.base.Base import BaseResult
import random
import copy


class Individual:
    def __init__(self, params: list[GaParam]) -> None:
        self.duration = None
        self.params = params

    def mutation(self, rate: float = 0.1) -> None:
        for p in self.params:
            if p.is_const:
                continue
            bits = list(p.encoded_index)
            for i in range(len(bits)):
                if random.random() < rate:
                    bits[i] = '1' if bits[i] == '0' else '0'
            p.update("".join(bits))

    def _update_params(self, repaired_params: list) -> None:
        by_name = {p.name: p for p in repaired_params}
        for gene in self.params:
            src = by_name.get(gene.name)
            if src is None or gene.is_const:
                continue
            gene.update(int(src.index))

    def compute_duration(self, duration: Callable[..., BaseResult]) -> float:
        result = duration(self.decoded_params)
        self._update_params(result.params)
        self.duration = result.duration
        return self.duration

    @property
    def decoded_params(self) -> list[GaParam]:
        return [copy.copy(p) for p in self.params]


class Population(list):
    def __init__(self, *args):
        super().__init__(*args)

    def crossing(self, parents: tuple) -> Individual:
        p1, p2 = parents
        child_params = []
        for g1, g2 in zip(p1.params, p2.params):
            child = copy.copy(g1)
            if not g1.is_const:
                e1, e2 = g1.encoded_index, g2.encoded_index
                if len(e1) > 1:
                    point = random.randint(1, len(e1) - 1)
                    child.update(e1[:point] + e2[point:])
                else:
                    child.update(random.choice((e1, e2)))
            child_params.append(child)
        return Individual(child_params)

    def tournament(self, k: int = 3) -> Individual:
        contenders = random.sample(list(self), k)
        return min(contenders, key=lambda ind: ind.duration)
