from __future__ import annotations

from tiling.base.Base import BaseParam
import random
import copy


class Particle:
    def __init__(self, params: list[BaseParam],
                 v_max_frac: float = 0.5) -> None:
        self.set_params(params)

        self.v_max = [v_max_frac * (len(self._template[i].domain) - 1)
                      for i in self._movable_idx]
        self.v = [random.uniform(-vm, vm) for vm in self.v_max]

        self.pbest_x = list(self.x)
        self.pbest_dur = float("inf")

    def _clamp_index(self, value: float, gene_i: int) -> int:
        dom_len = len(self._template[self._movable_idx[gene_i]].domain)
        idx = int(round(value))
        return max(0, min(idx, dom_len - 1))

    def as_params(self) -> list[BaseParam]:
        out = [copy.copy(p) for p in self._template]
        for gi, ti in enumerate(self._movable_idx):
            idx = self._clamp_index(self.x[gi], gi)
            out[ti].update(idx)
        return out
    
    def set_params(self, params: list[BaseParam]) -> None:
        self._template = [copy.copy(p) for p in params]
        self._movable_idx = [i for i, p in enumerate(self._template)
                             if not p.is_const and len(p.domain) > 1]
        self.x = [float(self._template[i].index) for i in self._movable_idx]

    def update_velocity(self, nbest_x: list[float],
                        chi: float, c1: float, c2: float) -> None:
        for i in range(len(self.x)):
            cognitive = c1 * random.random() * (self.pbest_x[i] - self.x[i])
            social = c2 * random.random() * (nbest_x[i] - self.x[i])
            vi = chi * (self.v[i] + cognitive + social)
            vi = max(-self.v_max[i], min(vi, self.v_max[i]))
            self.v[i] = vi

    def update_position(self) -> None:
        for i in range(len(self.x)):
            self.x[i] += self.v[i]
            hi = len(self._template[self._movable_idx[i]].domain) - 1
            self.x[i] = max(0.0, min(self.x[i], float(hi)))

    def remember(self, dur: float) -> None:
        if dur < self.pbest_dur:
            self.pbest_dur = dur
            self.pbest_x = list(self.x)


class Swarm(list):
    def __init__(self, particles: list[Particle]):
        super().__init__(particles)

    def neighbour_best_x(self, i: int) -> list[float]:
        n = len(self)
        left = (i - 1) % n
        right = (i + 1) % n
        candidates = [self[left], self[i], self[right]]
        best = min(candidates, key=lambda p: p.pbest_dur)
        return list(best.pbest_x)

    def global_best(self) -> tuple[list[float], float]:
        best = min(self, key=lambda p: p.pbest_dur)
        return list(best.pbest_x), best.pbest_dur
