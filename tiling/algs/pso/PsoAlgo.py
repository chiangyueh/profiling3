from __future__ import annotations

from tiling.base import BaseResult, BaseValidator, BaseAlgo, BaseParam
from .Swarm import Swarm, Particle
from typing import Callable


class PsoAlgo(BaseAlgo):
    def __init__(self,
                 is_stop: Callable[[list[BaseResult]], bool],
                 validator: BaseValidator,
                 input_params: list[BaseParam],
                 swarm_size: int = 8,
                 chi: float = 0.7298,
                 c1: float = 2.05,
                 c2: float = 2.05,
                 v_max_frac: float = 0.5,
                 runner: str = "./run.sh",
                 cache_path: str = "msprof_cache.json",
                 verbose: bool = False) -> None:
        super().__init__(is_stop, validator, input_params, runner=runner, cache_path=cache_path, verbose=verbose)
        self.swarm_size = swarm_size
        self.chi = chi
        self.c1 = c1
        self.c2 = c2
        self.v_max_frac = v_max_frac

        self.swarm = self._init_swarm(input_params)
        self._step()
        self.gbest_x, self.gbest_dur = self.swarm.global_best()
        print("SWARM IS CREATED")

    def _init_swarm(self, input_params: list) -> Swarm:
        combs = self.validator.get_combinations(self.swarm_size, input_params)
        particles = [Particle(params, v_max_frac=self.v_max_frac) for params in combs]
        return Swarm(particles)

    def run(self, *args, **kwargs) -> BaseResult:
        for i, particle in enumerate(self.swarm):
            nbest_x = self.swarm.neighbour_best_x(i)     
            particle.update_velocity(nbest_x, self.chi, self.c1, self.c2)
            particle.update_position()

        self._step()

        gx, gd = self.swarm.global_best()
        if gd < self.gbest_dur:
            self.gbest_x, self.gbest_dur = gx, gd

        best_particle = min(self.swarm, key=lambda p: p.pbest_dur)
        saved = list(best_particle.x)
        best_particle.x = list(best_particle.pbest_x)
        best_params = best_particle.as_params()
        best_particle.x = saved

        return BaseResult(float(self.gbest_dur), best_params)
    
    
    def _step(self) -> None:
        for p in self.swarm:
            result = self._duration(p.as_params())
            p.set_params(result.params)
            p.remember(result.duration)
