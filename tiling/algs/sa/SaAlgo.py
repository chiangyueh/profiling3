# NEW BEGIN
from tiling.base.Base import BaseAlgo, BaseResult, BaseValidator, BaseParam
from typing import Callable
import random
import math
import copy


class SaAlgo(BaseAlgo):
    def __init__(self,
                 is_stop: Callable[[list[BaseResult]], bool],
                 validator: BaseValidator,
                 input_params: list[BaseParam],
                 t_start: float = 10.0,
                 t_min: float = 1e-2,
                 cooling: float = 0.95,
                 runner: str = "./run.sh",
                 cache_path: str = "msprof_cache.json",
                 verbose: bool = False) -> None:
        super().__init__(is_stop, validator, input_params, runner=runner, cache_path=cache_path, verbose=verbose)
        self.t_start = t_start
        self.t_min = t_min
        self.cooling = cooling
        self.temperature = t_start

        init = self._init_state(input_params)
        result = self._duration(init)
        self.current = result.params
        self.current_dur = result.duration
        self.best = [copy.copy(p) for p in self.current]
        self.best_dur = self.current_dur
        print("INITIAL STATE IS CREATED")

    def _init_state(self, input_params: list) -> list[BaseParam]:
        return self.validator.get_combinations(1, input_params)[0]

    def _neighbour(self, params: list[BaseParam]) -> list[BaseParam]:
        neighbour = [copy.copy(p) for p in params]
        movable = [p for p in neighbour if not p.is_const and len(p.domain) > 1]
        if not movable:
            return neighbour
        gene = random.choice(movable)
        step = random.choice((-1, 1))
        new_index = gene.index + step
        new_index = max(0, min(new_index, len(gene.domain) - 1))
        gene.update(new_index)
        return neighbour

    def _accept(self, delta: float) -> bool:
        if delta < 0:
            return True
        if self.temperature <= 0:
            return False
        return random.random() < math.exp(-delta / self.temperature)

    def run(self, *args, **kwargs) -> BaseResult:
        candidate = self._neighbour(self.current)
        result = self._duration(candidate)
        cand_params = result.params
        cand_dur = result.duration

        delta = cand_dur - self.current_dur
        if self._accept(delta):
            self.current = cand_params
            self.current_dur = cand_dur

        if self.current_dur < self.best_dur:
            self.best = [copy.copy(p) for p in self.current]
            self.best_dur = self.current_dur

        self.temperature = max(self.t_min, self.temperature * self.cooling)

        return BaseResult(float(self.best_dur), [copy.copy(p) for p in self.best])
# NEW END
