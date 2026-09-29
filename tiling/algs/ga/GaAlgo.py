from __future__ import annotations

from tiling.base.Base import BaseAlgo, BaseResult, BaseValidator
from .Population import Individual, Population
from typing import Callable


class GaAlgo(BaseAlgo):
    def __init__(self,
                 is_stop: Callable[[list[BaseResult]], bool],
                 validator: BaseValidator,
                 input_params: list,
                 pop_size: int = 8,
                 mut_rate: float = 0.1,
                 tournament_k: int = 3,
                 runner: str = "./run.sh",
                 cache_path: str = "msprof_cache.json",
                 verbose: bool = False) -> None:
        super().__init__(is_stop, validator, runner=runner, cache_path=cache_path, verbose=verbose)
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.tournament_k = tournament_k
        self.population = self._init_population(input_params)
 
    def _init_population(self, input_params: list) -> Population:
        pop = Population()
        combs = self.validator.get_combinations(self.pop_size, input_params)
        for params in combs:
            pop.append(Individual(params=params))
        print("POPULATION IS CREATED")
        return pop
    
    def run(self, *args, **kwargs) -> BaseResult:
        for ind in self.population:
            if ind.duration is None:
                ind.compute_duration(self._duration)
 
        elite: Individual = min(self.population, key=lambda i: i.duration)
        new_pop = Population()
        new_pop.append(elite)
 
        while len(new_pop) < self.pop_size:
            p1 = self.population.tournament(self.tournament_k)
            p2 = self.population.tournament(self.tournament_k)
            child = self.population.crossing((p1, p2))
            child.mutation(self.mut_rate)
            child.compute_duration(self._duration)
            new_pop.append(child)
 
        self.population = new_pop
 
        best: Individual = min(new_pop, key=lambda i: i.duration)
        return BaseResult(float(best.duration), best.decoded_params)
