from __future__ import annotations

from typing import Callable
from dataclasses import dataclass, field
from tiling.limits import OpLimits
from pathlib import Path
import json
import numpy as np
import copy
import heapq
import random


@dataclass
class BaseParam:
    name: str
    value: int
    is_const: bool
    domain: list = field(default_factory=list)
    index: int = 0

    def __post_init__(self):
        if self.is_const and not self.domain:
            self.domain = [self.value]
        self.index = self.domain.index(self.value)

    def __repr__(self) -> str:
        return f"name={self.name}, value={self.value}, is_const={self.is_const}"
    
    def update(self, index: int) -> None:
        self.index = index
        self.value = self.domain[index]


@dataclass
class BaseResult:
    duration: float
    params: list[BaseParam]

    def __repr__(self) -> str:
        return f"duration={self.duration}\nparams={[f'{param.name}={param.value}' for param in self.params]}\n"


class BaseValidator:
    def __init__(self,
                 limits: OpLimits,
                 param_funcs: dict[str, list[tuple[Callable[[dict[str, BaseParam]], bool | tuple[bool]],
                                                   Callable[[dict[str, BaseParam]], dict[BaseParam]]]]]):
        self.limits = limits
        self.param_funcs = param_funcs
    
    def get_combinations(self, num: int, const_params: list) -> list[list[BaseParam]]:
        combs = []
        while len(combs) < num:
            params = [self._make_param(p.name, p.value, True) for p in const_params]
            for name, domain in self.limits.domains.items():
                params.append(self._make_param(name, random.choice(domain), False, domain))
            params = self.get_all_params(params)
            if self.is_valid(params):
                combs.append(params)
        return combs
    
    
    
    def _make_param(self, name: str, value: int, is_const: bool, domain: list[int] | None = None) -> BaseParam:
        raise NotImplementedError
    
    def _ceil_div(self, a: int, b: int) -> int:
        return (a + b - 1) // b

    def _value(self, params: dict[str, BaseParam], name: str, default: int | None = None) -> int:
        param = params.get(name)
        if param is None:
            if default is None:
                raise KeyError(f"missing param: {name}")
            return default
        return param.value
    
    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        raise NotImplementedError

    def get_all_params(self, params: list[BaseParam]) -> list[BaseParam]:
        derived = self.get_derived_params(params)
        derived_names = {p.name for p in derived}
        return [p for p in params if p.name not in derived_names] + derived
    
    def is_valid(self, params: list[BaseParam]) -> bool:
        ps_dict = {p.name: p for p in params}
        for limit_name in self.param_funcs:
            valid = self._is_valid(ps_dict, limit_name)
            if isinstance(valid, tuple):
                valid = all(valid)
            if not valid: 
                return False       
        return True
        
    def _is_valid(self,
                  params: dict[str, BaseParam],
                  limit_name: str) -> bool | tuple[bool]:
        valid_func, _ = self.param_funcs[limit_name]
        return valid_func(params)

    def repair(self, params: list[BaseParam]) -> list[BaseParam]:
        ps_dict = {p.name: p for p in params}
        for limit_name in self.param_funcs:
            valid = self._is_valid(ps_dict, limit_name)
            if isinstance(valid, tuple):
                valid = all(valid)
            if not valid:
                _, repair_func = self.param_funcs[limit_name]
                repaired_params = repair_func(ps_dict)
                ps_dict.update(repaired_params)
                # производные зависят от свободных: пересчитываем сразу,
                # иначе следующий предикат в цикле увидит их устаревшими
                ps_dict = {p.name: p for p in self.get_all_params(list(ps_dict.values()))}
        return self.get_all_params(list(ps_dict.values()))

    def repair_dijkstra(
        self,
        params: dict[str, BaseParam],
        is_valid: Callable[[dict[str, int]], bool],
        step_weight: Callable[[str, list[int]], float] | None = None,
    ) -> dict[BaseParam]:

        if step_weight is None:
            step_weight = lambda name, domain: 1.0 / len(domain)

        names = list(params.keys())
        domains = {n: params[n].domain for n in names}
        start = tuple(params[n].index for n in names)
        weights = {n: step_weight(n, domains[n]) for n in names}

        def values_of(state: tuple[int, int, tuple[int]]) -> dict[str, BaseParam]:
            out = {}
            for i, n in enumerate(names):
                p = copy.copy(params[n])
                p.update(state[i])
                out[n] = p
            return out

        def in_bounds(axis: int, idx: int) -> bool:
            return 0 <= idx < len(domains[names[axis]])
        
        counter = 0
        pq = [(0.0, counter, start)]
        best_cost = {start: 0.0}

        while pq:
            cost, _, state = heapq.heappop(pq)
            if cost > best_cost.get(state, float("inf")):
                continue
            
            if is_valid(values_of(state)):
                out = {}
                for i, n in enumerate(names):
                    p = copy.copy(params[n])
                    p.update(state[i])
                    out[n] = p
                return out
            
            for axis in range(len(names)):
                for delta in (-1, 1):
                    nidx = state[axis] + delta
                    if not in_bounds(axis, nidx):
                        continue
                    nstate = state[:axis] + (nidx,) + state[axis + 1:]
                    ncost = cost + weights[names[axis]]
                    if ncost < best_cost.get(nstate, float("inf")):
                        best_cost[nstate] = ncost
                        counter += 1
                        heapq.heappush(pq, (ncost, counter, nstate))
                        
        return dict(params)
    
    
    def _movable_params(self, params: dict[str, BaseParam], names: list[str]) -> dict[str, BaseParam]:
        return {n: params[n] for n in names if not params[n].is_const}

    def _const_params(self, params: dict[str, BaseParam], names: list[str]) -> dict[str, BaseParam]:
        return {n: params[n] for n in names if params[n].is_const}

class BaseAlgo:
    def __init__(self,
                 is_stop: Callable[[list[BaseResult]], bool],
                 validator: BaseValidator,
                 runner: str = "./run.sh",
                 verbose: bool = False,
                 cache_path: str = "msprof_cache.json") -> None:
        self.is_stop = is_stop
        self.validator = validator
        self.runner = runner
        self.verbose = verbose
        self.cache_path = Path(cache_path)
        self._cache = self._load_cache()

    def __call__(self, *args, **kwargs) -> list[BaseResult]:
        k = 1
        results = [self.run(*args, **kwargs)]
        if self.verbose:
            print(f"STEP={k}, RESULTS={results}")
        while not self.is_stop(results):
            results.append(self.run(*args, **kwargs))
            k += 1
            if self.verbose:
                print(f"STEP={k}, RESULTS={results}")
        return results

    def run(self, *args, **kwargs) -> BaseResult:
        raise NotImplementedError
    
    def _run_estimator(self, params: list[BaseParam]) -> float:
        raise NotImplementedError
    
    def _key(self, params: list[BaseParam]) -> str:
        return ";".join(f"{n}={v}" for n, v in sorted((p.name, p.value) for p in params))

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    raw = json.load(f)
                return {k: (float("inf") if v == "inf" else float(v)) for k, v in raw.items()}
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        merged = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    merged = json.load(f)
            except Exception:
                merged = {}
        for k, v in self._cache.items():
            merged[k] = "inf" if v == float("inf") else v
        tmp = self.cache_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=4)
        tmp.replace(self.cache_path) 

    def _duration(self, params: list[BaseParam]) -> BaseResult:
        params = self.validator.get_all_params(params)
        if not self.validator.is_valid(params):
            if self.verbose:
                print("PARAMS NEED TO REPAIR")
            params = self.validator.repair(params) 
            
        key = self._key(params)
        if key in self._cache:
            if self.verbose:
                print(f"CACHE HIT: {BaseResult(self._cache[key], params)}")
            return BaseResult(self._cache[key], params)
         
        if self.validator.is_valid(params):
            dur = self._run_estimator(params)
            if not self._is_right():
                dur = float('inf')
        else:
            dur = float("inf")
        result = BaseResult(dur, params)  
        if self.verbose:
            print(f"RUN: {result}")
            
        self._cache[key] = dur
        self._save_cache()
        
        return result
    
    
    def _is_right(self,
                  absolute_tol: float = 1e-9,
                  error_tol: float = 1e-4) -> bool:
        output = np.fromfile("./output/output.bin", dtype=np.float32).reshape(-1)
        golden = np.fromfile("./output/golden.bin", dtype=np.float32).reshape(-1)
        different_elements_num = (np.abs(output - golden) >= absolute_tol).sum()
        error_ratio = different_elements_num / golden.size
        if self.verbose:
            print(f"IS_RIGHT: {error_ratio <= error_tol}, error ratio = {error_ratio}")
        return error_ratio <= error_tol
