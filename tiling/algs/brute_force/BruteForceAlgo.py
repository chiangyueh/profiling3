from tiling.base import BaseResult, BaseAlgo, BaseParam
from typing import Iterator
import itertools

class BruteForceAlgo(BaseAlgo):
    def run(self, *args, **kwargs) -> BaseResult:
        best_dur = float('inf')
        best_params: list[BaseParam] | None = None
        skipped = 0

        for step, comb in enumerate(self._iter_combinations(), start=1):
            if not self.validator.is_valid(comb):
                skipped += 1
                continue

            result = self._duration(comb)
            if result.duration < best_dur:
                best_dur = result.duration
                best_params = result.params
                if self.verbose:
                    print(f"NEW BEST (comb={step}): {result}")

        if self.verbose:
            print(f"SKIPPED INVALID: {skipped}")
            if best_params is None:
                print("NO VALID COMBINATION")

        return BaseResult(best_dur, best_params if best_params is not None else [])

    def _iter_combinations(self) -> Iterator[list[BaseParam]]:
        domains = self.validator.limits.domains
        names = list(domains.keys())

        for values in itertools.product(*(domains[name] for name in names)):
            params = [BaseParam(name=p.name, value=p.value, is_const=True)
                      for p in self.input_params]
            for name, value in zip(names, values):
                params.append(BaseParam(name=name, value=value,
                                        is_const=False, domain=domains[name]))
            yield self.validator.get_all_params(params)
