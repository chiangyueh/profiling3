from tiling.base import BaseResult, BaseAlgo, BaseValidator, BaseParam
from typing import Callable


class DummyAlgo(BaseAlgo):
    """Ничего не ищет: меряет ровно те конфиги, которые ему дали.

    Нужен, чтобы снимать замеры по готовому списку тайлингов — например
    baseline от автотайлинга — теми же средствами, что и поисковые алгоритмы:
    через _duration, то есть с кэшем и сверкой выхода с golden. Возвращает
    лучший из замеренных.
    """

    def __init__(self,
                 is_stop: Callable[[list[BaseResult]], bool],
                 validator: BaseValidator,
                 input_params: list[BaseParam],
                 configs: list[dict[str, int]],
                 runner: str = "./run.sh",
                 verbose: bool = False,
                 cache_path: str = "msprof_cache.json") -> None:
        super().__init__(is_stop, validator, input_params,
                         runner=runner, verbose=verbose, cache_path=cache_path)
        self.configs = configs

    def run(self, *args, **kwargs) -> BaseResult:
        best_dur = float('inf')
        best_params: list[BaseParam] | None = None

        for step, config in enumerate(self.configs, start=1):
            # config перекрывает input_params, а не дополняет: иначе одно имя
            # попало бы в набор дважды и в env уехало бы неизвестно какое
            values = {p.name: p.value for p in self.input_params}
            values.update(config)
            params = [BaseParam(name=name, value=value, is_const=True)
                      for name, value in values.items()]

            result = self._duration(params)
            if self.verbose:
                print(f"CONFIG {step}/{len(self.configs)}: {result}")

            if result.duration < best_dur:
                best_dur = result.duration
                best_params = result.params

        if self.verbose and best_params is None:
            print("NO MEASURED CONFIG")

        return BaseResult(best_dur, best_params if best_params is not None else [])
