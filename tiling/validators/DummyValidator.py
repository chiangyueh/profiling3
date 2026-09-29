from tiling.base import BaseValidator, BaseParam
from typing import Iterator
import itertools

class DummyValidator(BaseValidator):

    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        ps_dict = {p.name: p for p in params}
        stepM = ps_dict['MM_STEP_M'].value
        stepN = ps_dict['MM_STEP_N'].value
        stepKa = ps_dict['MM_STEP_Ka'].value
        stepKb = ps_dict['MM_STEP_Kb'].value

        derived = {
            'MM_DEPTH_A1': stepM * stepKa,
            'MM_DEPTH_B1': stepN * stepKb * 2,
        }

        return [self._make_param(name, value, True) for name, value in derived.items()]

    def iter_combinations(self, const_params: list[BaseParam]) -> Iterator[list[BaseParam]]:
        """Ленивый полный перебор декартова произведения domains.

        Комбинации не материализуются: пространство поиска легко доходит до
        миллиардов конфигов.
        """
        names = list(self.limits.domains.keys())
        domains = [self.limits.domains[name] for name in names]

        for values in itertools.product(*domains):
            params = [self._make_param(p.name, p.value, True) for p in const_params]
            for name, value in zip(names, values):
                params.append(self._make_param(name, value, False, self.limits.domains[name]))
            yield self.get_all_params(params)

    def get_combinations(self, num: int, const_params: list[BaseParam]) -> list[list[BaseParam]]:
        return list(itertools.islice(self.iter_combinations(const_params), num))
