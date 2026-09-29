from tiling.base import BaseParam
from tiling.limits import MatmulLimits
from .MatmulValidator import MatmulValidator
from typing import Callable    

NCALC_THRES = 16
DB_SIZE = 2
VECTOR_D_BASE = 2048
CALC_ND_BASIC = (6144, 4096, 2048)
MIN_TAIL = 512  
BLOCK_BYTE_SIZE = 32
N_ALIGNED = 16
MATA_D = 16384
MATA_MIN_N = 7168


class MatmulSplitKValidator(MatmulValidator):
    def __init__(self,
                 limits: MatmulLimits,
                 param_funcs: dict[
                     str, list[tuple[Callable[[dict[str, BaseParam]], bool | tuple[bool]],
                                     Callable[[dict[str, BaseParam]], dict[BaseParam]]]]] = {}
                 ) -> None:
        super().__init__(limits, param_funcs)
        self.param_funcs.update({
            'single_core': (self._single_core_is_valid,
                            self._repair_single_core),
            'core_num': (self._core_num_is_valid,
                         self._repair_core_num),
            'depth': (self._depth_is_valid,
                      self._repair_depth),
            'kb_full_load': (self._kb_full_load_is_valid,
                             self._repair_kb_full_load),
        })

    def _depth_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool]:
        """depthA1/depthB1 задают двойную буферизацию L1, отдельно для A и B.

        Matmul API проверяет это в DepthCheck() (matmul_shape_tiling.h):
        depth должен делиться на stepM*stepKa (соотв. stepN*stepKb), а частное
        быть 1 или 2. Частное и есть признак DB: в планировщике
        cacheA1Factor_ = (depthA1 / (stepM * stepKa) - 1) & 1.
        """
        depthA1 = params['MM_DEPTH_A1'].value
        depthB1 = params['MM_DEPTH_B1'].value
        cacheA1 = params['MM_STEP_M'].value * params['MM_STEP_Ka'].value
        cacheB1 = params['MM_STEP_N'].value * params['MM_STEP_Kb'].value

        return (depthA1 % cacheA1 == 0, depthA1 // cacheA1 in (1, DB_SIZE),
                depthB1 % cacheB1 == 0, depthB1 // cacheB1 in (1, DB_SIZE))

    def _repair_depth(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_DEPTH_A1', 'MM_DEPTH_B1']
        add_param_names = ['MM_STEP_M', 'MM_STEP_N', 'MM_STEP_Ka', 'MM_STEP_Kb',
                           'MM_BASE_M', 'MM_BASE_N', 'MM_BASE_K']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)
        add_ctx = {name: params[name] for name in add_param_names}

        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values, **add_ctx}
            depth = all(self._depth_is_valid(new_params))
            l1 = self._l1_size_is_valid(new_params)
            return depth and l1

        repaired_params = self.repair_dijkstra(movable_params, is_valid, context=params)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}

    def _kb_full_load_is_valid(self, params: dict[str, BaseParam]) -> bool:
        """B должна быть целиком загружена по K в L1.

        Оба целевых ядра собраны с MM_CFG_PRELOAD_MK = GetMDLConfig(.., 2),
        то есть doMTE2Preload = PRELOAD_N, а планировщик на это держит ASSERT
        IsBKL1FullLoad(), который раскрывается как stepKb >= Ceil(singleK, baseK)
        (k_loop_mdl_base.h: isB1KFullLoad_ = stepKb >= kIter_).
        """
        stepKb = params['MM_STEP_Kb'].value
        singleK = params['MM_SINGLE_K'].value
        baseK = params['MM_BASE_K'].value
        return stepKb >= self._ceil_div(singleK, baseK)

    def _repair_kb_full_load(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_STEP_Kb', 'MM_SINGLE_K', 'MM_BASE_K']
        add_param_names = ['MM_STEP_M', 'MM_STEP_N', 'MM_STEP_Ka',
                           'MM_BASE_M', 'MM_BASE_N', 'MM_DEPTH_A1', 'MM_DEPTH_B1']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)
        add_ctx = {name: params[name] for name in add_param_names}

        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values, **add_ctx}
            kb = self._kb_full_load_is_valid(new_params)
            depth = all(self._depth_is_valid(new_params))
            l1 = self._l1_size_is_valid(new_params)
            l0 = all(self._base_tiles_is_valid(new_params))
            return kb and depth and l1 and l0

        repaired_params = self.repair_dijkstra(movable_params, is_valid, context=params)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}

    def _core_num_is_valid(self, params: dict[str, BaseParam]) -> bool:
        num_cores = params['MM_CORE_NUM'].value
        return 1 <= num_cores <= self.limits.max_cores

    def _repair_core_num(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_CORE_NUM']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)

        def is_valid(values: dict[str, int]) -> bool:
            return self._core_num_is_valid({**const_params, **values})

        repaired_params = self.repair_dijkstra(movable_params, is_valid, context=params)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}

    def _single_core_is_valid(self, params: dict[str, BaseParam]) -> tuple[bool]:
        used_param_names = ['MM_SINGLE_M', 'MM_SINGLE_N', 'MM_SINGLE_K',
                            'MM_BASE_M', 'MM_BASE_N', 'MM_BASE_K']
        singleM, singleN, singleK, baseM, baseN, baseK = [params[name].value
                                                          for name in used_param_names]
        return (singleM >= baseM, singleN >= baseN, singleK >= baseK)

    def _repair_single_core(self, params: dict[str, BaseParam]) -> dict[BaseParam]:
        used_param_names = ['MM_SINGLE_M', 'MM_SINGLE_N', 'MM_SINGLE_K',
                            'MM_BASE_M', 'MM_BASE_N', 'MM_BASE_K']
        add_param_names = ['MM_STEP_M', 'MM_STEP_N', 'MM_STEP_Ka', 'MM_STEP_Kb',
                           'MM_DB_L0A', 'MM_DB_L0B', 'MM_DB_L0C']
        const_params = self._const_params(params, used_param_names)
        movable_params = self._movable_params(params, used_param_names)
        add_ctx = {name: params[name] for name in add_param_names}

        def is_valid(values: dict[str, int]) -> bool:
            new_params = {**const_params, **values, **add_ctx}
            single = all(self._single_core_is_valid(new_params))
            l1 = self._l1_size_is_valid(new_params)
            l0 = all(self._base_tiles_is_valid(new_params))
            return single and l1 and l0

        repaired_params = self.repair_dijkstra(movable_params, is_valid, context=params)
        return {name: repaired_params.get(name, params[name])
                for name in used_param_names}

    # --- тайлинг ND->NZ конверсии (CalcNd2NzTiling на хосте) ---------------

    def _ub_overflow(self, n_aligned16: int, n_value: int,
                     base_n: int, base_d: int, dtype_size: int) -> bool:
        if base_n == 0 or dtype_size == 0:
            return False
        if self._ceil_div(n_aligned16, base_n) == self._ceil_div(n_value, base_n):
            return False
        return ((n_aligned16 - (n_value // base_n - 1) * base_n) * base_d
                > self.limits.UB_size // 2 // dtype_size)

    def _calc_nd2nz(self, n_value: int, d_value: int,
                    dtype_size: int, used_cores: int) -> tuple[int, int]:
        """Возвращает (baseN, baseD) для конверсии одного операнда."""
        # ветка mata
        if (d_value % MATA_D == 0 and n_value >= MATA_MIN_N
                and dtype_size == 2 and self.limits.max_cores >= 24):
            return 96, 512

        c0 = BLOCK_BYTE_SIZE // dtype_size
        n_aligned16 = self._align_up(n_value, N_ALIGNED)
        d_aligned_c0 = self._align_up(d_value, c0)
        vector_cores = max(2 * used_cores, 1)

        # мелкая глубина: baseD покрывает весь dValue
        if d_value <= VECTOR_D_BASE // dtype_size:
            base_d = max(self._align_up(d_value, c0), 1)
            base_n = self.limits.UB_size // 2 // dtype_size // base_d
            rnd = max(self._ceil_div(self._ceil_div(n_aligned16, vector_cores),
                                     max(base_n, 1)), 1)
            base_n = max(self._ceil_div(self._ceil_div(n_aligned16, vector_cores), rnd),
                         NCALC_THRES)
            while base_n > NCALC_THRES and self._ub_overflow(
                    n_aligned16, n_value, base_n, base_d, dtype_size):
                base_n -= 1
            return base_n, base_d

        last_tail = 0
        best_n = NCALC_THRES
        best_d = CALC_ND_BASIC[1] // dtype_size
        for base in CALC_ND_BASIC:
            base_d = max(min(d_aligned_c0, base // dtype_size), 1)
            d_loop = self._ceil_div(d_aligned_c0, base_d)
            d_tail = d_aligned_c0 % base_d
            if 0 < d_tail < MIN_TAIL // dtype_size:
                if base_d * dtype_size == CALC_ND_BASIC[0]:
                    continue
                d_loop -= 1
                if d_loop < 1:
                    continue
                base_d = max(self._align_up(self._ceil_div(d_aligned_c0, d_loop), c0), 1)

            base_n = max(self.limits.UB_size // 2 // dtype_size // base_d, NCALC_THRES)
            if base_n * base_d * dtype_size * 2 > self.limits.UB_size:
                continue
            if self._ub_overflow(n_aligned16, n_value, base_n, base_d, dtype_size):
                continue

            n_loop = self._ceil_div(n_aligned16, base_n)
            tail = (n_loop * d_loop) % vector_cores
            while base_n > NCALC_THRES:
                if self._ub_overflow(n_aligned16, n_value, base_n, base_d, dtype_size):
                    base_n -= 1
                    n_loop = self._ceil_div(n_aligned16, base_n)
                    tail = (n_loop * d_loop) % vector_cores
                    continue
                if tail == 0:
                    return base_n, base_d
                if tail > last_tail:
                    last_tail, best_d, best_n = tail, base_d, base_n
                base_n -= 1
                n_loop = self._ceil_div(n_aligned16, base_n)
                tail = (n_loop * d_loop) % vector_cores
        return best_n, best_d

    def _nd2nz_derived(self, ps_dict: dict[str, BaseParam],
                       used_cores: int) -> dict[str, int]:
        """
        baseAN/baseAD/baseBN/baseBD. Считаются только при взведённом nd2nz,
        иначе ядро их не читает и хост оставляет нули.
        Транспонирования нет, поэтому для A это (M, K), для B — (K, N).
        """
        M = ps_dict['MM_M'].value
        N = ps_dict['MM_N'].value
        K = ps_dict['MM_K'].value
        ds = self.limits.dtype_size

        an = ad = bn = bd = 0
        if ps_dict['MM_ND2NZ_A'].value:
            an, ad = self._calc_nd2nz(M, K, ds, used_cores)
        if ps_dict['MM_ND2NZ_B'].value:
            bn, bd = self._calc_nd2nz(K, N, ds, used_cores)

        return {'MM_ND2NZ_BASE_AN': an, 'MM_ND2NZ_BASE_AD': ad,
                'MM_ND2NZ_BASE_BN': bn, 'MM_ND2NZ_BASE_BD': bd}


class MatmulSingleSplitKValidator(MatmulSplitKValidator):
    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        # singleCoreM/N/K и число ядер хост выводит сам, но это его эвристика,
        # а не требование ядра: kCnt/mCnt/nCnt считаются с явными хвостами.
        # Поэтому они остаются свободными, а ограничения на них — в param_funcs.
        ps_dict = {p.name: p for p in params}
        num_cores = ps_dict['MM_CORE_NUM'].value

        derived = dict(self._nd2nz_derived(ps_dict, num_cores))
        return [self._make_param(name, value, True) for name, value in derived.items()]


class MatmulDetSplitKValidator(MatmulSplitKValidator):
    def get_derived_params(self, params: list[BaseParam]) -> list[BaseParam]:
        # См. комментарий в MatmulSingleSplitKValidator: singleCoreM/N/K и число
        # ядер свободны, хост фиксирует их только эвристикой блока 3x3.
        ps_dict = {p.name: p for p in params}
        num_cores = ps_dict['MM_CORE_NUM'].value

        derived = dict(self._nd2nz_derived(ps_dict, num_cores))
        return [self._make_param(name, value, True) for name, value in derived.items()]