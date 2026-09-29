
from tiling.limits import MatmulLimits
from tiling.algs.estimator_algs.analytic.data import DataInfoBlock, TilingInfo, Traffic, HardwareInfo
from .CostModel import CostModel
from tiling.algs.estimator_algs.analytic.Cache import Cache

class CostModelDetSplitK(CostModel):
    """
    Детерминированный split-K: две фазы на двух типах ядер.

    ФАЗА 1 (кубические ядра, AIC)
        Матрица C режется на полосы. При orderFlag полоса = M x singleCoreN,
        иначе singleCoreM x N. Полосы обходятся внешним циклом outIndex.
        Внутри полосы куски K раздаются ПО ВСЕМ ядрам: ядро берёт отрезок
        [index, index+realRound) и сворачивает свои куски в СВОЙ срез
        workspace (первый кусок — запись, остальные — atomic add).
        Срезов ровно usedCoreNum, независимо от kCnt.

    ФАЗА 2 (векторные ядра, AIV)
        Читают все usedCoreNum срезов, складывают в UB в фиксированном
        порядке и пишут полосу в C. Отсюда детерминизм: atomic нет,
        порядок сложений не зависит от того, кто успел раньше.
    """
    def cost(self) -> Traffic:
        t = self.tiling
        ceil_div = DataInfoBlock._ceil_div

        order_flag = (t.iterateOrder == 0)   # orderFlag = !iterateOrder

        m_cnt = ceil_div(self.M, t.singleCoreM)
        n_cnt = ceil_div(self.N, t.singleCoreN)
        k_cnt = ceil_div(self.K, t.singleCoreK)
        out_cnt = n_cnt if order_flag else m_cnt

        per_core = self._assign_k(k_cnt)
        max_real = max((len(ks) for ks in per_core), default=0)

        for out_index in range(out_cnt):
            row0, rows, col0, cols = self._polosa(out_index, order_flag)

            # ── фаза 1: куб ──
            for j in range(max_real):
                active = [(core, ks[j]) for core, ks in enumerate(per_core)
                          if j < len(ks)]
                self.traffic.rounds += 1
                self.traffic.core_slots_used += len(active)

                for core, k_index in active:
                    k_use = min(t.singleCoreK, self.K - k_index * t.singleCoreK)
                    self._cube_task(core, row0, rows, col0, cols,
                                    k_index * t.singleCoreK, k_use,
                                    atomic=(j > 0))

            # ── фаза 2: вектор ──
            self._vector_reduce(row0, rows, col0, cols)

        return self.traffic

    def _assign_k(self, k_cnt: int) -> list[list[int]]:
        """Куски K раздаются ядрам непрерывными отрезками (как в ядре)."""
        ceil_div = DataInfoBlock._ceil_div
        round_ = ceil_div(k_cnt, self.num_cores)
        pre = k_cnt % self.num_cores or self.num_cores
        out = []
        for core in range(self.num_cores):
            if core < pre:
                start, real = core * round_, round_
            else:
                start, real = core * (round_ - 1) + pre, round_ - 1
            out.append([start + j for j in range(max(0, real))])
        return out

    def _polosa(self, out_index: int, order_flag: bool):
        """Геометрия полосы: (row0, rows, col0, cols)."""
        t = self.tiling
        if order_flag:                       # полосы по N, высота — вся M
            col0 = out_index * t.singleCoreN
            return 0, self.M, col0, min(t.singleCoreN, self.N - col0)
        row0 = out_index * t.singleCoreM     # полосы по M, ширина — вся N
        return row0, min(t.singleCoreM, self.M - row0), 0, self.N

    # ── фаза 1 ────────────────────────────────────────────────────────

    def _ws_index(self, core: int, m_tile: int, n_tile: int) -> tuple[int, int]:
        """
        Срез каждого ядра лежит в СВОЁЙ области workspace, поэтому индекс
        должен различаться по ядрам — иначе общий L2 склеит срезы разных
        ядер в одну строку и попадания будут фиктивными.
        """
        return (m_tile + core * self._m_tiles_stride, n_tile)

    def _cube_task(self, core, row0, rows, col0, cols, k0, k_use, atomic) -> None:
        t = self.tiling
        ceil_div = DataInfoBlock._ceil_div

        m_tile0, n_tile0, k_tile0 = row0 // t.baseM, col0 // t.baseN, k0 // t.baseK
        self._m_tiles_stride = ceil_div(self.M, t.baseM) + 1

        for nt in range(ceil_div(cols, t.baseN)):
            n_tile = n_tile0 + nt
            for mt in range(ceil_div(rows, t.baseM)):
                m_tile = m_tile0 + mt
                shape = (min(t.baseM, self.M - m_tile * t.baseM),
                         min(t.baseN, self.N - n_tile * t.baseN))
                ws_idx = self._ws_index(core, m_tile, n_tile)
                self.alloc_l0c(ws_idx, core, shape)

                for ks in range(ceil_div(k_use, t.baseK)):
                    k_tile = k_tile0 + ks
                    self.l1_to_l0AB("A", (m_tile, k_tile), core)
                    self.l1_to_l0AB("B", (k_tile, n_tile), core)
                    self.mmad(shape[0],
                              min(t.baseK, self.K - k_tile * t.baseK),
                              shape[1])

                # выгрузка в СВОЙ срез workspace, не в C:
                # final=False — окончательную запись сделает вектор
                self.l0c_to_gm(ws_idx, core, atomic=atomic, final=False)

    # ── фаза 2 ────────────────────────────────────────────────────────

    def _vector_reduce(self, row0, rows, col0, cols) -> None:
        """Чтение usedCoreNum срезов, сложение в UB, запись полосы в C."""
        t = self.tiling
        ceil_div = DataInfoBlock._ceil_div
        m_tile0, n_tile0 = row0 // t.baseM, col0 // t.baseN

        for core in range(self.num_cores):
            for nt in range(ceil_div(cols, t.baseN)):
                n_tile = n_tile0 + nt
                for mt in range(ceil_div(rows, t.baseM)):
                    m_tile = m_tile0 + mt
                    shape = (min(t.baseM, self.M - m_tile * t.baseM),
                             min(t.baseN, self.N - n_tile * t.baseN))
                    size = shape[0] * shape[1] * 4
                    ws_idx = self._ws_index(core, m_tile, n_tile)

                    if self.from_cache("L2", "C", ws_idx, 0) is None:
                        self.traffic.gm_to_l2 += size
                        self.to_cache("L2", DataInfoBlock("C", "L2", ws_idx,
                                                          shape, 4), 0)
                    self.traffic.ws_to_ub += size

        self.traffic.c_hbm_bytes += rows * cols * self.c_dtype_size

    # ── время ─────────────────────────────────────────────────────────

    def time_estimate(self) -> dict[str, float]:
        """
        Две фазы перекрываются пинг-понгом, поэтому берём max, а не сумму.
        Вектор работает на своих ядрах и своих трубах.
        """
        tr = self.traffic
        t = self.tiling
        occ = tr.occupancy(self.num_cores)
        eff = max(1e-9, self.num_cores * occ)

        cube = tr.mac_count / (eff * self.cube_mac_per_cycle)
        mte2 = tr.l2_to_l1 / (eff * self.mte2_bytes_per_cycle)
        mte1 = tr.l1_to_l0 / (eff * self.mte1_bytes_per_cycle)
        fixp = tr.l0c_to_gm / (eff * self.fixp_bytes_per_cycle)

        cube_path = max(cube, mte1) if t.dbL0A == 2 and t.dbL0B == 2 else cube + mte1
        cube_path = max(cube_path, mte2)
        cube_path = max(cube_path, fixp) if t.dbL0C == 2 else cube_path + fixp

        v = self.num_vec_cores
        vec_mte2 = tr.ws_to_ub / (v * self.mte2_bytes_per_cycle)
        vec_add = tr.ws_to_ub / (v * self.vec_bytes_per_cycle)
        vec_mte3 = tr.c_hbm_bytes / (v * self.mte3_bytes_per_cycle)
        vec_path = max(vec_mte2, vec_add, vec_mte3)   # пинг-понг в UB

        hbm = (tr.hbm_bytes + tr.ws_to_ub) / self.hbm_bytes_per_cycle
        l2 = (tr.l2_to_l1 + tr.l0c_to_gm + tr.ws_to_ub) / self.l2_bytes_per_cycle

        parts = {"cube": cube, "mte1": mte1, "mte2": mte2, "fixp": fixp,
                 "vec_mte2": vec_mte2, "vec_add": vec_add, "vec_mte3": vec_mte3,
                 "hbm": hbm, "l2": l2}
        cycles = max(cube_path, vec_path, hbm, l2)
        return {"cycles": cycles, "seconds": cycles / self.freq_hz,
                "cube_path": cube_path, "vec_path": vec_path, **parts}
