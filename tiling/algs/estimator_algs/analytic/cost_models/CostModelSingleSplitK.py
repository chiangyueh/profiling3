from tiling.limits import MatmulLimits
from tiling.algs.estimator_algs.analytic.data import DataInfoBlock, TilingInfo, Traffic, HardwareInfo
from .CostModel import CostModel
from tiling.algs.estimator_algs.analytic.Cache import Cache

class CostModelSingleSplitK(CostModel):
    """
    Обход single-core split-K (rowOrder == 0, схема MKN).

        for блок C (singleCoreM x singleCoreN), раздача ПЛОСКАЯ
          for полоска по M, высотой stepM*baseM
            for кусок K, глубиной singleCoreK      <- весь K здесь
              for плитка по N (baseN)              <- A резидентна в L1
                for плитка по M (baseM)
                  for шаг по K (baseK): MMAD
                  L0C -> GM: k==0 запись, иначе:w атомарное сложение

    Отличия от base, влияющие на счётчики:
      * L1 не сбрасывается — окно A живёт через весь проход по N,
        поэтому l1_to_l0 заметно больше l2_to_l1
      * плитка C выгружается loopK раз, а не один
      * патчинга нет: index = blockIdx * round, блоки идут подряд
    """

    def cost(self) -> Traffic:
        t = self.tiling
        ceil_div = DataInfoBlock._ceil_div

        m_cnt = ceil_div(self.M, t.singleCoreM)
        n_cnt = ceil_div(self.N, t.singleCoreN)
        total_cnt = m_cnt * n_cnt
        loop_k = ceil_div(self.K, t.singleCoreK)

        inner_block_m = t.stepM * t.baseM
        m_core_tail = self.M - (m_cnt - 1) * t.singleCoreM
        n_core_tail = self.N - (n_cnt - 1) * t.singleCoreN
        k_core_tail = self.K - (loop_k - 1) * t.singleCoreK

        per_core = self._assign(total_cnt)
        max_round = max((len(blocks) for blocks in per_core), default=0)

        for j in range(max_round):
            active = [(core, blocks[j]) for core, blocks in enumerate(per_core)
                      if j < len(blocks)]
            self.traffic.rounds += 1
            self.traffic.core_slots_used += len(active)

            geom = {}
            for core, index in active:
                m_index, n_index = index % m_cnt, index // m_cnt
                m_use = m_core_tail if m_index == m_cnt - 1 else t.singleCoreM
                n_use = n_core_tail if n_index == n_cnt - 1 else t.singleCoreN
                geom[core] = (m_index, n_index, m_use, n_use,
                              ceil_div(m_use, inner_block_m))

            inner_loop_m = max(g[4] for g in geom.values())

            for inner_m in range(inner_loop_m):
                for k in range(loop_k):
                    k_use = k_core_tail if k == loop_k - 1 else t.singleCoreK
                    for core, _ in active:
                        m_index, n_index, m_use, n_use, loops = geom[core]
                        if inner_m >= loops:
                            continue
                        self._task(core, m_index, n_index, m_use, n_use,
                                   inner_m, inner_block_m, k, k_use,
                                   atomic=(k > 0), final=(k == loop_k - 1))

        return self.traffic

    def _assign(self, total_cnt: int) -> list[list[int]]:
        """InitBlockIndex: непрерывный отрезок индексов на ядро."""
        ceil_div = DataInfoBlock._ceil_div
        round_ = ceil_div(total_cnt, self.num_cores)
        pre = total_cnt % self.num_cores or self.num_cores
        assignment = []
        for core in range(self.num_cores):
            if core < pre:
                start, real = core * round_, round_
            else:
                start, real = core * (round_ - 1) + pre, round_ - 1
            assignment.append([start + j for j in range(max(0, real))])
        return assignment

    def _task(self, core, m_index, n_index, m_use, n_use,
              inner_m, inner_block_m, k, k_use, atomic, final) -> None:
        """Одно задание библиотеке: область stepM*baseM x nCoreUse, глубина kCoreUse."""
        t = self.tiling
        ceil_div = DataInfoBlock._ceil_div

        row0 = m_index * t.singleCoreM + inner_m * inner_block_m
        rows = min(inner_block_m, m_use - inner_m * inner_block_m)
        col0 = n_index * t.singleCoreN
        k0 = k * t.singleCoreK

        m_tile0, n_tile0, k_tile0 = row0 // t.baseM, col0 // t.baseN, k0 // t.baseK

        # N снаружи, M внутри: окно A держится в L1 через все столбцы
        for nt in range(ceil_div(n_use, t.baseN)):
            n_tile = n_tile0 + nt
            for mt in range(ceil_div(rows, t.baseM)):
                m_tile = m_tile0 + mt
                shape = (min(t.baseM, self.M - m_tile * t.baseM),
                         min(t.baseN, self.N - n_tile * t.baseN))
                self.alloc_l0c((m_tile, n_tile), core, shape)

                for ks in range(ceil_div(k_use, t.baseK)):
                    k_tile = k_tile0 + ks
                    self.l1_to_l0AB("A", (m_tile, k_tile), core)
                    self.l1_to_l0AB("B", (k_tile, n_tile), core)
                    self.mmad(shape[0],
                              min(t.baseK, self.K - k_tile * t.baseK),
                              shape[1])

                self.l0c_to_gm((m_tile, n_tile), core, atomic=atomic, final=final)
