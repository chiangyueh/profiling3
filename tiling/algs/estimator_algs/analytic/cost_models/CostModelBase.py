from tiling.limits import MatmulLimits
from tiling.algs.estimator_algs.analytic.data import DataInfoBlock, TilingInfo, Traffic, HardwareInfo
from .CostModel import CostModel
from tiling.algs.estimator_algs.analytic.Cache import Cache


class CostModelBase(CostModel):
    def cost(self) -> Traffic:
        t = self.tiling

        m_total = DataInfoBlock._ceil_div(self.M, t.baseM)
        n_total = DataInfoBlock._ceil_div(self.N, t.baseN)
        k_total = DataInfoBlock._ceil_div(self.K, t.baseK)

        m_patches = DataInfoBlock._ceil_div(m_total, t.mTileBlock)
        n_patches = DataInfoBlock._ceil_div(n_total, t.nTileBlock)

        for mp in range(m_patches):
            for step in range(n_patches):
                np_ = (n_patches - 1 - step) if (mp % 2) else step

                blocks = [
                    (m, n)
                    for m in range(mp * t.mTileBlock,
                                   min((mp + 1) * t.mTileBlock, m_total))
                    for n in range(np_ * t.nTileBlock,
                                   min((np_ + 1) * t.nTileBlock, n_total))
                ]

                for active in self.rounds_over(blocks):
                    self._run_round(active, k_total)

        return self.traffic

    def _run_round(self, active, k_total: int) -> None:
        t = self.tiling

        for core, _ in active:
            self.drop_l1(core)

        for core, (m, n) in active:
            rows = min(t.baseM, self.M - m * t.baseM)
            cols = min(t.baseN, self.N - n * t.baseN)
            self.alloc_l0c((m, n), core, (rows, cols))

        for k in range(k_total):
            for core, (m, n) in active:
                self.l1_to_l0AB("A", (m, k), core)
                self.l1_to_l0AB("B", (k, n), core)

                rows = min(t.baseM, self.M - m * t.baseM)
                cols = min(t.baseN, self.N - n * t.baseN)
                depth = min(t.baseK, self.K - k * t.baseK)
                self.mmad(rows, depth, cols)

        for core, (m, n) in active:
            self.l0c_to_gm((m, n), core)


