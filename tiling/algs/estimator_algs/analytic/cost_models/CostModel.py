from tiling.limits import MatmulLimits
from tiling.algs.estimator_algs.analytic.data import DataInfoBlock, TilingInfo, Traffic, HardwareInfo
from tiling.algs.estimator_algs.analytic.Cache import Cache

class CostModel:
    SHARED = ("GM", "L2")
    PRIVATE = ("L1", "L0A", "L0B", "L0C")
    def __init__(self,
                 tiling: TilingInfo,
                 limits: MatmulLimits,
                 hardware: HardwareInfo):
        self.tiling = tiling
        self.limits = limits
        self.num_cores = hardware.num_cores
        self.dtype_size = limits.dtype_size
        self.cube_mac_per_cycle = hardware.cube_mac_per_cycle
        self.mte2_bytes_per_cycle = hardware.mte2_bytes_per_cycle
        self.mte1_bytes_per_cycle = hardware.mte1_bytes_per_cycle
        self.fixp_bytes_per_cycle = hardware.fixp_bytes_per_cycle
        self.hbm_bytes_per_cycle = hardware.hbm_bytes_per_cycle
        self.l2_bytes_per_cycle = hardware.l2_bytes_per_cycle
        self.freq_hz = hardware.freq_hz
        self.vec_bytes_per_cycle = hardware.vec_bytes_per_cycle
        self.mte3_bytes_per_cycle = hardware.mte3_bytes_per_cycle
        self.vec_cores_per_cube = hardware.vec_cores_per_cube
        self.c_dtype_size = hardware.c_dtype_size
        self.num_vec_cores = hardware.num_cores * hardware.vec_cores_per_cube

    def _init_caches(self) -> None:
        self.caches: dict[str, Cache] = {
            "GM": Cache("GM", float('inf')),
            "L2": Cache("L2", self.limits.L2_size),
        }
        self.core_caches: list[dict[str, Cache]] = [
            {
                "L1": Cache("L1", self.limits.L1_size),
                "L0A": Cache("L0A", self.limits.L0A_size // self.tiling.dbL0A),
                "L0B": Cache("L0B", self.limits.L0B_size // self.tiling.dbL0B),
                "L0C": Cache("L0C", self.limits.L0C_size // self.tiling.dbL0C),
            }
            for _ in range(self.num_cores)
        ]
        self.traffic = Traffic()

    def _cache(self, cache_type: str, core: int) -> Cache:
        if cache_type in self.SHARED:
            return self.caches[cache_type]
        return self.core_caches[core][cache_type]

    def __call__(self, M: int, N: int, K: int) -> Traffic:
        self._init_caches()
        self.M, self.N, self.K = M, N, K
        return self.cost()

    def tile_shape(self, matrix_name: str, cache_type: str) -> tuple[int, int]:
        t = self.tiling
        if cache_type in ("L1", "L2"):
            if matrix_name == "A":
                return (t.stepM * t.baseM, t.stepKa * t.baseK)
            return (t.stepKb * t.baseK, t.stepN * t.baseN)
        if matrix_name == "A":
            return (t.baseM, t.baseK)
        return (t.baseK, t.baseN)

    def total_dims(self, matrix_name: str) -> tuple[int, int]:
        return (self.M, self.K) if matrix_name == "A" else (self.K, self.N)

    def block_at(self, matrix_name: str, cache_type: str,
                 fine_index: tuple[int, int]) -> DataInfoBlock:
        tile = self.tile_shape(matrix_name, cache_type)
        base = self.tile_shape(matrix_name, "GM")
        dims = self.total_dims(matrix_name)
        index = tuple(fine_index[d] * base[d] // tile[d] for d in range(2))
        shape = tuple(min(tile[d], dims[d] - index[d] * tile[d]) for d in range(2))
        return DataInfoBlock(matrix_name, cache_type, index, shape, self.dtype_size)

    def cost(self) -> Traffic:
        raise NotImplementedError

    def to_cache(self, cache_type: str, block: DataInfoBlock, core: int) -> int:
        tagged = block.with_cache_type(cache_type)
        self._cache(cache_type, core).add_data(tagged)
        return tagged.size

    def from_cache(self, cache_type: str, matrix_name: str,
                   cache_index: tuple[int, int], core: int) -> DataInfoBlock | None:
        name = DataInfoBlock.get_name(matrix_name, cache_type, cache_index)
        return self._cache(cache_type, core).get_data(name)

    def peek_cache(self, cache_type: str, matrix_name: str,
                   cache_index: tuple[int, int], core: int) -> DataInfoBlock | None:
        name = DataInfoBlock.get_name(matrix_name, cache_type, cache_index)
        return self._cache(cache_type, core).peek(name)

    def gm_to_l2(self, matrix_name: str, cache_index: tuple[int, int], core: int) -> int:
        block = self.block_at(matrix_name, "L2", cache_index)
        moved = self.to_cache("L2", block, core)
        self.traffic.gm_to_l2 += moved
        return moved

    def gm_to_l1(self, matrix_name: str, cache_index: tuple[int, int], core: int) -> int:
        l1_block = self.block_at(matrix_name, "L1", cache_index)
        transfer_cost = 0
        if self.from_cache("L2", matrix_name, l1_block.cache_index, core) is None:
            transfer_cost += self.gm_to_l2(matrix_name, cache_index, core)
        moved = self.to_cache("L1", l1_block, core)
        self.traffic.l2_to_l1 += moved
        return transfer_cost + moved

    def l1_to_l0AB(self, matrix_name: str, cache_index: tuple[int, int], core: int) -> int:
        dst = f"L0{matrix_name}"
        l0_block = self.block_at(matrix_name, dst, cache_index)
        if self.from_cache(dst, matrix_name, l0_block.cache_index, core) is not None:
            return 0

        l1_block = self.block_at(matrix_name, "L1", cache_index)
        transfer_cost = 0
        if self.from_cache("L1", matrix_name, l1_block.cache_index, core) is None:
            transfer_cost += self.gm_to_l1(matrix_name, cache_index, core)

        moved = self.to_cache(dst, l0_block, core)
        self.traffic.l1_to_l0 += moved
        return transfer_cost + moved

    def alloc_l0c(self, cache_index: tuple[int, int], core: int,
                  shape: tuple[int, int], dtype_size: int = 4) -> int:
        block = DataInfoBlock("C", "L0C", cache_index, shape, dtype_size)
        return self.to_cache("L0C", block, core)

    def l0c_to_gm(self, cache_index: tuple[int, int], core: int,
                  atomic: bool = False, final: bool = True) -> int:
        """
        Выгрузка аккумулятора. Идёт через L2, поэтому плитка C может
        остаться там резидентной между кусками K — без этого split-K
        был бы неприемлемо дорог.

        atomic: read-modify-write, при промахе в L2 старое значение
                подтягивается из памяти.
        final:  последняя выгрузка этой плитки, её и видит HBM.
        """
        block = self.from_cache("L0C", "C", cache_index, core)
        moved = block.size
        self.traffic.l0c_to_gm += moved
        if atomic:
            self.traffic.l0c_atomic += moved

        if self.from_cache("L2", "C", cache_index, core) is None:
            if atomic:
                self.traffic.gm_to_l2 += moved
            self.to_cache("L2", block.with_cache_type("L2"), core)
        if final:
            self.traffic.c_hbm_bytes += moved
        return moved

    def mmad(self, m: int, k: int, n: int) -> None:
        self.traffic.mac_count += m * k * n

    def drop_l1(self, core: int) -> None:
        self.core_caches[core]["L1"].clear()

    def rounds_over(self, work: list):
        for start in range(0, len(work), self.num_cores):
            chunk = work[start:start + self.num_cores]
            self.traffic.rounds += 1
            self.traffic.core_slots_used += len(chunk)
            yield list(enumerate(chunk))

    def time_estimate(self) -> dict[str, float]:
        """Traffic -> такты. Вызывать после __call__."""
        tr = self.traffic
        t = self.tiling
        occ = tr.occupancy(self.num_cores)
        eff = max(1e-9, self.num_cores * occ)

        # приватные трубы: глобальные счётчики делим на эффективное число ядер
        cube = tr.mac_count / (eff * self.cube_mac_per_cycle)
        mte2 = tr.l2_to_l1 / (eff * self.mte2_bytes_per_cycle)
        mte1 = tr.l1_to_l0 / (eff * self.mte1_bytes_per_cycle)
        fixp = tr.l0c_to_gm / (eff * self.fixp_bytes_per_cycle)

        # двойная буферизация перекрывает трубы, одинарная заставляет ждать
        core = max(cube, mte1) if t.dbL0A == 2 and t.dbL0B == 2 else cube + mte1
        core = max(core, mte2)              # L1 всегда с двойной буферизацией
        core = max(core, fixp) if t.dbL0C == 2 else core + fixp

        # общие на кристалл: считаются от глобальных сумм
        hbm = tr.hbm_bytes / self.hbm_bytes_per_cycle
        l2 = tr.l2_to_l1 / self.l2_bytes_per_cycle

        parts = {"cube": cube, "mte1": mte1, "mte2": mte2,
                 "fixp": fixp, "hbm": hbm, "l2": l2}
        cycles = max(core, hbm, l2)
        return {"cycles": cycles, "seconds": cycles / self.freq_hz,
                "core_path": core, **parts}

