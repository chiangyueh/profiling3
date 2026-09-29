from dataclasses import dataclass
import itertools

@dataclass
class TilingInfo:
    baseM: int
    baseN: int
    baseK: int
    singleCoreM: int
    singleCoreN: int
    singleCoreK: int
    mTileBlock: int
    nTileBlock: int
    stepM: int
    stepN: int
    stepKa: int
    stepKb: int
    dbL0A: int
    dbL0B: int
    dbL0C: int
    iterateOrder: int = 0

@dataclass
class Traffic:
    gm_to_l2: int = 0
    l2_to_l1: int = 0
    l1_to_l0: int = 0
    l0c_to_gm: int = 0       
    c_hbm_bytes: int = 0    
    l0c_atomic: int = 0      

    ws_to_ub: int = 0       

    mac_count: int = 0
    rounds: int = 0
    core_slots_used: int = 0

    @property
    def hbm_bytes(self) -> int:
        return self.gm_to_l2 + self.c_hbm_bytes

    def occupancy(self, used_cores: int) -> float:
        total = self.rounds * used_cores
        return self.core_slots_used / total if total else 0.0

    @property
    def arithmetic_intensity(self) -> float:
        return (2 * self.mac_count) / self.hbm_bytes if self.hbm_bytes else float("inf")

@dataclass
class HardwareInfo:
    num_cores: int = 20
    cube_mac_per_cycle: float = 4096.0   # на ядро
    mte2_bytes_per_cycle: float = 64.0   # GM/L2 -> L1, на ядро
    mte1_bytes_per_cycle: float = 256.0  # L1 -> L0,    на ядро
    fixp_bytes_per_cycle: float = 64.0   # L0C -> GM,   на ядро
    hbm_bytes_per_cycle: float = 888.9   # на кристалл
    l2_bytes_per_cycle: float = 2777.8   # на кристалл
    freq_hz: float = 1.8e9
    vec_bytes_per_cycle: float = 128.0   # Add в UB,  на векторное ядро
    mte3_bytes_per_cycle: float = 64.0   # UB -> GM,  на векторное ядро
    vec_cores_per_cube: int = 2          # NUM_AIV_TO_AIC_RATIO
    c_dtype_size: int = 4


@dataclass
class DataInfoBlock:
    matrix_name: str
    cache_type: str
    cache_index: tuple[int, int]
    shape: tuple[int, int]
    dtype_size: int

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1] * self.dtype_size

    @property
    def name(self) -> str:
        return self.get_name(self.matrix_name, self.cache_type, self.cache_index)

    @staticmethod
    def get_name(matrix_name: str, cache_type: str, cache_index: tuple[int, int]) -> str:
        return f"{matrix_name}::{cache_type}:{cache_index}"

    def with_cache_type(self, cache_type: str) -> "DataInfoBlock":
        return DataInfoBlock(self.matrix_name, cache_type, self.cache_index,
                             self.shape, self.dtype_size)

    def split_block(self, size_tiles: tuple[int, int], cache_type: str) -> list["DataInfoBlock"]:
        splitted_blocks = []
        num_blocks = tuple(self._ceil_div(self.shape[dim], size_tiles[dim]) for dim in range(2))
        comb_idxs = itertools.product(range(num_blocks[0]), range(num_blocks[1]))
        for block_index in comb_idxs:
            block_shape = tuple(
                min(size_tiles[dim], self.shape[dim] - block_index[dim] * size_tiles[dim])
                for dim in range(2)
            )
            splitted_blocks.append(DataInfoBlock(
                self.matrix_name, cache_type, block_index, block_shape, self.dtype_size))
        return splitted_blocks

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

