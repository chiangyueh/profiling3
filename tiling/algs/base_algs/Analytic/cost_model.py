from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .parameters import (
    ACCUMULATOR_BYTES, AL1_FIXED_FIELDS, ATOMIC_READ_BYTES_PER_CYCLE,
    ATOMIC_WRITE_BYTES_PER_CYCLE, BASE_FIXED_FIELDS, BASE_K_ALIGNMENT,
    BASE_K_BYTES, BASE_L0_BUFFERS, BASE_L0_TILES,
    BL1_FIXED_FIELDS, BL1_FULL_LOAD_EXTRA_TILE, BL1_SINGLE_CORE_M_TILES,
    CUBE_MACS_PER_EVENT, DEFAULT_MAX_CORES, DETERMINISTIC_FIXED_FIELDS,
    DETERMINISTIC_M_EDGE_BLOCK, DETERMINISTIC_OVERLAP_TILE,
    DETERMINISTIC_PROTOCOLS, DETERMINISTIC_SERIAL_LAYOUTS,
    DETERMINISTIC_STEP_MN, FIELDS, FP32_NT_BASE_K_ALIGNMENT,
    FRACTAL_MN, FULL_LOAD_BASE_TILES, HARDWARE, INPUT_BYTES,
    K_SATURATION_CAPACITY_MN, K_SATURATION_MN, L0C_SINGLE_BUFFER_CAPACITY_MULTIPLIER,
    L1_ALLOCATION_QUANTUM_BYTES, L1_OPERAND_CAPACITY_SHARE,
    L1_PIPELINE_DEPTH_MULTIPLIER, L1_STATE_LIMIT, L2_20_CORE_MAX_CONFLICT,
    L2_20_CORE_MIN_CONFLICT, L2_ALL_M_BLOCKS, L2_GENERAL_MAX_CONFLICT,
    L2_GENERAL_MIN_CONFLICT, LOW_K_BASE_K_BYTES, LOW_K_LIMIT_BY_INPUT_BYTES,
    MAX_GENERATED_CANDIDATES, MAX_VECTOR_CORES,
    MTE1_A_BYTES_PER_EVENT, MTE1_B_BYTES_PER_EVENT, ONE_AIC_WAVE_TILE,
    OUTPUT_BYTES, OUTPUT_BYTES_PER_CYCLE, PMU_FALLBACK, PMU_SERVICE,
    REDUCTION_BYTES_PER_CYCLE, REDUCTION_ELEMENTS_PER_CYCLE,
    REFERENCE_L2_BYTES, REFERENCE_WORKING_SET_BYTES,
    SINGLE_SPLIT_BALANCED_PROTOCOL, SINGLE_SPLIT_FIXED_FIELDS,
    SINGLE_SPLIT_PROTOCOLS,
    SPLIT_BALANCE_ALIGNMENT, SPLIT_BALANCE_MIN_M, SPLIT_BASE_K_BYTES,
    SPLIT_BASE_MN, SPLIT_LOW_OCCUPANCY_CORES,
    SPLIT_STEP_K, TILING_ENABLE, TILING_FIXPIPE_DIVISOR, TILING_ROUTE_RADIX,
    TRANSACTION_BYTES, VALID_BUFFER_MODES, VECTOR_CORES_PER_AIC,
)
INDEX = {name: index for index, name in enumerate(FIELDS)}


class Route(str, Enum):
    BASE = "BASE"
    SINGLE_CORE_SPLIT_K = "SINGLE_CORE_SPLIT_K"
    DETERMINISTIC_SPLIT_K = "DETERMINISTIC_SPLIT_K"
    AL1_FULL_LOAD = "AL1_FULL_LOAD"
    BL1_FULL_LOAD = "BL1_FULL_LOAD"
    BL1_FULL_LOAD_FIXPIPE = "BL1_FULL_LOAD_FIXPIPE"
    BL1_FULL_LOAD_VEC_NZ2ND = "BL1_FULL_LOAD_VEC_NZ2ND"


@dataclass(frozen=True)
class Workload:
    m: int
    n: int
    k: int
    dtype: str
    trans_a: bool
    trans_b: bool
    max_cores: int = DEFAULT_MAX_CORES

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k, self.max_cores) <= 0:
            raise ValueError("M, N, K and max_cores must be positive")
        if self.dtype not in INPUT_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")


@dataclass(frozen=True)
class Hardware:
    aic_cores: int = HARDWARE["aic_cores"]
    l0a_bytes: int = HARDWARE["l0a_bytes"]
    l0b_bytes: int = HARDWARE["l0b_bytes"]
    l0c_bytes: int = HARDWARE["l0c_bytes"]
    l1_bytes: int = HARDWARE["l1_bytes"]
    ub_bytes: int = HARDWARE["ub_bytes"]
    l2_bytes: int = HARDWARE["l2_bytes"]
    l2_read_bytes_per_cycle_per_core: float = HARDWARE["l2_read_bytes_per_cycle_per_core"]
    hbm_bytes_per_cycle_per_core: float = HARDWARE["hbm_bytes_per_cycle_per_core"]

    @property
    def effective_l1_bytes(self) -> int:
        return ceil_div(self.l1_bytes, L1_ALLOCATION_QUANTUM_BYTES) * L1_ALLOCATION_QUANTUM_BYTES

    def l0c_capacity(self, db_l0c: int) -> int:
        return self.l0c_bytes * (L0C_SINGLE_BUFFER_CAPACITY_MULTIPLIER if db_l0c == 1 else 1)


ASCEND_910B3 = Hardware()


@dataclass(frozen=True)
class Schedule:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(FIELDS):
            raise ValueError("Candidate23 must contain exactly 23 integers")

    @classmethod
    def make(cls, **values: int) -> "Schedule":
        missing = set(FIELDS).difference(values)
        if missing:
            raise ValueError(f"missing Candidate23 fields: {sorted(missing)}")
        return cls(tuple(int(values[name]) for name in FIELDS))

    def __getitem__(self, name: str) -> int:
        return self.values[INDEX[name]]

    def as_dict(self) -> dict[str, int]:
        return dict(zip(FIELDS, self.values))

    def replace(self, **updates: int) -> "Schedule":
        values = self.as_dict()
        values.update(updates)
        return Schedule.make(**values)

    def signature(self) -> str:
        return ":".join(map(str, self.values))


@dataclass(frozen=True)
class Candidate:
    schedule: Schedule
    route: Route
    reason: str


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def base_k_alignment(workload: Workload) -> int:
    return FP32_NT_BASE_K_ALIGNMENT if workload.dtype == "fp32" and not workload.trans_a and workload.trans_b else BASE_K_ALIGNMENT


def layout(workload: Workload) -> str:
    return ("T" if workload.trans_a else "N") + ("T" if workload.trans_b else "N")


def route_of(schedule: Schedule) -> Route:
    enable = schedule["tilingEnable"]
    if enable % TILING_ROUTE_RADIX == TILING_ENABLE["DETERMINISTIC_SPLIT_K"]:
        return Route.DETERMINISTIC_SPLIT_K
    if enable % TILING_ROUTE_RADIX == TILING_ENABLE["SINGLE_CORE_SPLIT_K"]:
        return Route.SINGLE_CORE_SPLIT_K
    full = (enable // TILING_ROUTE_RADIX) % TILING_ROUTE_RADIX
    fix = (enable // TILING_FIXPIPE_DIVISOR) % TILING_ROUTE_RADIX
    if full == 1:
        return Route.AL1_FULL_LOAD
    if full == 2 and fix == 1:
        return Route.BL1_FULL_LOAD_FIXPIPE
    if full == 2 and fix == 2:
        return Route.BL1_FULL_LOAD_VEC_NZ2ND
    if full == 2:
        return Route.BL1_FULL_LOAD
    return Route.BASE


def validate(workload: Workload, schedule: Schedule, hardware: Hardware = ASCEND_910B3) -> tuple[str, ...]:
    """Check the common device capacities and the constructed route grammar."""

    errors: list[str] = []
    v = schedule
    if not 1 <= v["usedCoreNum"] <= min(workload.max_cores, hardware.aic_cores):
        errors.append("used_core_range")
    positive = (
        "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
        "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
        "stepN", "stepKa", "stepKb", "dbL0A", "dbL0B", "dbL0C",
        "l2MTileCnt", "l2NTileCnt",
    )
    if any(v[name] <= 0 for name in positive):
        errors.append("nonpositive_field")
    if v["baseM"] % FRACTAL_MN or v["baseN"] % FRACTAL_MN or v["baseK"] % base_k_alignment(workload):
        errors.append("fractal_alignment")
    if v["singleCoreM"] < v["baseM"] or v["singleCoreN"] < v["baseN"]:
        errors.append("single_core_extent")
    if v["dbL0A"] not in VALID_BUFFER_MODES or v["dbL0B"] not in VALID_BUFFER_MODES or v["dbL0C"] not in VALID_BUFFER_MODES:
        errors.append("buffer_mode")
    ib = INPUT_BYTES[workload.dtype]
    if v["baseM"] * v["baseK"] * ib * v["dbL0A"] > hardware.l0a_bytes:
        errors.append("l0a_capacity")
    if v["baseN"] * v["baseK"] * ib * v["dbL0B"] > hardware.l0b_bytes:
        errors.append("l0b_capacity")
    if v["baseM"] * v["baseN"] * ACCUMULATOR_BYTES * v["dbL0C"] > hardware.l0c_capacity(v["dbL0C"]):
        errors.append("l0c_capacity")
    l1 = (v["depthA1"] * v["baseM"] * v["baseK"] + v["depthB1"] * v["baseN"] * v["baseK"]) * ib
    if l1 > hardware.effective_l1_bytes:
        errors.append("l1_capacity")
    if v["depthA1"] < v["stepM"] * v["stepKa"] or v["depthB1"] < v["stepN"] * v["stepKb"]:
        errors.append("l1_pipeline_depth")
    if v["l2MTileBlock"] == 0 or v["l2NTileBlock"] == 0:
        if not (v["l2MTileBlock"] == v["l2NTileBlock"] == 0 and v["l2MTileCnt"] == v["l2NTileCnt"] == 1):
            errors.append("l2_zero_block")
    else:
        mt = ceil_div(workload.m, v["singleCoreM"])
        nt = ceil_div(workload.n, v["singleCoreN"])
        if v["l2MTileCnt"] != ceil_div(mt, v["l2MTileBlock"]) or v["l2NTileCnt"] != ceil_div(nt, v["l2NTileBlock"]):
            errors.append("l2_group_coverage")
    route = route_of(v)
    if route == Route.BASE and (v["tilingEnable"] != TILING_ENABLE["BASE"] or v["singleCoreK"] < workload.k):
        errors.append("base_grammar")
    if route == Route.SINGLE_CORE_SPLIT_K and v["tilingEnable"] != TILING_ENABLE["SINGLE_CORE_SPLIT_K"]:
        errors.append("single_split_grammar")
    if route == Route.DETERMINISTIC_SPLIT_K:
        if v["tilingEnable"] != TILING_ENABLE["DETERMINISTIC_SPLIT_K"] or (v["stepM"], v["stepN"]) not in DETERMINISTIC_STEP_MN:
            errors.append("deterministic_grammar")
    return tuple(errors)


def _matrix_bytes(m: int, k: int, n: int, ib: int, ob: int) -> int:
    return (m * k + k * n) * ib + m * n * ob


@dataclass(frozen=True)
class _SplitAxes:
    outer_base: int
    inner_base: int
    outer_value: int
    inner_value: int
    outer_bytes: int
    inner_bytes: int
    max_conflict: int
    min_conflict: int


def _split_axes(workload: Workload, schedule: Schedule, cores: int) -> _SplitAxes:
    bm, bn = schedule["baseM"], schedule["baseN"]
    outer_base, inner_base = bm, bn
    outer_value, inner_value = workload.m, workload.n
    if bn >= bm:
        outer_base, inner_base = bn, bm
        outer_value, inner_value = workload.n, workload.m
    max_conflict = min(cores, L2_GENERAL_MAX_CONFLICT)
    min_conflict = min(cores, L2_GENERAL_MIN_CONFLICT)
    if cores == HARDWARE["aic_cores"]:
        max_conflict, min_conflict = L2_20_CORE_MAX_CONFLICT, L2_20_CORE_MIN_CONFLICT
    ib = INPUT_BYTES[workload.dtype]
    return _SplitAxes(outer_base, inner_base, outer_value, inner_value, ib, ib, max_conflict, min_conflict)


def _search_l2_tile(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
    outer_original: int,
    inner_original: int,
    inner_bad: bool,
    limit: float,
) -> tuple[int, ...] | None:
    cores = min(workload.max_cores, hardware.aic_cores)
    axes = _split_axes(workload, schedule, cores)
    inner_max_conflict = axes.min_conflict if inner_bad else axes.max_conflict
    outer_min_use = max(cores // axes.max_conflict, 1)
    inner_min_use = max(cores // inner_max_conflict, 1)
    selected = None
    for outer_use in range(cores, outer_min_use - 1, -1):
        for inner_use in range(cores, inner_min_use - 1, -1):
            outer_tile = max(outer_original // (axes.outer_base * outer_use), 1)
            inner_tile = max(inner_original // (axes.inner_base * inner_use), 1)
            outer_split = align_up(ceil_div(axes.outer_value, outer_tile), axes.outer_base)
            inner_split = align_up(ceil_div(axes.inner_value, inner_tile), axes.inner_base)
            size = _matrix_bytes(outer_split, workload.k, inner_split, axes.outer_bytes, OUTPUT_BYTES[workload.dtype])
            if size > limit:
                continue
            outer_tail = (axes.outer_value + outer_split - 1) % outer_split + 1
            inner_tail = (axes.inner_value + inner_split - 1) % inner_split + 1
            outer_tail_count = ceil_div(outer_tail, axes.outer_base)
            inner_tail_count = ceil_div(inner_tail, axes.inner_base)
            if outer_tail_count * axes.max_conflict < cores or inner_tail_count * inner_max_conflict < cores:
                continue
            outer_conflict = ceil_div(cores, outer_tail_count)
            inner_conflict = ceil_div(cores, inner_tail_count)
            if selected is None or (selected[4] >= outer_conflict and selected[5] >= inner_conflict):
                selected = (outer_tile, inner_tile, outer_split, inner_split, outer_conflict, inner_conflict)
    return selected


def _capacity_l2_group(workload: Workload, schedule: Schedule, hardware: Hardware, all_default: bool = True) -> tuple[int, int, int, int, int]:
    """CANN source L2 capacity/conflict split, including its 4-by-AIC/4 ALL state."""

    ib, ob = INPUT_BYTES[workload.dtype], OUTPUT_BYTES[workload.dtype]
    cores = min(workload.max_cores, hardware.aic_cores)
    limit = REFERENCE_WORKING_SET_BYTES * hardware.l2_bytes / REFERENCE_L2_BYTES
    total = _matrix_bytes(workload.m, workload.k, workload.n, ib, ob)
    m_split, n_split = workload.m, workload.n
    capacity_split = False
    if total > limit:
        if schedule["baseN"] >= schedule["baseM"]:
            found = _search_l2_tile(workload, schedule, hardware, workload.n, workload.m, workload.trans_a, limit)
            if found is not None:
                _, _, n_split, m_split, _, _ = found
                capacity_split = True
        else:
            found = _search_l2_tile(workload, schedule, hardware, workload.m, workload.n, not workload.trans_b, limit)
            if found is not None:
                _, _, m_split, n_split, _, _ = found
                capacity_split = True
    mb = ceil_div(m_split, schedule["baseM"])
    nb = ceil_div(n_split, schedule["baseN"])
    mc = ceil_div(workload.m, mb * schedule["baseM"])
    nc = ceil_div(workload.n, nb * schedule["baseN"])
    if all_default and not capacity_split:
        mb = min(L2_ALL_M_BLOCKS, ceil_div(workload.m, schedule["baseM"]))
        nb = min(max(1, cores // L2_ALL_M_BLOCKS), ceil_div(workload.n, schedule["baseN"]))
        mc = ceil_div(workload.m, mb * schedule["baseM"])
        nc = ceil_div(workload.n, nb * schedule["baseN"])
    return mc, nc, mb, nb, 0


def _tile_ok(
    workload: Workload, hardware: Hardware, bm: int, bn: int, bk: int,
    dba: int = BASE_L0_BUFFERS[0], dbb: int = BASE_L0_BUFFERS[1],
    dbc: int = BASE_L0_BUFFERS[2],
) -> tuple[int, ...] | None:
    ib = INPUT_BYTES[workload.dtype]
    ka = base_k_alignment(workload)
    bm = max(FRACTAL_MN, min(align_up(workload.m, FRACTAL_MN), bm // FRACTAL_MN * FRACTAL_MN))
    bn = max(FRACTAL_MN, min(align_up(workload.n, FRACTAL_MN), bn // FRACTAL_MN * FRACTAL_MN))
    bk = max(ka, min(align_up(workload.k, ka), bk // ka * ka))
    if bm * bk * ib * dba > hardware.l0a_bytes or bn * bk * ib * dbb > hardware.l0b_bytes:
        return None
    if bm * bn * ACCUMULATOR_BYTES * dbc > hardware.l0c_capacity(dbc):
        return None
    return bm, bn, bk, dba, dbb, dbc


def _l1_states(workload: Workload, hardware: Hardware, tile: tuple[int, ...]) -> list[tuple[int, ...]]:
    bm, bn, bk, dba, dbb, dbc = tile
    ib = INPUT_BYTES[workload.dtype]
    a, b = bm * bk * ib, bn * bk * ib
    sa = max(1, (hardware.effective_l1_bytes // L1_OPERAND_CAPACITY_SHARE) // max(1, L1_PIPELINE_DEPTH_MULTIPLIER * a))
    sb = max(1, (hardware.effective_l1_bytes // L1_OPERAND_CAPACITY_SHARE) // max(1, L1_PIPELINE_DEPTH_MULTIPLIER * b))
    if sa >= sb:
        sa = max(sb, sa // sb * sb)
    else:
        sb = max(sa, sb // sa * sa)
    states: list[tuple[int, ...]] = []

    def add(step_a: int, step_b: int) -> None:
        if step_a >= step_b:
            step_a = max(step_b, step_a // step_b * step_b)
        else:
            step_b = max(step_a, step_b // step_a * step_a)
        state = (
            L1_PIPELINE_DEPTH_MULTIPLIER * step_a,
            L1_PIPELINE_DEPTH_MULTIPLIER * step_b,
            step_a, step_b, dba, dbb, dbc,
        )
        if state[0] * a + state[1] * b <= hardware.effective_l1_bytes and state not in states:
            states.append(state)

    add(sa, sb)
    add(min(1 << (sa.bit_length() - 1), 1 << (sb.bit_length() - 1)), min(1 << (sa.bit_length() - 1), 1 << (sb.bit_length() - 1)))
    return states[:L1_STATE_LIMIT]


def _base_tiles(workload: Workload, hardware: Hardware) -> list[tuple[tuple[int, ...], str]]:
    ib = INPUT_BYTES[workload.dtype]
    bk = BASE_K_BYTES // ib
    raw = [
        (BASE_L0_TILES[0][0], BASE_L0_TILES[0][1], bk, "l0_128x256"),
        (BASE_L0_TILES[1][0], BASE_L0_TILES[1][1], bk, "l0_256x128"),
        (BASE_L0_TILES[2][0], BASE_L0_TILES[2][1], bk, "l0_128x128"),
    ]
    max_bk = min(
        align_up(workload.k, base_k_alignment(workload)),
        hardware.l0a_bytes // (K_SATURATION_CAPACITY_MN * ib),
        hardware.l0b_bytes // (K_SATURATION_CAPACITY_MN * ib),
    )
    max_bk = max(base_k_alignment(workload), max_bk // base_k_alignment(workload) * base_k_alignment(workload))
    raw.append((K_SATURATION_MN, K_SATURATION_MN, max_bk, "k_saturation"))
    if workload.k <= LOW_K_LIMIT_BY_INPUT_BYTES[ib]:
        raw.append((*BASE_L0_TILES[2], min(max_bk, LOW_K_BASE_K_BYTES // ib), "low_k_square"))

    cores = min(workload.max_cores, hardware.aic_cores)
    m_tasks = ceil_div(workload.m, ONE_AIC_WAVE_TILE)
    n_wave = max(1, ceil_div(cores, m_tasks))
    raw.append((ONE_AIC_WAVE_TILE, align_up(ceil_div(workload.n, n_wave), FRACTAL_MN), bk, "one_aic_wave_n"))
    n_tasks = ceil_div(workload.n, ONE_AIC_WAVE_TILE)
    m_wave = max(1, ceil_div(cores, n_tasks))
    raw.append((align_up(ceil_div(workload.m, m_wave), FRACTAL_MN), ONE_AIC_WAVE_TILE, bk, "one_aic_wave_m"))

    result: list[tuple[tuple[int, ...], str]] = []
    seen: set[tuple[int, ...]] = set()
    for bm, bn, kb, reason in raw:
        tile = _tile_ok(workload, hardware, bm, bn, kb)
        if tile is not None and tile not in seen:
            seen.add(tile)
            result.append((tile, reason))
    return result


def _base_candidates(workload: Workload, hardware: Hardware) -> Iterable[Candidate]:
    for tile, tile_reason in _base_tiles(workload, hardware):
        bm, bn, bk, dba, dbb, dbc = tile
        used = min(workload.max_cores, hardware.aic_cores)
        for depth_a, depth_b, step_a, step_b, ba, bb, bc in _l1_states(workload, hardware, tile):
            proto = Schedule.make(
                usedCoreNum=max(1, used), singleCoreM=bm, singleCoreN=bn, singleCoreK=workload.k,
                baseM=bm, baseN=bn, baseK=bk, depthA1=depth_a, depthB1=depth_b,
                stepKa=step_a, stepKb=step_b, dbL0A=ba, dbL0B=bb, dbL0C=bc,
                **BASE_FIXED_FIELDS,
            )
            source_all = _capacity_l2_group(workload, proto, hardware, True)
            source_full = _capacity_l2_group(workload, proto, hardware, False)
            for l2, l2_reason in ((source_all, "source_all_l2_group"), (source_full, "source_capacity_l2_group")):
                schedule = proto.replace(l2MTileCnt=l2[0], l2NTileCnt=l2[1], l2MTileBlock=l2[2], l2NTileBlock=l2[3], l2IterateOrder=l2[4])
                if not validate(workload, schedule, hardware):
                    yield Candidate(schedule, Route.BASE, f"{tile_reason}+{l2_reason}")


def _simple_l2(workload: Workload, sm: int, sn: int, used: int, order: int = 0) -> tuple[int, ...]:
    mt, nt = ceil_div(workload.m, sm), ceil_div(workload.n, sn)
    choices = []
    for mb in range(1, min(mt, used) + 1):
        center = min(nt, max(1, round(used / mb)))
        for nb in {max(1, center - 1), center, min(nt, center + 1)}:
            tasks = mb * nb
            choices.append(((abs(tasks - used), ceil_div(mt, mb) * ceil_div(nt, nb), -tasks), mb, nb))
    _, mb, nb = min(choices)
    return ceil_div(mt, mb), ceil_div(nt, nb), mb, nb, order


def _single_split_candidates(workload: Workload, hardware: Hardware) -> Iterable[Candidate]:
    ib = INPUT_BYTES[workload.dtype]
    bk = SPLIT_BASE_K_BYTES // ib
    single_k = SPLIT_STEP_K * bk
    if single_k >= workload.k:
        return
    core_cap = min(workload.max_cores, hardware.aic_cores)
    n_tiles = ceil_div(workload.n, SPLIT_BASE_MN)
    m_parts = max(1, core_cap // min(n_tiles, core_cap))
    balanced_m = max(SPLIT_BALANCE_MIN_M, align_up(ceil_div(workload.m, m_parts), SPLIT_BALANCE_ALIGNMENT))
    states = SINGLE_SPLIT_PROTOCOLS + ((balanced_m, SPLIT_BASE_MN, *SINGLE_SPLIT_BALANCED_PROTOCOL),)
    for sm, sn, step_m, step_n, depth_a, depth_b, order, reason in states:
        used = min(core_cap, ceil_div(workload.m, sm) * ceil_div(workload.n, sn))
        l2 = _simple_l2(workload, sm, sn, max(1, used))
        schedule = Schedule.make(
            usedCoreNum=max(1, used), singleCoreM=sm, singleCoreN=sn, singleCoreK=single_k,
            baseM=SPLIT_BASE_MN, baseN=SPLIT_BASE_MN, baseK=bk, depthA1=depth_a, depthB1=depth_b,
            stepM=step_m, stepN=step_n, iterateOrder=order, stepKa=SPLIT_STEP_K, stepKb=SPLIT_STEP_K,
            l2MTileCnt=l2[0], l2NTileCnt=l2[1], l2MTileBlock=l2[2], l2NTileBlock=l2[3],
            **SINGLE_SPLIT_FIXED_FIELDS,
        )
        if not validate(workload, schedule, hardware):
            yield Candidate(schedule, Route.SINGLE_CORE_SPLIT_K, reason)


def _deterministic_core_states(workload: Workload, hardware: Hardware, chunks: int) -> tuple[int, ...]:
    maximum = min(workload.max_cores, hardware.aic_cores, chunks)
    return tuple(sorted({maximum, min(maximum, SPLIT_LOW_OCCUPANCY_CORES)}))


def _deterministic_candidates(workload: Workload, hardware: Hardware) -> Iterable[Candidate]:
    ib = INPUT_BYTES[workload.dtype]
    bk = SPLIT_BASE_K_BYTES // ib
    single_k = SPLIT_STEP_K * bk
    chunks = ceil_div(workload.k, single_k)
    if chunks < 2:
        return
    for used in _deterministic_core_states(workload, hardware, chunks):
        options: list[tuple[tuple[int, ...], Schedule, str]] = []
        protocols = (
            (DETERMINISTIC_PROTOCOLS[0][0], max(SPLIT_BASE_MN, align_up(workload.n, FRACTAL_MN)), *DETERMINISTIC_PROTOCOLS[0][2:]),
            (max(SPLIT_BASE_MN, align_up(workload.m, FRACTAL_MN)), DETERMINISTIC_PROTOCOLS[1][1], *DETERMINISTIC_PROTOCOLS[1][2:]),
        )
        for sm, sn, step_m, step_n, depth_a, depth_b, order, l2_order, name in protocols:
            schedule = Schedule.make(
                usedCoreNum=used, singleCoreM=sm, singleCoreN=sn, singleCoreK=single_k,
                baseM=SPLIT_BASE_MN, baseN=SPLIT_BASE_MN, baseK=bk, depthA1=depth_a, depthB1=depth_b,
                stepM=step_m, stepN=step_n, iterateOrder=order, stepKa=SPLIT_STEP_K, stepKb=SPLIT_STEP_K,
                l2MTileBlock=max(1, ceil_div(workload.m, sm)),
                l2NTileBlock=max(1, ceil_div(workload.n, sn)),
                l2IterateOrder=l2_order, **DETERMINISTIC_FIXED_FIELDS,
            )
            if validate(workload, schedule, hardware):
                continue
            m_segments, n_segments = ceil_div(workload.m, sm), ceil_div(workload.n, sn)
            repeated = ((n_segments - 1) * workload.m * workload.k * ib if order == 0 else (m_segments - 1) * workload.n * workload.k * ib)
            key = (repeated, order, schedule.values)
            options.append((key, schedule, name))
        if options:
            _, schedule, name = min(options, key=lambda item: item[0])
            yield Candidate(schedule, Route.DETERMINISTIC_SPLIT_K, f"{name}+k_wave_{used}")


def _full_load_candidates(workload: Workload, hardware: Hardware) -> Iterable[Candidate]:
    ib = INPUT_BYTES[workload.dtype]
    ka = base_k_alignment(workload)
    bk = BASE_K_BYTES // ib
    for bm, bn in FULL_LOAD_BASE_TILES:
        if _tile_ok(workload, hardware, bm, bn, bk) is None:
            continue
        resident_a = align_up(workload.m, FRACTAL_MN) * align_up(workload.k, ka) * ib
        step_k = ceil_div(workload.k, bk)
        if resident_a + bn * bk * ib <= hardware.effective_l1_bytes:
            sn = max(bn, align_up(ceil_div(workload.n, min(workload.max_cores, hardware.aic_cores)), FRACTAL_MN))
            nt = ceil_div(workload.n, sn)
            schedule = Schedule.make(
                usedCoreNum=max(1, min(workload.max_cores, hardware.aic_cores, nt)),
                singleCoreM=align_up(workload.m, FRACTAL_MN), singleCoreN=sn, singleCoreK=workload.k,
                baseM=bm, baseN=bn, baseK=bk, depthA1=step_k, stepKa=step_k,
                l2NTileBlock=nt, **AL1_FIXED_FIELDS,
            )
            if not validate(workload, schedule, hardware):
                yield Candidate(schedule, Route.AL1_FULL_LOAD, "resident_a_in_l1")
            break

    resident_b = align_up(workload.k, ka) * align_up(workload.n, max(FRACTAL_MN, K_SATURATION_CAPACITY_MN // ib)) * ib
    for bm, bn in FULL_LOAD_BASE_TILES + (BL1_FULL_LOAD_EXTRA_TILE,):
        if _tile_ok(workload, hardware, bm, bn, bk) is None:
            continue
        step_n, step_k = ceil_div(workload.n, bn), ceil_div(workload.k, bk)
        if resident_b + bm * bk * (BL1_SINGLE_CORE_M_TILES * step_k) * ib > hardware.effective_l1_bytes:
            continue
        sm = BL1_SINGLE_CORE_M_TILES * bm
        used = min(workload.max_cores, hardware.aic_cores, ceil_div(workload.m, sm))
        schedule = Schedule.make(
            usedCoreNum=max(1, used), singleCoreM=sm, singleCoreN=workload.n, singleCoreK=workload.k,
            baseM=bm, baseN=bn, baseK=bk, depthA1=BL1_SINGLE_CORE_M_TILES * step_k, depthB1=step_n * step_k,
            stepN=step_n, stepKa=step_k, stepKb=step_k,
            l2MTileBlock=ceil_div(workload.m, sm), **BL1_FIXED_FIELDS,
        )
        if not validate(workload, schedule, hardware):
            yield Candidate(schedule, Route.BL1_FULL_LOAD, "resident_b_in_l1")
        break


def generate(workload: Workload, hardware: Hardware = ASCEND_910B3) -> tuple[Candidate, ...]:
    candidates = list(_base_candidates(workload, hardware))
    candidates.extend(_single_split_candidates(workload, hardware))
    candidates.extend(_deterministic_candidates(workload, hardware))
    candidates.extend(_full_load_candidates(workload, hardware))
    unique = {candidate.schedule.values: candidate for candidate in candidates}
    if not unique:
        raise RuntimeError("no legal Candidate23")
    if len(unique) > MAX_GENERATED_CANDIDATES:
        raise RuntimeError(f"bounded candidate invariant failed: {len(unique)} > {MAX_GENERATED_CANDIDATES}")
    return tuple(unique.values())


def _service(pipe: str, key: str) -> tuple[float, float, float]:
    return PMU_SERVICE[pipe].get(key, PMU_FALLBACK[pipe])


def _row_transactions(rows: int, row_bytes: int) -> int:
    return rows * align_up(row_bytes, TRANSACTION_BYTES)


def _overlap(cube: float, mte1: float, mte2: float, output: float, schedule: Schedule, route: Route, specialized_det: bool) -> float:
    if route == Route.DETERMINISTIC_SPLIT_K and specialized_det:
        return max(cube, mte1, mte2, output)
    service = max(cube, mte1) if schedule["dbL0A"] == schedule["dbL0B"] == 2 else cube + mte1
    a_buffers = schedule["depthA1"] // max(1, schedule["stepM"] * schedule["stepKa"])
    b_buffers = schedule["depthB1"] // max(1, schedule["stepN"] * schedule["stepKb"])
    service = max(service, mte2) if a_buffers >= 2 and b_buffers >= 2 else service + mte2
    if schedule["dbL0C"] == 2 and route not in (Route.BL1_FULL_LOAD_FIXPIPE, Route.BL1_FULL_LOAD_VEC_NZ2ND):
        return max(service, output)
    return service + output


def cost(workload: Workload, schedule: Schedule, hardware: Hardware = ASCEND_910B3) -> dict[str, float]:
    print(f"WORKLOAD: {workload}, SCHEDULE: {schedule}")
    route = route_of(schedule)
    ib, ob = INPUT_BYTES[workload.dtype], OUTPUT_BYTES[workload.dtype]
    ka = base_k_alignment(workload)
    pm = align_up(workload.m, FRACTAL_MN)
    pn = align_up(workload.n, FRACTAL_MN)
    pk = align_up(workload.k, ka)
    sm, sn, sk = schedule["singleCoreM"], schedule["singleCoreN"], schedule["singleCoreK"]
    cores = schedule["usedCoreNum"]
    output_tasks = ceil_div(workload.m, sm) * ceil_div(workload.n, sn)

    if route == Route.DETERMINISTIC_SPLIT_K:
        chunks = ceil_div(workload.k, sk)
        waves = ceil_div(chunks, cores)
        critical_k = min(pk, waves * sk)
        mac = pm * pn * critical_k
        requests = (pm + pn) * critical_k * ib
        output_bytes = pm * pn * ACCUMULATOR_BYTES
        transaction_multiplier, transaction_k = 1, critical_k
        idle_k_slots = waves * cores - chunks
    else:
        waves = ceil_div(output_tasks, cores)
        asm, asn = align_up(sm, FRACTAL_MN), align_up(sn, FRACTAL_MN)
        mac = waves * asm * asn * pk
        requests = waves * (asm + asn) * pk * ib
        output_bytes = waves * asm * asn * ob
        if route == Route.AL1_FULL_LOAD:
            requests = (pm * pk + waves * asn * pk) * ib
        elif route in (Route.BL1_FULL_LOAD, Route.BL1_FULL_LOAD_FIXPIPE, Route.BL1_FULL_LOAD_VEC_NZ2ND):
            requests = (pk * pn + waves * asm * pk) * ib
        transaction_multiplier, transaction_k = waves, pk
        idle_k_slots = 0

    key = f"{workload.dtype}:{layout(workload)}"
    cube_events = mac / CUBE_MACS_PER_EVENT[workload.dtype]
    cube = tuple(cube_events * factor for factor in _service("cube", key))
    mte1_events = (
        mac / schedule["baseN"] * ib / MTE1_B_BYTES_PER_EVENT
        + mac / schedule["baseM"] * ib / MTE1_A_BYTES_PER_EVENT
    )
    mte1 = tuple(mte1_events * factor for factor in _service("mte1", key))
    if workload.trans_a:
        a_bytes = _row_transactions(transaction_k, align_up(sm, FRACTAL_MN) * ib)
    else:
        a_bytes = _row_transactions(align_up(sm, FRACTAL_MN), transaction_k * ib)
    if workload.trans_b:
        b_bytes = _row_transactions(align_up(sn, FRACTAL_MN), transaction_k * ib)
    else:
        b_bytes = _row_transactions(transaction_k, align_up(sn, FRACTAL_MN) * ib)
    transactions = transaction_multiplier * (a_bytes + b_bytes) / TRANSACTION_BYTES
    request_floor = requests / hardware.l2_read_bytes_per_cycle_per_core
    mte2 = tuple(max(request_floor, transactions * factor) for factor in _service("mte2", key))
    all_hbm_mte2 = max(mte2[2], requests / hardware.hbm_bytes_per_cycle_per_core)
    output_service = output_bytes / OUTPUT_BYTES_PER_CYCLE

    suffix = 0.0
    atomic_events = 0.0
    sync_events = 0.0
    reduction = 0.0
    if route == Route.SINGLE_CORE_SPLIT_K:
        atomic_events = float(max(0, ceil_div(workload.k, sk) - 1))
        suffix = atomic_events * output_bytes * (
            1.0 / ATOMIC_READ_BYTES_PER_CYCLE + 1.0 / ATOMIC_WRITE_BYTES_PER_CYCLE
        )
    elif route == Route.DETERMINISTIC_SPLIT_K:
        vector_cores = min(MAX_VECTOR_CORES, VECTOR_CORES_PER_AIC * cores)
        reduction = (
            VECTOR_CORES_PER_AIC * cores * workload.m * workload.n * ACCUMULATOR_BYTES
            / (REDUCTION_BYTES_PER_CYCLE * vector_cores)
        )
        reduction += workload.m * workload.n * cores / (REDUCTION_ELEMENTS_PER_CYCLE * vector_cores)
        suffix = reduction
        sync_events = float(cores)

    complete_n = workload.n >= DETERMINISTIC_OVERLAP_TILE
    complete_m = (
        workload.m >= DETERMINISTIC_OVERLAP_TILE
        or align_up(workload.m, DETERMINISTIC_OVERLAP_TILE) - workload.m <= DETERMINISTIC_M_EDGE_BLOCK
    )
    specialized_det = (
        route == Route.DETERMINISTIC_SPLIT_K
        and complete_m
        and complete_n
        and layout(workload) not in DETERMINISTIC_SERIAL_LAYOUTS
    )
    lower = _overlap(cube[0], mte1[0], mte2[0], output_service, schedule, route, specialized_det) + suffix
    nominal = _overlap(cube[1], mte1[1], mte2[1], output_service, schedule, route, specialized_det) + suffix
    p90 = _overlap(cube[2], mte1[2], mte2[2], output_service, schedule, route, specialized_det) + suffix
    upper = _overlap(cube[2], mte1[2], all_hbm_mte2, output_service, schedule, route, specialized_det) + suffix
    l2_groups = max(1, schedule["l2MTileCnt"] * schedule["l2NTileCnt"])
    group_m = min(workload.m, max(1, schedule["l2MTileBlock"]) * sm)
    group_n = min(workload.n, max(1, schedule["l2NTileBlock"]) * sn)
    group_ws = _matrix_bytes(group_m, workload.k, group_n, ib, ob)
    return {
        "lower_cycles": lower, "nominal_cycles": nominal, "pmu_p90_cycles": p90,
        "upper_cycles": upper, "cube_cycles": cube[1], "mte1_cycles": mte1[1],
        "mte2_cycles": mte2[1], "request_bytes": float(requests),
        "output_cycles": output_service, "reduction_cycles": reduction,
        "atomic_events": atomic_events, "sync_events": sync_events,
        "idle_k_dispatch_slots": float(idle_k_slots), "output_waves": float(waves),
        "l2_groups": float(l2_groups), "group_working_set_bytes": float(group_ws),
    }


def rank_key(workload: Workload, candidate: Candidate, hardware: Hardware = ASCEND_910B3) -> tuple[object, ...]:
    c = cost(workload, candidate.schedule, hardware)
    schedule = candidate.schedule
    k_tiles = ceil_div(align_up(workload.k, base_k_alignment(workload)), schedule["baseK"])
    source_commands = (
        ceil_div(k_tiles, schedule["stepKa"])
        + ceil_div(k_tiles, schedule["stepKb"])
        + ceil_div(workload.m, schedule["singleCoreM"])
        * ceil_div(workload.n, schedule["singleCoreN"])
    )
    return (
        c["nominal_cycles"], c["pmu_p90_cycles"], c["upper_cycles"],
        c["sync_events"], c["atomic_events"], c["idle_k_dispatch_slots"],
        source_commands, candidate.schedule.values,
    )


def solve(workload: Workload, hardware: Hardware = ASCEND_910B3) -> tuple[Candidate, tuple[Candidate, ...]]:
    ranked = tuple(sorted(generate(workload, hardware), key=lambda item: rank_key(workload, item, hardware)))
    return ranked[0], ranked


def solve_candidate23(
    m: int, n: int, k: int, dtype: str, trans_a: bool, trans_b: bool,
    max_cores: int = DEFAULT_MAX_CORES,
) -> dict[str, object]:
    workload = Workload(int(m), int(n), int(k), str(dtype), bool(trans_a), bool(trans_b), int(max_cores))
    selected, ranked = solve(workload)
    print(ranked)
    return {
        "candidate23": selected.schedule.as_dict(),
        "tiling_signature": selected.schedule.signature(),
        "route": selected.route.value,
        "reason": selected.reason,
        "candidate_count": len(ranked),
        "legal": not validate(workload, selected.schedule),
        "cost": cost(workload, selected.schedule),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("dtype", choices=tuple(INPUT_BYTES))
    parser.add_argument("trans_a", type=int, choices=(0, 1))
    parser.add_argument("trans_b", type=int, choices=(0, 1))
    parser.add_argument("--max-cores", type=int, default=DEFAULT_MAX_CORES)
    args = parser.parse_args()
    print(json.dumps(solve_candidate23(args.m, args.n, args.k, args.dtype, bool(args.trans_a), bool(args.trans_b), args.max_cores), indent=2))
