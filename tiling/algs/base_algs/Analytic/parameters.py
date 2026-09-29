KIB = 1024
MIB = 1024 * KIB

FIELDS = (
    "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
    "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
    "stepN", "iterateOrder", "stepKa", "stepKb", "dbL0A",
    "dbL0B", "dbL0C", "l2MTileCnt", "l2NTileCnt", "l2MTileBlock",
    "l2NTileBlock", "l2IterateOrder", "tilingEnable",
)
INPUT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}
OUTPUT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}
ACCUMULATOR_BYTES = 4
FRACTAL_MN = 16
BASE_K_ALIGNMENT = 16
FP32_NT_BASE_K_ALIGNMENT = 8
VALID_BUFFER_MODES = (1, 2)
BASE_L0_BUFFERS = (2, 2, 1)
SPLIT_L0_BUFFERS = (2, 2, 2)
L1_OPERAND_CAPACITY_SHARE = 2
L1_PIPELINE_DEPTH_MULTIPLIER = 2
L1_STATE_LIMIT = 2


DEFAULT_MAX_CORES = 24
HARDWARE = {
    "aic_cores": 24,
    "l0a_bytes": 64 * KIB,
    "l0b_bytes": 64 * KIB,
    "l0c_bytes": 128 * KIB,
    "l1_bytes": 524_032,
    "ub_bytes": 196_352,
    "l2_bytes": 192 * MIB,
    "l2_read_bytes_per_cycle_per_core": 110.0,
    "hbm_bytes_per_cycle_per_core": 32.0,
}
L1_ALLOCATION_QUANTUM_BYTES = KIB
L0C_SINGLE_BUFFER_CAPACITY_MULTIPLIER = 2


TILING_ENABLE = {
    "BASE": 0,
    "SINGLE_CORE_SPLIT_K": 2,
    "DETERMINISTIC_SPLIT_K": 3,
    "AL1_FULL_LOAD": 10,
    "BL1_FULL_LOAD": 20,
}
TILING_ROUTE_RADIX = 10
TILING_FIXPIPE_DIVISOR = 1000
DETERMINISTIC_STEP_MN = ((3, 1), (1, 3))


BASE_K_BYTES = 128
BASE_L0_TILES = ((128, 256), (256, 128), (128, 128))
K_SATURATION_MN = 16
K_SATURATION_CAPACITY_MN = 32
LOW_K_LIMIT_BY_INPUT_BYTES = {2: 256, 4: 128}
LOW_K_BASE_K_BYTES = 256
ONE_AIC_WAVE_TILE = 128

SPLIT_BASE_K_BYTES = 256
SINGLE_SPLIT_PROTOCOLS = (
    (384, 128, 3, 1, 9, 6, 1, "split_m3"),
    (128, 384, 1, 3, 6, 9, 0, "split_n3"),
)
SINGLE_SPLIT_BALANCED_PROTOCOL = (3, 1, 9, 6, 1, "split_m_wave")
SPLIT_BALANCE_ALIGNMENT = 128
SPLIT_BALANCE_MIN_M = 384
SPLIT_BASE_MN = 128
SPLIT_STEP_K = 3
SPLIT_LOW_OCCUPANCY_CORES = 2

DETERMINISTIC_PROTOCOLS = (
    (384, None, 3, 1, 9, 6, 1, 0, "preload_a_mk"),
    (None, 384, 1, 3, 6, 9, 0, 1, "preload_b_nk"),
)
DETERMINISTIC_OVERLAP_TILE = 128
DETERMINISTIC_M_EDGE_BLOCK = 16
DETERMINISTIC_SERIAL_LAYOUTS = ("NT",)

FULL_LOAD_BASE_TILES = ((128, 256), (256, 128), (128, 128))
BL1_FULL_LOAD_EXTRA_TILE = (64, 64)
BL1_SINGLE_CORE_M_TILES = 2
MAX_GENERATED_CANDIDATES = 48

BASE_FIXED_FIELDS = {
    "stepM": 1, "stepN": 1, "iterateOrder": 0,
    "l2MTileCnt": 1, "l2NTileCnt": 1,
    "l2MTileBlock": 1, "l2NTileBlock": 1, "l2IterateOrder": 0,
    "tilingEnable": TILING_ENABLE["BASE"],
}
SINGLE_SPLIT_FIXED_FIELDS = {
    "dbL0A": SPLIT_L0_BUFFERS[0], "dbL0B": SPLIT_L0_BUFFERS[1], "dbL0C": SPLIT_L0_BUFFERS[2],
    "l2IterateOrder": 0,
    "tilingEnable": TILING_ENABLE["SINGLE_CORE_SPLIT_K"],
}
DETERMINISTIC_FIXED_FIELDS = {
    "dbL0A": SPLIT_L0_BUFFERS[0], "dbL0B": SPLIT_L0_BUFFERS[1], "dbL0C": SPLIT_L0_BUFFERS[2],
    "l2MTileCnt": 1, "l2NTileCnt": 1,
    "tilingEnable": TILING_ENABLE["DETERMINISTIC_SPLIT_K"],
}
AL1_FIXED_FIELDS = {
    "depthB1": 1, "stepM": 1, "stepN": 1, "iterateOrder": 0, "stepKb": 1,
    "dbL0A": 1, "dbL0B": 2, "dbL0C": 1,
    "l2MTileCnt": 1, "l2NTileCnt": 1,
    "l2MTileBlock": 1, "l2IterateOrder": 0,
    "tilingEnable": TILING_ENABLE["AL1_FULL_LOAD"],
}
BL1_FIXED_FIELDS = {
    "stepM": 1, "iterateOrder": 1,
    "dbL0A": 2, "dbL0B": 2, "dbL0C": 1,
    "l2MTileCnt": 1, "l2NTileCnt": 1,
    "l2NTileBlock": 1, "l2IterateOrder": 1,
    "tilingEnable": TILING_ENABLE["BL1_FULL_LOAD"],
}


REFERENCE_L2_BYTES = 192 * MIB
REFERENCE_WORKING_SET_BYTES = 100 * MIB
L2_GENERAL_MAX_CONFLICT = 6
L2_GENERAL_MIN_CONFLICT = 3
L2_20_CORE_MAX_CONFLICT = 5
L2_20_CORE_MIN_CONFLICT = 4
L2_ALL_M_BLOCKS = 4


TRANSACTION_BYTES = 128
CUBE_MACS_PER_EVENT = {"fp16": 4096.0, "bf16": 4096.0, "fp32": 2048.0}
MTE1_A_BYTES_PER_EVENT = 256.0
MTE1_B_BYTES_PER_EVENT = 512.0
OUTPUT_BYTES_PER_CYCLE = 64.0
ATOMIC_READ_BYTES_PER_CYCLE = 110.0
ATOMIC_WRITE_BYTES_PER_CYCLE = 86.0
VECTOR_CORES_PER_AIC = 2
MAX_VECTOR_CORES = 40
REDUCTION_BYTES_PER_CYCLE = 64.0
REDUCTION_ELEMENTS_PER_CYCLE = 128.0


PMU_SERVICE = {
    "cube": {
        "bf16:NN": (1.009706, 1.020883, 1.021089), "bf16:TN": (1.029379, 1.029950, 1.032176),
        "fp16:NN": (1.029128, 1.033937, 1.034302), "fp16:NT": (1.028430, 1.028731, 1.029029),
        "fp16:TT": (1.003662, 1.022052, 1.034961),
    },
    "mte1": {
        "bf16:NN": (0.877909, 1.067033, 1.067668), "bf16:TN": (1.142847, 1.143867, 1.156761),
        "fp16:NN": (0.998051, 1.141298, 1.152905), "fp16:NT": (0.963927, 1.148921, 1.149366),
        "fp16:TT": (0.969744, 1.034452, 1.063317),
    },
    "mte2": {
        "bf16:NN": (1.141364, 1.146674, 1.155245), "bf16:TN": (0.964621, 1.194048, 1.204995),
        "fp16:NN": (0.965802, 0.986148, 0.991264), "fp16:NT": (0.928807, 0.971484, 0.974146),
        "fp16:TT": (0.940594, 0.976449, 1.147095),
    },
}
PMU_FALLBACK = {
    "cube": (1.009892, 1.028731, 1.034214),
    "mte1": (0.970598, 1.062559, 1.149623),
    "mte2": (0.944289, 0.982765, 1.165318),
}
