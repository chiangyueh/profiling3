from tiling import limits, valids, base, pso, msprof
from scripts.gen_data import gen_golden_data

SIZES = [
        (1, 7168, 65536),
        (16, 7168, 65536),
        (64, 7168, 65536),
        (128, 7168, 65536),
        ]

SWARM_SIZE = 16
ITERATIONS = 30

class Pso(msprof.AlgoProfile, pso.PsoAlgo): pass

def get_domains() -> dict:
    return {
        'MM_SINGLE_K': [128, 256, 384, 512],
        'MM_SINGLE_N': [1792, 3584, 7168],
        'MM_CORE_NUM': [8, 12, 16, 20],
        'MM_BASE_M': [16, 32, 64, 128],
        'MM_BASE_N': [64, 128, 256],
        'MM_BASE_K': [64, 128, 256],
        'MM_STEP_M': [1, 2, 3],
        'MM_STEP_N': [1, 2],
        'MM_STEP_Ka': [1, 2, 3, 4],
        'MM_STEP_Kb': [1, 2, 3, 4],
        'MM_DEPTH_A1': [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24],
        'MM_DEPTH_B1': [1, 2, 3, 4, 6, 8, 12, 16],
        'MM_DB_L0C': [1, 2]}


def get_input_params(M: int, N: int, K: int) -> list[base.BaseParam]:
    return [
        base.BaseParam(name="MM_M", value=M, is_const=True),
        base.BaseParam(name="MM_N", value=N, is_const=True),
        base.BaseParam(name="MM_K", value=K, is_const=True),
        base.BaseParam(name="MM_SINGLE_M", value=128, is_const=True),
        base.BaseParam(name="MM_ITERATE_ORDER", value=1, is_const=True),
        base.BaseParam(name="MM_ND2NZ_A", value=1, is_const=True),
        base.BaseParam(name="MM_ND2NZ_B", value=0, is_const=True),
        base.BaseParam(name="MM_DB_L0A", value=2, is_const=True),
        base.BaseParam(name="MM_DB_L0B", value=2, is_const=True),
    ]


def get_validator(domains: dict) -> valids.matmul.MatmulDetSplitKValidator:
    lims = limits.MatmulLimits(
        max_cores=20,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L1_size=512 * 1024,
        L2_size=192 * 1024**2,
        # ровно как у хоста: UB_SIZE = 196352 в matmul_v3_base_tiling.cpp,
        # а не 192*1024 — разница в 256 Б меняет MM_ND2NZ_BASE_AN
        UB_size=196352,
        domains=domains,
        dtype_size=2,
    )
    return valids.matmul.MatmulDetSplitKValidator(lims)


for M, N, K in SIZES:

    domains = get_domains()
    print(f"DOMAINS: {domains}")
    gen_golden_data(M, N, K)
    algo = Pso(
        
        is_stop=lambda res: len(res) >= ITERATIONS,
        validator=get_validator(domains),
        input_params=get_input_params(M, N, K),
        swarm_size=SWARM_SIZE,
        cache_path=f"cache_det_{M}_{N}_{K}.json",
        verbose=True,
    )
    print(f"START: M={M}, N={N}, K={K}")
    results = algo()
    print(f"END: M={M}, N={N}, K={K}")
    print(f"BEST: {results[-1]}")
