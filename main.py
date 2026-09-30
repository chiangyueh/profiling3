import sys

if any(arg == "--id" or arg.startswith("--id=") for arg in sys.argv[1:]):
    from attention_main import main as run_attention

    run_attention()
    raise SystemExit(0)

from tiling import limits, valids, pso, base, cache_model, bf, msprof
from scripts.gen_data import gen_golden_data

SIZES = [
        (64, 7168, 65536)
        ]

class PsoAlgo(cache_model.BaseAlgoModel, bf.BruteForceAlgo): pass

def get_domains() -> dict:
    return {
        'MM_SINGLE_M': [32, 64],
        'MM_SINGLE_N': [x for x in range(1024, 7168 + 1, 1024)],
        'MM_SINGLE_K': [32, 64, 128, 256, 384],
        'MM_STEP_M': [1,2,3],
        'MM_STEP_N': [1,2,3],
        'MM_BASE_M': [32, 64],
        'MM_BASE_N': [32, 64, 128, 256, 512],
        'MM_BASE_K': [32, 64, 128, 256, 512],
        'MM_STEP_Ka': [x for x in range(1, 8 + 1)],
        'MM_STEP_Kb': [x for x in range(1, 8 + 1)],
        'MM_M_TILE_BLOCK': [x+1 for x in range(8)],
        'MM_N_TILE_BLOCK': [x+1 for x in range(8)],
        'MM_DB_L0C': [1,2]}

    
def get_validator(domains: dict) -> valids.matmul.MatmulDetSplitKValidator:
    lims = limits.MatmulLimits(
        max_cores=20,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L1_size=512 * 1024,
        L2_size=192 * 1024**2,               
        domains=domains,       
        dtype_size=2,  
    )
    return valids.matmul.MatmulDetSplitKValidator(lims)
     

for M, N, K in SIZES:
           
    input_params = [
        base.BaseParam(name="MM_M", value=M, is_const=True),
        base.BaseParam(name="MM_N", value=N, is_const=True),
        base.BaseParam(name="MM_K", value=K, is_const=True),
        base.BaseParam(name='MM_DB_L0A', value=2, is_const=True),
        base.BaseParam(name='MM_DB_L0B', value=2, is_const=True),
        base.BaseParam(name='MM_ITERATE_ORDER', value=1, is_const=True)
    ]
    domains = get_domains()
    print(f"DOMAINS: {domains}")
    # gen_golden_data(M, N, K)
    algo = PsoAlgo(
        is_stop=lambda res: len(res) >= 1,
        validator=get_validator(domains),
        input_params=input_params,
        cache_path=f"cache_{M}_{N}_{K}_single.json",
        verbose=True,
    )
    print(f"START: M={M}, N={N}, K={K}")
    results = algo()
    print(f"END: M={M}, N={N}, K={K}")
