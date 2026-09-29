from __future__ import annotations

import os

from tiling import base, estimator_algs, limits, pso, valids


KERNEL = os.environ["ATTENTION_KERNEL"]

# (B, N1, N2, S1, S2, D, DV). Replace this list with the collected
# DeepSeek/Pangu workload shapes before starting a search campaign.
SIZES = [
    (1, 8, 1, 128, 1536, 128, 128),
]

FEATURES = {
    "DTYPE_BYTES": 4,
    "LAYOUT": valids.attention.LAYOUT_BNSD,
    "DETERMINISTIC": 0,
    "SPARSE_MODE": 0,
    "IS_SPARSE": 0,
    "HAS_MASK": 0,
    "HAS_PSE": 0,
    "HAS_DROP": 0,
    "HAS_ROPE": 0,
    "HAS_SINK": 0,
    "HAS_ACTUAL_SEQ": 1,
    "HAS_START_IDX": 0,
}


class PsoAlgo(estimator_algs.AlgoProfileEst, pso.PsoAlgo):
    pass


def get_domains(kernel: str) -> dict[str, list[int]]:
    if kernel in ("fa_general", "fa_varlen"):
        return {
            "FA_S1_BASE": [64, 128],
            "FA_S2_BASE": [64, 128],
            "FA_N_RATIO": [1, 2, 4, 5, 6, 8],
        }
    if kernel == "fag_mla":
        return {}
    if kernel == "fag_same_ab":
        return {
            "FAG_S1_CV_INNER": [256, 512, 1024],
            "FAG_S2_CV_INNER": [256, 512, 1024],
        }
    if kernel == "fag_generic":
        return {
            "FAG_S1_INNER": list(range(16, 129, 16)),
            "FAG_S2_INNER": list(range(16, 65, 16)),
            "FAG_S1_CV_RATIO": [1, 4],
            "FAG_S2_CV_RATIO": [2, 4, 8, 16],
        }
    raise ValueError(f"unsupported attention route: {kernel}")


def get_validator(kernel: str, domains: dict[str, list[int]]):
    hardware = limits.AttentionLimits(
        max_cores=48,
        aic_num=24,
        UB_size=192 * 1024,
        L1_size=512 * 1024,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L2_size=192 * 1024**2,
        domains=domains,
        calc_type_size=4,
    )
    return valids.attention.create_validator(kernel, hardware)


def get_input_params(kernel: str, shape: tuple[int, ...]) -> list[base.BaseParam]:
    batch, q_heads, kv_heads, s1, s2, d, dv = shape
    prefix = "FA" if kernel.startswith("fa_") else "FAG"
    values = {
        f"{prefix}_B": batch,
        f"{prefix}_N1": q_heads,
        f"{prefix}_N2": kv_heads,
        f"{prefix}_S1": s1,
        f"{prefix}_S2": s2,
        f"{prefix}_D": d,
        f"{prefix}_DV": dv,
        **{f"{prefix}_{name}": value for name, value in FEATURES.items()},
        "ATTENTION_GRAD": int(prefix == "FAG"),
    }
    return [base.BaseParam(name=name, value=value, is_const=True) for name, value in values.items()]


def main() -> None:
    domains = get_domains(KERNEL)
    validator = get_validator(KERNEL, domains)
    print(f"KERNEL: {KERNEL}")
    print(f"DOMAINS: {domains}")

    for shape in SIZES:
        input_params = get_input_params(KERNEL, shape)
        if not validator.get_combinations(1, input_params):
            raise ValueError(f"shape/features do not reach a legal {KERNEL} configuration: {shape}")

        shape_key = "_".join(str(value) for value in shape)
        algo = PsoAlgo(
            is_stop=lambda results: len(results) >= 48,
            swarm_size=64,
            validator=validator,
            input_params=input_params,
            runner="./run_attention.sh",
            cache_path=f"cache_{KERNEL}_{shape_key}_real.json",
            verbose=True,
        )
        print(f"START: {shape}")
        algo()
        print(f"END: {shape}")
