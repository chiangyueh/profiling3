from tiling import base, base_algs, limits, pso, valids


# Keep the same workflow as information.zip: edit the workload and domains
# here, then run `python3 main.py`.
OPERATOR = "flash_attention_score"

SIZES = [
    (1, 8, 8, 1024, 1024, 128, 128),
]


class PsoAlgo(base_algs.AttentionAlgoProfile, pso.PsoAlgo):
    pass


def get_domains(operator: str) -> dict:
    if operator == "flash_attention_score":
        return {
            "FA_S1_BASE": [64, 128, 256],
            "FA_S2_BASE": [64, 128, 256],
            "FA_D_BASE": [64, 128],
            "FA_CORE_NUM": [8, 16, 20],
        }
    if operator == "flash_attention_score_grad":
        return {
            "FAG_S1_INNER": [64, 128, 256],
            "FAG_S2_INNER": [64, 128, 256],
            "FAG_S1_CV_RATIO": [1, 2, 4],
            "FAG_S2_CV_RATIO": [1, 2, 4],
            "FAG_CORE_NUM": [8, 16, 20],
        }
    raise ValueError(f"unsupported operator: {operator}")


def get_validator(operator: str, domains: dict) -> valids.AttentionValidator:
    return valids.AttentionValidator(
        limits.AttentionLimits(
            domains=domains,
            operator=operator,
            max_cores=20,
        )
    )


def get_input_params(operator: str, shape: tuple) -> list:
    batch, q_heads, kv_heads, s1, s2, d, dv = shape
    prefix = "FA" if operator == "flash_attention_score" else "FAG"
    values = {
        f"{prefix}_B": batch,
        f"{prefix}_N1": q_heads,
        f"{prefix}_N2": kv_heads,
        f"{prefix}_S1": s1,
        f"{prefix}_S2": s2,
        f"{prefix}_D": d,
        f"{prefix}_DV": dv,
        f"{prefix}_DTYPE_BYTES": 2,
        f"{prefix}_LAYOUT": 0,
        "ATTENTION_GRAD": int(operator == "flash_attention_score_grad"),
    }
    return [base.BaseParam(name=name, value=value, is_const=True) for name, value in values.items()]


def main() -> None:
    domains = get_domains(OPERATOR)
    print(f"OPERATOR: {OPERATOR}")
    print(f"DOMAINS: {domains}")

    for shape in SIZES:
        batch, q_heads, kv_heads, s1, s2, d, dv = shape
        algo = PsoAlgo(
            is_stop=lambda results: len(results) >= 48,
            swarm_size=64,
            validator=get_validator(OPERATOR, domains),
            input_params=get_input_params(OPERATOR, shape),
            runner="./run.sh",
            cache_path=(
                f"cache_{OPERATOR}_{batch}_{q_heads}_{kv_heads}_{s1}_{s2}_{d}_{dv}_real.json"
            ),
            verbose=True,
        )
        print(f"START: B={batch}, N1={q_heads}, N2={kv_heads}, S1={s1}, S2={s2}, D={d}, DV={dv}")
        algo()
        print(f"END: B={batch}, N1={q_heads}, N2={kv_heads}, S1={s1}, S2={s2}, D={d}, DV={dv}")


if __name__ == "__main__":
    main()
