from __future__ import annotations
from .. import fa_latency_inputs as input_cost
from .schedule_common import controlled_program as program, sequence_work


def sameab_workload(inputs: dict, hardware: dict) -> dict:
    p = input_cost.normalize(inputs)
    align, ceil = input_cost.align, input_cost.ceil_div
    if p["dtype"] not in ("FP16", "BF16") or p["layout"] not in (
        "BNSD",
        "SBH",
        "BSND",
        "TND",
    ):
        raise ValueError("unsupported SameAB source-analysis dtype or layout")
    tnd = p["layout"] == "TND"
    if tnd:
        eligible = p["N1"] != 1 and (
            (
                p["B"] >= 4
                and p["B"] * p["S1"] >= 8192
                and p["B"] * p["S2"] >= 8192
                and min(p["S1"], p["S2"]) >= 512
            )
            or min(p["S1"], p["S2"]) > 5120
        )
    else:
        eligible = (
            ((p["D"] % 16 or p["D"] == 96) and p["S2"] >= 512)
            or (p["S2"] > 1024 and 128 < p["D"] < 196)
            or (p["D"] == 64 and p["S2"] % 64 and 2048 < p["S2"] < 18432)
        )
    if not eligible:
        raise ValueError("inputs do not select a supported SameAB kernel")
    policy = (
        "UNSPLITK"
        if not tnd
        and p["D"] == 192
        and p["S1"] % 128 == 0
        and p["S2"] % 128 == 0
        and p["layout"] == "BNSD"
        else "NORMAL"
    )
    best = (
        not tnd
        and (p["D"] % 16 or p["D"] == 96)
        and p["S2"] >= 512
        and p["S1"] > 2048
        and p["layout"] != "BSND"
        and int(hardware["l2_bytes"]) > 96 * 1024**2
    )
    base = min(
        384 if best and ceil(align(p["S1"]), 384) < ceil(align(p["S1"]), 256) else 256,
        align(p["S1"]),
    )
    ratio = (
        4
        if tnd
        and align(p["D"]) == 128
        and int(hardware["l2_bytes"]) <= 96 * 1024**2
        and p["B"] * p["S1"] >= 48000
        else 8
    )
    if best and 1025 <= p["S2"] <= 1040:
        ratio = 5
    if not tnd and p["D"] == 64 and p["layout"] == "BNSD" and 2049 <= p["S2"] <= 2304:
        ratio = 6
    s2_base = min(128, align(p["S2"]))
    total = p["B"] * p["N1"] * ceil(p["S1"], base)
    cubes = min(total, int(hardware["aic_cores"]))
    split = ceil(total, cubes)
    schedule = {
        "family": "varlen" if tnd else "same_ab",
        "priority": 94 if tnd else 95,
        "same_ab": True,
        "s1_base": base,
        "s2_base": s2_base,
        "n_ratio": min(ratio, ceil(p["S2"], s2_base)),
        "s1_outer": ceil(p["S1"], base),
        "total_tasks": total,
        "split_factor": split,
        "core_num": 2 * (cubes if tnd else ceil(total, split)),
    }
    compiled_nz = tnd or (p["S2"] % 64 != 0 and policy == "NORMAL")
    l1_carry = tnd and base <= 256 and p["D"] in (80, 88, 128, 192)
    implementation = (
        "TND_L1_CARRY"
        if l1_carry
        else (
            "TND_GM"
            if tnd
            else (
                "UNSPLITK_TSCM"
                if policy == "UNSPLITK"
                else "NORMAL_NZ" if compiled_nz else "NORMAL_ND"
            )
        )
    )
    base, window = schedule["s1_base"], schedule["s2_base"] * schedule["n_ratio"]
    full, tail = divmod(p["S1"], base)
    blocks = ([base] if full else []) + ([tail] if tail else [])
    for cube_rows in blocks:
        for offset in range(0, p["S2"], window):
            columns = min(window, p["S2"] - offset)
            padded = align(columns, 64)
            for rows in (ceil(cube_rows, 2), cube_rows // 2):
                if not rows:
                    continue
                vec1_base = (
                    rows if tnd and columns < 8 else min(1024 // padded * 8, rows)
                )
                vec2_base = (
                    1024 // align(p["D"]) * 8 if align(p["D"]) > 64 else min(128, rows)
                )
                if min(vec1_base, vec2_base) <= 0:
                    raise ValueError("unsupported official SameAB Vector geometry")
            if min(align(p["S1"]), 8192 // align(p["DV"]) // 16 * 16) == 0:
                raise ZeroDivisionError("integer division or modulo by zero")
    return {
        "schedule": schedule,
        "implementation": implementation,
        "matmul_policy": policy,
        "bmm1_template_format": "NZ" if compiled_nz else "ND",
        "l1_carry": l1_carry,
    }


def schedule(inputs, hardware):
    work = sameab_workload(inputs, hardware)
    s = dict(work["schedule"])
    s.update(
        reuse=False,
        l1=work["l1_carry"],
        nz=work["bmm1_template_format"] == "NZ",
        implementation=work["implementation"],
        policy=work["matmul_policy"],
    )
    return s


def workload(work):
    return sequence_work(work, 1)
