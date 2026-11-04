from __future__ import annotations
from collections import Counter
from prediction import fa_latency_inputs as input_cost
SOURCE_ROOT = "attention/flash_attention_score/"


def sameab_workload(inputs: dict, hardware: dict) -> dict:
    p = input_cost.normalize(inputs)
    align, ceil = input_cost.align, input_cost.ceil_div
    if p["dtype"] not in ("FP16", "BF16") or p["layout"] not in ("BNSD", "SBH", "BSND", "TND"):
        raise ValueError("unsupported SameAB source-analysis dtype or layout")
    tnd = p["layout"] == "TND"
    if tnd:
        eligible = p["N1"] != 1 and ((p["B"] >= 4 and p["B"] * p["S1"] >= 8192 and p["B"] * p["S2"] >= 8192 and min(p["S1"], p["S2"]) >= 512) or min(p["S1"], p["S2"]) > 5120)
    else:
        eligible = ((p["D"] % 16 or p["D"] == 96) and p["S2"] >= 512) or (p["S2"] > 1024 and 128 < p["D"] < 196) or (p["D"] == 64 and p["S2"] % 64 and 2048 < p["S2"] < 18432)
    if not eligible:
        raise ValueError("inputs do not select a supported SameAB kernel")
    policy = "UNSPLITK" if not tnd and p["D"] == 192 and p["S1"] % 128 == 0 and p["S2"] % 128 == 0 and p["layout"] == "BNSD" else "NORMAL"
    best = not tnd and (p["D"] % 16 or p["D"] == 96) and p["S2"] >= 512 and p["S1"] > 2048 and p["layout"] != "BSND" and int(hardware["l2_bytes"]) > 96 * 1024 ** 2
    base = min(384 if best and ceil(align(p["S1"]), 384) < ceil(align(p["S1"]), 256) else 256, align(p["S1"]))
    ratio = 4 if tnd and align(p["D"]) == 128 and int(hardware["l2_bytes"]) <= 96 * 1024 ** 2 and p["B"] * p["S1"] >= 48000 else 8
    if best and 1025 <= p["S2"] <= 1040:
        ratio = 5
    if not tnd and p["D"] == 64 and p["layout"] == "BNSD" and 2049 <= p["S2"] <= 2304:
        ratio = 6
    s2_base = min(128, align(p["S2"]))
    total = p["B"] * p["N1"] * ceil(p["S1"], base)
    cubes = min(total, int(hardware["aic_cores"]))
    split = ceil(total, cubes)
    schedule = {"family": "varlen" if tnd else "same_ab", "priority": 94 if tnd else 95, "same_ab": True,
                "s1_base": base, "s2_base": s2_base, "n_ratio": min(ratio, ceil(p["S2"], s2_base)),
                "s1_outer": ceil(p["S1"], base), "total_tasks": total, "split_factor": split,
                "core_num": 2 * (cubes if tnd else ceil(total, split))}
    compiled_nz = tnd or (p["S2"] % 64 != 0 and policy == "NORMAL")
    l1_carry = tnd and base <= 256 and p["D"] in (80, 88, 128, 192)
    implementation = "TND_L1_CARRY" if l1_carry else "TND_GM" if tnd else "UNSPLITK_TSCM" if policy == "UNSPLITK" else "NORMAL_NZ" if compiled_nz else "NORMAL_ND"
    stride_factor = p["B"] if p["layout"] == "SBH" else 1
    query_heads, kv_heads = (1, 1) if p["layout"] == "BNSD" else (p["N1"], p["N2"])
    row_strides = {"query": stride_factor * query_heads * p["D"], "key": stride_factor * kv_heads * p["D"],
                   "value": stride_factor * kv_heads * p["DV"]}
    adapter = bool(p["has_drop_mask"] and (p["S2"] % 8 or tnd and p["B"] > 1))
    base, window = schedule["s1_base"], schedule["s2_base"] * schedule["n_ratio"]
    full, tail = divmod(p["S1"], base)
    blocks = [(base, full)] if full else []
    if tail:
        blocks.append((tail, 1))
    totals, phases = Counter(), []
    for cube_rows, repeats in blocks:
        multiplicity = repeats * p["B"] * p["N1"]
        for offset in range(0, p["S2"], window):
            columns = min(window, p["S2"] - offset)
            padded = align(columns, 64)
            first, last = offset == 0, offset + columns == p["S2"]
            nz2nd = compiled_nz and columns % 64 != 0
            clients, counts = [], Counter()
            for rows in (ceil(cube_rows, 2), cube_rows // 2):
                if not rows:
                    clients.append({"rows": 0, "vec1_calls": 0, "vec2_calls": 0})
                    continue
                vec1_base = rows if tnd and columns < 8 else min(1024 // padded * 8, rows)
                vec2_base = 1024 // align(p["D"]) * 8 if align(p["D"]) > 64 else min(128, rows)
                if min(vec1_base, vec2_base) <= 0:
                    raise ValueError("unsupported official SameAB Vector geometry")
                vec1_calls, vec2_calls = ceil(rows, vec1_base), ceil(rows, vec2_base)
                softmax_rows = (vec1_calls - 1) * align(vec1_base, 8) + align(rows - (vec1_calls - 1) * vec1_base, 8)
                clients.append({"rows": rows, "vec1_base": vec1_base, "vec2_base": vec2_base,
                                "vec1_calls": vec1_calls, "vec2_calls": vec2_calls})
                counts.update(vec1_calls=vec1_calls, vec2_calls=vec2_calls,
                              softmax_initial_calls=vec1_calls if first else 0,
                              softmax_update_calls=0 if first else vec1_calls,
                              softmax_padded_elements=softmax_rows * padded,
                              bmm1_nz_to_nd_calls=vec1_calls if nz2nd else 0,
                              tail_mask_calls=vec1_calls if columns % 64 else 0,
                              probability_cast_elements=rows * padded,
                              probability_nd_to_nz_calls=vec1_calls,
                              bmm2_nz_to_nd_calls=vec2_calls,
                              rescale_add_elements=0 if first else rows * align(p["DV"]),
                              final_divide_elements=rows * align(p["DV"]) if last else 0,
                              accumulator_read_bytes=0 if first else rows * align(p["DV"]) * 4,
                              accumulator_write_bytes=0 if last else rows * align(p["DV"]) * 4,
                              final_output_bytes=rows * p["DV"] * 2 if last else 0)
            counts.update(bmm1_calls=1, bmm2_calls=1,
                          model_vec1_calls=1,
                          model_vec2_calls=ceil(cube_rows, min(align(p["S1"]), 8192 // align(p["DV"]) // 16 * 16)),
                          model_accumulator_read_bytes=cube_rows * p["DV"] * 4,
                          model_output_write_bytes=cube_rows * p["DV"] * 2)
            totals.update({name: value * multiplicity for name, value in counts.items()})
            phases.append({"cube_rows": cube_rows, "columns": columns, "columns_align16": align(columns), "columns_align64": padded,
                           "query_blocks": multiplicity, "s2_offset": offset,
                           "phase": "single" if first and last else "first" if first else "last" if last else "middle",
                           "bmm1_actual_format": "NZ" if nz2nd else "ND", "vector_clients": clients, "counts_per_query_block": dict(counts)})
    return {"schedule": schedule, "implementation": implementation, "matmul_policy": policy,
            "layout": p["layout"], "input_dtype": p["dtype"], "row_strides_elements": row_strides,
            "drop_mask_mode": "byte" if adapter else "bit" if p["has_drop_mask"] else "none",
            "bmm1_template_format": "NZ" if compiled_nz else "ND", "bmm2_probability_format": "NZ",
            "l1_carry": l1_carry, "softmax_reduce_size": 1 if not tnd and base > 256 else 8,
            "totals": dict(totals), "phases": phases,
            "vector_calls_meaning": "inner SoftMaxCompute and ProcessVec2 chunk iterations across both Vector clients",
            "bmm_calls_meaning": "outer IterateBmm1/IterateBmm2 or TND ComputeMm1/ComputeMm2 invocations, not hardware MMA instructions",
            "scope": "dense FP16/BF16; uniform-length TND; no PSE/attention mask/Rope/sink",
            "matmul_geometry": "baseM/baseN/baseK and depths still require the deterministic Host Matmul tiler; not guessed here",
            "source": {"dispatch": SOURCE_ROOT + "op_kernel/flash_attention_score.cpp:419",
                       "dense_vector": SOURCE_ROOT + "op_kernel/arch32/flash_attention_score_s1s2_bn2gs1_sab.h:924",
                       "dense_accumulator": SOURCE_ROOT + "op_kernel/arch32/flash_attention_score_s1s2_bn2gs1_sab.h:2368",
                       "tnd": SOURCE_ROOT + "op_kernel/arch32/flash_attention_var_len_score_sab.h:634"}}
