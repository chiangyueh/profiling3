import math

from .. import inputs as cost


def vector_job(p, s, rows, columns, first, last):
    align, ceil = cost.align, cost.ceil_div
    batch = s["family"] in ("b", "b_drop")
    windowed = s["same_ab"] or s["family"] in ("general", "varlen")
    padded = (
        align(columns, 64) if s["family"] == "s1" or s["same_ab"] else align(columns)
    )
    v1base = min(rows, s["s1_base"] if batch else max(1, 1024 // padded * 8))
    if p["layout"] == "TND" and columns < 8:
        v1base = rows
    if windowed:
        v2base = (
            max(1, 1024 // align(p["D"]) * 8)
            if align(p["D"]) > 64
            else min(128, rows) if s["same_ab"] or p["layout"] == "TND" else rows
        )
    else:
        v2base = min(
            align(p["S1"]),
            max(
                1,
                (
                    8192 // align(p["D"]) // 16 * 16
                    if align(p["D"]) <= 512
                    else 8192 // align(p["D"])
                ),
            ),
        )
    c1, c2 = ceil(rows, v1base), ceil(rows, v2base)
    soft_rows = sum(align(min(v1base, rows - pos), 8) for pos in range(0, rows, v1base))
    soft_elements = soft_rows * padded
    nz1 = s["nz"] and (s["family"] == "s1" or columns % 64 != 0)
    probability_nz = s["same_ab"] or (not batch and s["nz"])
    nz2 = (
        s["same_ab"]
        or p["layout"] == "TND"
        or (s["family"] in ("s1", "b", "b_drop") and p["DV"] % 16 != 0)
    )
    width = align(p["DV"])
    stats = rows * 32
    drop_bytes = (
        rows
        * (align(columns, 32) if s["drop_adapter"] else ceil(align(columns, 32), 8))
        if p["has_drop_mask"]
        else 0
    )
    stage1 = {
        "softmax_k": soft_elements / 1000,
        "reduction_k": soft_rows * math.ceil(math.log2(max(2, padded))) / 1000,
        "cast_k": rows * padded / 1000,
        "format_k": rows * padded * (int(nz1) + int(probability_nz)) / 1000,
        "online_k": (0 if first else rows * 8) / 1000,
        "drop_k": rows * padded / 1000 if p["has_drop_mask"] else 0,
        "chunks": c1,
        "calls": 1,
        "read_kb": (4 * rows * align(columns) + drop_bytes) / 1024,
        "write_kb": (
            2 * rows * align(columns) + (stats if last else 0)
            if windowed
            else 2 * rows * align(columns) + 2 * stats
        )
        / 1024,
        "loads": c1 * (1 + int(bool(p["has_drop_mask"]))),
        "stores": c1 + int(last) if windowed else c1 * 3,
        "vec1_calls": c1,
        "softmax_initial_calls": c1 if first else 0,
        "softmax_update_calls": 0 if first else c1,
    }
    stage1.update(
        nz2nd_k=rows * padded * int(nz1) / 1000,
        nd2nz_k=rows * padded * int(probability_nz) / 1000,
        drop_chunks=c1 if p["has_drop_mask"] else 0,
        alignment_read_kb=4 * rows * (padded - align(columns)) / 1024 if nz1 else 0,
    )
    output_elements = rows * width
    stage2 = {
        "softmax_k": 0,
        "reduction_k": 0,
        "cast_k": output_elements / 1000 if last else 0,
        "format_k": output_elements / 1000 if nz2 else 0,
        "online_k": 0,
        "rescale_k": output_elements * int(not first) * 2 / 1000,
        "normalize_k": output_elements * int(last) / 1000,
        "drop_k": 0,
        "chunks": c2,
        "calls": 1,
        "read_kb": (
            4 * output_elements
            + (0 if windowed else stats)
            + (0 if first else 4 * output_elements)
        )
        / 1024,
        "write_kb": (
            (2 * rows * p["DV"] if last else 4 * output_elements)
            + (stats if windowed and last else 0)
        )
        / 1024,
        "loads": c2 * (1 + int(not windowed) + int(not first)),
        "stores": c2 + ((1 if s["same_ab"] else c2) if windowed and last else 0),
        "vec2_calls": c2,
        "accumulator_read_bytes": 0 if first else 4 * output_elements,
        "accumulator_write_bytes": 0 if last else 4 * output_elements,
        "final_output_bytes": 2 * rows * p["DV"] if last else 0,
    }
    output_stride = p["DV"] * (
        p["N1"] * (p["B"] if p["layout"] == "SBH" else 1)
        if p["layout"] != "BNSD"
        else 1
    )
    stage2["address_gap_kb"] = (
        max(0, (rows - 1) * (output_stride - p["DV"]) * 2) / 1024 if last else 0
    )
    stage2.update(nz2nd_k=output_elements / 1000 if nz2 else 0, nd2nz_k=0)
    return stage1, stage2
