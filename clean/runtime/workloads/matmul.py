from .. import inputs as cost


def cube_job(
    p,
    s,
    geometry,
    stage,
    rows,
    columns,
    heads,
    kv_heads,
    shared,
    carried,
    q_carried=False,
):
    align, ceil = cost.align, cost.ceil_div
    m, n, k = (
        rows,
        columns if stage == 0 else p["DV"],
        p["D"] if stage == 0 else columns,
    )
    bm, bn, bk = (geometry[name] for name in ("base_m", "base_n", "base_k"))
    mt, nt, kt = ceil(m, bm), ceil(n, bn), ceil(k, bk)
    tiles = heads * mt * nt * kt
    a_bytes = heads * m * align(k, 16) * 2
    b_bytes = kv_heads * columns * align(p["D"] if stage == 0 else p["DV"], 16) * 2
    if shared or carried:
        b_bytes = 0
    if q_carried:
        a_bytes = 0
    factor = p["B"] if p["layout"] == "SBH" else 1
    qstride = p["D"] * (p["N1"] * factor if p["layout"] != "BNSD" else 1)
    kvstride = (p["D"] if stage == 0 else p["DV"]) * (
        p["N2"] * factor if p["layout"] != "BNSD" else 1
    )
    gap = (
        heads * max(0, m - 1) * max(0, qstride - p["D"]) * 2
        if stage == 0 and not q_carried
        else 0
    )
    if not shared and not carried:
        gap += (
            kv_heads
            * max(0, columns - 1)
            * max(0, kvstride - (p["D"] if stage == 0 else p["DV"]))
            * 2
        )
    output_width = 4
    nz = (
        (s["nz"] and (s["family"] == "s1" or columns % 64 != 0))
        if stage == 0
        else (
            s["same_ab"]
            or p["layout"] == "TND"
            or s["family"] in ("s1", "b", "b_drop")
            and p["DV"] % 16 != 0
        )
    )
    write_kb = heads * m * align(n) * output_width / 1024
    loads = heads * ceil(mt * kt, geometry["depth_a1"]) + (
        0 if shared or carried else kv_heads * ceil(nt * kt, geometry["depth_b1"])
    )
    calls, k_slices = 1, []
    if p["layout"] == "TND" and s["same_ab"] and s["l1"]:
        if stage == 0:
            calls = ceil(columns, 256)
            loads = (0 if q_carried else heads) + kv_heads * calls
        else:
            parts, nloops = [
                min(192, columns - start) for start in range(0, columns, 192)
            ], ceil(p["DV"], 128)
            total_k = sum(align(part) for part in parts)
            k_slices = [align(part) / total_k for part in parts]
            a_bytes *= nloops
            calls = nloops * (len(parts) + 1)
            loads = (heads + kv_heads) * nloops * len(parts)
    k_parts = (
        [align(min(bk, k - start)) for start in range(0, k, bk)]
        if s["policy"] == "NORMAL"
        else []
    )
    return {
        "flops_m": 2 * heads * align(m) * align(n) * align(k) / 1e6,
        "tiles": tiles,
        "output_tiles": heads * mt * nt,
        "l0_kb": heads
        * 2
        * (align(m) * align(k) * nt + align(n) * align(k) * mt)
        / 1024,
        "read_kb": (a_bytes + b_bytes) / 1024,
        "address_gap_kb": gap / 1024,
        "write_kb": write_kb,
        "nd_write_kb": 0 if nz else write_kb,
        "nz_write_kb": write_kb if nz else 0,
        "loads": loads,
        "stores": heads * mt * nt,
        "calls": calls,
        "commands": calls + tiles,
        "k_slices": k_slices,
        "pipeline_k_slices": (
            [part / sum(k_parts) for part in k_parts] if len(k_parts) > 1 else []
        ),
        "pipeline_k_buffers": min(
            max(1, geometry["depth_a1"] // max(1, geometry.get("step_m", 1))),
            max(1, geometry["depth_b1"] // max(1, geometry.get("step_n", 1))),
        ),
        "query_rows": heads * rows,
    }


def matmul_cache_trace(
    p, mm, stage, rows, columns, shared=False, carried=False, q_carried=False
):
    ceil, align = cost.ceil_div, cost.align
    m, n, k = (
        rows,
        columns if stage == 0 else p["DV"],
        p["D"] if stage == 0 else columns,
    )
    bm, bn, bk = (mm[name] for name in ("base_m", "base_n", "base_k"))
    mt, nt, kt = ceil(m, bm), ceil(n, bn), ceil(k, bk)
    sm, sn = max(1, mm["step_m"]), max(1, mm["step_n"])
    order = mm["iterate_order"]

    def capacity(depth, major):
        if depth > 2:
            return depth if depth >= kt * major else depth - 2
        return 0 if depth < kt * major or kt == 1 and depth == 2 else depth

    ac, bc = capacity(mm["depth_a1"], sm), capacity(mm["depth_b1"], sn)
    astream = max(1, mm["depth_a1"] - ac)
    bstream = max(1, mm["depth_b1"] - bc)
    ap, bp = 0, 0
    trace = []
    outer = range(0, mt, sm) if order == 1 else range(0, nt, sn)
    for major in outer:
        if order == 1:
            ap = 0
        else:
            bp = 0
        minor = range(0, nt, sn) if order == 1 else range(0, mt, sm)
        for secondary in minor:
            first = (
                range(secondary, min(secondary + sn, nt))
                if order == 1
                else range(secondary, min(secondary + sm, mt))
            )
            for i in first:
                if order == 1:
                    bp = 0
                else:
                    ap = 0
                inner = (
                    range(major, min(major + sm, mt))
                    if order == 1
                    else range(major, min(major + sn, nt))
                )
                for j in inner:
                    mi, ni = (j, i) if order == 1 else (i, j)
                    parts = []
                    for ki in range(kt):
                        ai = (mi * kt + ki) % (sm * kt) if order == 1 else ki
                        bi = ki if order == 1 else (ni * kt + ki) % (sn * kt)
                        amiss = not (ai < ac and ai < ap) and not q_carried
                        bmiss = not (bi < bc and bi < bp) and not (shared or carried)
                        if amiss and ai < ac:
                            ap += 1
                        if bmiss and bi < bc:
                            bp += 1
                        ak = align(min(bk, k - ki * bk))
                        ar, br = min(bm, m - mi * bm), min(bn, n - ni * bn)
                        ar_bytes = 2 * ar * ak if amiss else 0
                        br_bytes = 2 * br * ak if bmiss else 0
                        factor = p["B"] if p["layout"] == "SBH" else 1
                        astride = p["D"] * (
                            p["N1"] * factor if p["layout"] != "BNSD" else 1
                        )
                        bwidth = p["D"] if stage == 0 else p["DV"]
                        bstride = bwidth * (
                            p["N2"] * factor if p["layout"] != "BNSD" else 1
                        )
                        gap = (
                            max(0, ar - 1) * max(0, astride - p["D"]) * 2
                            if amiss and stage == 0
                            else 0
                        )
                        gap += (
                            max(0, br - 1) * max(0, bstride - bwidth) * 2
                            if bmiss
                            else 0
                        )
                        parts.append(
                            {
                                "read_kb": (ar_bytes + br_bytes) / 1024,
                                "loads": int(amiss) + int(bmiss),
                                "address_gap_kb": gap / 1024,
                                "a_slot": ai if ai < ac else ac + ki % astream,
                                "b_slot": bi if bi < bc else bc + ki % bstream,
                                "a_miss": amiss,
                                "b_miss": bmiss,
                                "a_cached": ai < ac,
                                "b_cached": bi < bc,
                            }
                        )
                    trace.append(parts)
    return trace


def matmul_microtiles(p, s, mm, stage, rows, columns, heads):
    m, n, k = (
        rows,
        columns if stage == 0 else p["DV"],
        p["D"] if stage == 0 else columns,
    )
    bm, bn, bk = (mm[name] for name in ("base_m", "base_n", "base_k"))
    manual = p["layout"] == "TND" and s["l1"]
    nsplit = (
        256
        if manual and stage == 0
        else 128 if manual and s["same_ab"] and stage == 1 else n
    )
    ksplit = (192 if s["same_ab"] else 256) if manual and stage == 1 else k
    kparts = [
        min(bk, min(ksplit, k - outer) - inner)
        for outer in range(0, k, ksplit)
        for inner in range(0, min(ksplit, k - outer), bk)
    ]
    result = []
    api = 0
    for head in range(heads):
        for start in range(0, n, nsplit):
            count = min(nsplit, n - start)
            mt, nt = cost.ceil_div(m, bm), cost.ceil_div(count, bn)
            sm, sn = max(1, mm["step_m"]), max(1, mm["step_n"])
            order = mm["iterate_order"]
            for major in range(0, mt if order == 1 else nt, sm if order == 1 else sn):
                for secondary in range(
                    0, nt if order == 1 else mt, sn if order == 1 else sm
                ):
                    for first in range(
                        secondary,
                        min(
                            secondary + (sn if order == 1 else sm),
                            nt if order == 1 else mt,
                        ),
                    ):
                        for second in range(
                            major,
                            min(
                                major + (sm if order == 1 else sn),
                                mt if order == 1 else nt,
                            ),
                        ):
                            mi, ni = (second, first) if order == 1 else (first, second)
                            rm, rn = min(bm, m - mi * bm), min(bn, count - ni * bn)
                            mad_m, mad_n = (
                                (bm, bn)
                                if s.get("static_matmul")
                                else (16 if rm == 1 else rm, rn)
                            )
                            barrier = int((mad_m // 16) * (mad_n // 16) < 10)
                            result.append(
                                {
                                    "m": rm,
                                    "n": rn,
                                    "k": kparts,
                                    "barrier": barrier,
                                    "api": api if manual else 0,
                                    "head": head,
                                }
                            )
            api += 1
    return result


def refine_matmul_job(p, s, mm, stage, rows, columns, heads, job):
    tiles = matmul_microtiles(p, s, mm, stage, rows, columns, heads)
    job["micro_tiles"] = tiles
    job["tiles"] = sum(len(t["k"]) for t in tiles)
    job["output_tiles"] = job["stores"] = len(tiles)
    job["flops_m"] = sum(
        2
        * cost.align(t["m"])
        * cost.align(t["n"])
        * cost.align(k)
        / 1e6
        for t in tiles
        for k in t["k"]
    )
    job["l0_kb"] = sum(
        2
        * (cost.align(t["m"]) + cost.align(t["n"]))
        * cost.align(k)
        / 1024
        for t in tiles
        for k in t["k"]
    )
    manual = p["layout"] == "TND" and s["l1"]
    job["calls"] = len({t["api"] for t in tiles}) if manual else 1
    api_steps = job["calls"]
    if manual and stage == 1:
        width = 192 if s["same_ab"] else 256
        api_steps *= cost.ceil_div(columns, width) + 1
    job["commands"] = (
        api_steps + job["tiles"] + sum(t["barrier"] * len(t["k"]) for t in tiles)
    )
    job["api_steps"] = api_steps
    job["m_barriers"] = sum(t["barrier"] * len(t["k"]) for t in tiles)
    return job
