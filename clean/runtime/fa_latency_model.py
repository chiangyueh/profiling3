from __future__ import annotations
import ctypes
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
from . import fa_latency_inputs as cost
from . import sameab_schedule as legacy

ROOT = Path(__file__).resolve().parents[1]
PIPES = {
    "cube": ("aic_cube_active_sum_us", ("flops_m", "tiles", "output_tiles")),
    "mte1": ("aic_mte1_active_sum_us", ("l0_kb", "tiles")),
    "mte2": ("aic_mte2_active_sum_us", ("read_kb", "loads", "address_gap_kb")),
    "fixpipe": (
        "aic_fixpipe_active_sum_us",
        ("nd_write_kb", "nz_write_kb", "stores", "calls"),
    ),
    "vector": (
        "aiv_vec_active_sum_us",
        (
            "softmax_k",
            "reduction_k",
            "cast_k",
            "nz2nd_k",
            "nd2nz_k",
            "online_k",
            "rescale_k",
            "normalize_k",
            "drop_k",
            "unpack_k",
            "chunks",
        ),
    ),
    "vmte2": ("aiv_mte2_active_sum_us", ("read_kb", "alignment_read_kb", "loads")),
    "vmte3": ("aiv_mte3_active_sum_us", ("write_kb", "stores", "address_gap_kb")),
    "cscalar": ("aic_scalar_active_sum_us", ("workers", "batch_entries", "commands")),
    "vscalar": (
        "aiv_scalar_active_sum_us",
        ("workers", "batch_entries", "chunks", "drop_chunks", "adapter_workers"),
    ),
}
PHASES = ("startup", "drop_adapter", "bmm1", "vec1", "bmm2", "vec2")


def schedule(inputs, hardware):
    p = cost.normalize(inputs)
    if p["dtype"] not in ("FP16", "BF16") or p["layout"] not in (
        "BNSD",
        "SBH",
        "BSND",
        "TND",
    ):
        raise ValueError(
            "source model supports dense FP16/BF16 BNSD/SBH/BSND and uniform-length TND"
        )
    if int(hardware["aiv_cores"]) != 2 * int(hardware["aic_cores"]):
        raise ValueError("source model requires two Vector cores per Cube")
    try:
        work = legacy.sameab_workload(p, hardware)
        s = dict(work["schedule"])
        s.update(
            reuse=False,
            l1=work["l1_carry"],
            nz=work["bmm1_template_format"] == "NZ",
            implementation=work["implementation"],
            policy=work["matmul_policy"],
        )
    except ValueError as error:
        if str(error) != "inputs do not select a supported SameAB kernel":
            raise
        b, n, nk, q, k, d, dv = (p[name] for name in cost.INPUT_FIELDS[:7])
        align, ceil = cost.align, cost.ceil_div
        aq, ak, ad = align(q), align(k), align(d)
        aic, aiv = int(hardware["aic_cores"]), int(hardware["aiv_cores"])
        same, reuse, l1, nz = False, False, False, False
        if p["layout"] == "TND":
            family, priority = "varlen", 94
            best = (
                128
                if b * q <= 10000 or k >= 1024
                else 256 if b >= 5 or k >= 512 else 512
            )
            base, ratio = min(best, aq), min(8, ceil(k, min(128, ak)))
        elif k > 1024:
            family, priority = "general", 96
            base = min(
                128 if b * n * (n // nk) * ceil(q, min(64, aq)) > aiv else 64, aq
            )
            ratio = min(
                (
                    5
                    if 1025 <= k <= 1040
                    else (
                        6
                        if d == 64 and p["layout"] == "BNSD" and 2049 <= k <= 2304
                        else 8
                    )
                ),
                ceil(k, min(128, ak)),
            )
            reuse = d <= 128 and not (
                d == 64 and p["layout"] == "BNSD" and k % 256 and k not in (4032, 42192)
            )
            nz = bool(k % 64 and d != 64)
        else:
            bytes1, bytes2 = n * ((aq + ak) * ad + ak) * 2, n * (aq + ad) * ak * 2
            is_b = (
                ak <= 128
                and bytes1 < int(hardware["l1_bytes"])
                and bytes2 < int(hardware["l1_bytes"])
                and n * aq * ak <= 65536
            )
            if is_b:
                family, priority = ("b_drop" if p["has_drop_mask"] else "b"), 98
                base, ratio = min(aq, 256, 8192 // ak // 16 * 16), 0
            else:
                family, priority = "s1", 97
                initial = min(128, aq)
                l1 = ad == 64 and 512 <= ak <= 1024 and q <= 3840
                reuse = not l1 and (
                    (k >= 512 and b * n > 64 and d <= 128)
                    or (b * n >= 4800 and k == 256 and d == 144)
                )
                nz = bool(
                    k % 64 or k == 64 or k >= 704 and k % 128 and b * n * q > 20480
                )
                maximum = 1 if l1 and ak == 512 or ak < 128 else 4
                if nz and d <= 256:
                    maximum = 4
                ratio, total = 1, b * n * ceil(q, initial)
                while (
                    ratio < maximum
                    and total > (ratio - 1) * aiv // ratio
                    and initial * ratio < aq
                    and initial * ratio * ak <= 131072
                ):
                    ratio += 1
                    total = b * n * ceil(ceil(q, initial), ratio)
                while (
                    total <= (ratio - 1) * aiv // ratio
                    or initial * ratio > aq
                    or initial * ratio * ak > 131072
                ):
                    ratio -= 1
                    if ratio <= 0:
                        raise ValueError(
                            "official S1 block geometry is not representable"
                        )
                base = initial * ratio
        outer = ceil(q, base)
        total = b if family in ("b", "b_drop") else b * n * outer
        split = ceil(total, min(total, aiv)) * (2 if reuse else 1)
        cores = min(total, aiv)
        if family in ("s1", "b", "b_drop"):
            cores = min(aiv, ceil(total, split // (2 if reuse else 1)))
            if reuse and cores % 2:
                cores = min(aiv, cores + 1)
        s = {
            "family": family,
            "priority": priority,
            "same_ab": False,
            "reuse": reuse,
            "l1": l1,
            "nz": nz,
            "s1_base": base,
            "s2_base": ak if family in ("b", "b_drop") else min(128, ak),
            "n_ratio": ratio,
            "s1_outer": outer,
            "total_tasks": total,
            "split_factor": split,
            "core_num": cores,
        }
        s.update(policy="NORMAL")
        if p["layout"] == "TND":
            s["l1"] = s["s1_base"] <= 128 and max(p["D"], p["DV"]) <= 128
            s["nz"] = True
            s["implementation"] = "TND_GENERAL_L1" if s["l1"] else "TND_GENERAL_GM"
        else:
            s["implementation"] = (
                "B_BATCH"
                if s["family"] in ("b", "b_drop")
                else "S1" if s["family"] == "s1" else "GENERAL"
            )
            s["implementation"] += "_NZ" if s["nz"] else "_ND"
            s["implementation"] += (
                "_L1" if s["l1"] else "_REUSE" if s["reuse"] else "_GM"
            )
    s["drop_adapter"] = bool(
        p["has_drop_mask"] and (p["S2"] % 8 or p["layout"] == "TND" and p["B"] > 1)
    )
    return s


def cann_root():
    choices = [
        os.environ.get(name, "")
        for name in (
            "ASCEND_CANN_PACKAGE_PATH",
            "ASCEND_HOME_PATH",
            "ASCEND_TOOLKIT_HOME",
        )
    ]
    choices += [
        "/usr/local/Ascend/cann-8.5.0",
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/cann-8.5.2",
    ]
    for raw in choices:
        p = Path(raw)
        headers = any(
            (p / prefix / "tiling/matmul/matmul_tiling.h").is_file()
            for prefix in ("include", "include/ascendc/host_api")
        )
        library = any(
            (p / "lib64" / name).is_file()
            for name in ("libtiling_api.so", "libtiling_api.a")
        )
        if raw and headers and library:
            return p.resolve()
    raise RuntimeError("CANN Host Matmul tiling API headers/library are missing")


def mm_requests(p, s, hardware):
    align = cost.align
    base, s2base = s["s1_base"], s["s2_base"]
    same, batch = s["same_ab"], s["family"] in ("b", "b_drop")
    window = (
        min(s2base * s["n_ratio"], p["S2"])
        if same or s["family"] in ("general", "varlen")
        else p["S2"]
    )
    qstride = p["D"] * (
        1
        if p["layout"] == "BNSD"
        else p["N1"] * (p["B"] if p["layout"] == "SBH" else 1)
    )
    kstride = p["D"] * (
        1
        if p["layout"] == "BNSD"
        else p["N2"] * (p["B"] if p["layout"] == "SBH" else 1)
    )
    vstride = kstride // p["D"] * p["DV"]
    l1 = int(hardware["l1_bytes"]) - (
        align(p["S2"]) * align(p["D"]) * 4 if s["family"] == "s1" and s["l1"] else 0
    )
    requests = []
    for stage in (1, 2):
        m, n, k = (
            min(base, p["S1"]),
            window if stage == 1 else p["DV"],
            p["D"] if stage == 1 else window,
        )
        om, on, ka, kb = p["S1"], s2base * s["n_ratio"], qstride, kstride
        fm = fn = fk = -1
        if stage == 2:
            on, ka, kb = vstride, window, vstride
        if batch:
            m = p["S1"]
            om, on = p["S1"], p["S2"] if stage == 1 else p["DV"]
            ka, kb = (qstride, kstride) if stage == 1 else (p["S2"], p["DV"])
            fm, fn = base, (
                s2base
                if stage == 1
                else min(align(p["DV"]), (131072 // (base * 4)) // 16 * 16)
            )
        elif s["family"] == "s1":
            om, on, ka, kb = (
                (p["S1"], p["S2"], qstride, kstride)
                if stage == 1
                else (p["S1"], vstride, p["S2"], vstride)
            )
            fm, fn = min(128, align(p["S1"])), min(
                128, align(p["S2"] if stage == 1 else p["DV"])
            )
            if stage == 2 and p["S1"] == p["S2"] and p["S2"] >= 928 and p["D"] == 64:
                fm, fn, fk = 64, 64, 256
        elif p["layout"] == "TND":
            if stage == 2 and s["l1"]:
                fm, fn = align(min(base, p["S1"])), -1 if same else align(p["DV"])
        elif stage == 1 and (64 < p["D"] <= 128 or s["policy"] == "UNSPLITK"):
            fm, fn, fk = (
                min(128 if same else base, align(p["S1"])),
                128,
                align(p["D"]) if same else p["D"],
            )
        elif stage == 2:
            if same and (64 < p["DV"] <= 128 or s["policy"] == "UNSPLITK"):
                fm, fn = min(128, align(p["S1"])), align(p["DV"])
            if p["DV"] == 64 and p["layout"] == "BNSD" and m >= 64:
                fm, fn, fk = 64, 64, 256
        requests.append(
            (
                m,
                n,
                k,
                om,
                on,
                ka,
                kb,
                fm,
                fn,
                fk,
                l1,
                131072,
                int(hardware["ub_bytes"]),
                2 if p["dtype"] == "BF16" else 0,
                int(batch and stage == 2),
                int(s["family"] == "s1" and stage == 2),
                p["B"],
                p["S1"],
                p["S2"],
                p["N2"],
                p["N1"] // p["N2"],
                stage,
                int(batch),
                int(same),
                int(s["policy"] == "UNSPLITK"),
            )
        )
    return requests


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


def adapter_jobs(p, s, hardware):
    if not s["drop_adapter"]:
        return []
    count = p["B"] * p["N1"] * p["S1"] * p["S2"]
    aligned = cost.align(count, 2048)
    single = (
        cost.ceil_div(cost.ceil_div(count, 2048), int(hardware["aiv_cores"])) * 2048
    )
    base = min(single, cost.ceil_div(int(hardware["ub_bytes"]), 6) // 2048 * 2048)
    factor, total = cost.ceil_div(single, base), cost.ceil_div(aligned, base)
    jobs = []
    for core in range(int(hardware["aiv_cores"])):
        chunks = max(0, min(factor, total - core * factor))
        elements = max(0, min(chunks * base, count - core * factor * base))
        if elements:
            jobs.append(
                {
                    "unpack_k": elements / 1000,
                    "chunks": chunks,
                    "calls": chunks,
                    "read_kb": cost.ceil_div(elements, 8) / 1024,
                    "write_kb": elements / 1024,
                    "loads": chunks,
                    "stores": chunks,
                }
            )
    return jobs


@lru_cache(maxsize=1)
def graph_engine():
    source = ROOT / "native" / "fa_latency_eval.cpp"
    signature = cost.sha256(
        {"source": source.read_text(), "machine": os.uname().machine}
    )
    folder = ROOT / "out" / "fa_latency_host"
    folder.mkdir(parents=True, exist_ok=True)
    library, marker = folder / "latency_eval.so", folder / "latency_eval_signature"
    if not library.is_file() or not marker.is_file() or marker.read_text() != signature:
        result = subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(library) + ".tmp",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError(
                "CPU latency evaluator build failed:\n" + result.stdout[-2000:]
            )
        Path(str(library) + ".tmp").replace(library)
        marker.write_text(signature)
    engine = ctypes.CDLL(str(library))
    engine.fa_latency_order.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    engine.fa_latency_order.restype = ctypes.c_int
    engine.fa_latency_eval.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 6
    engine.fa_latency_eval.restype = ctypes.c_double
    return engine


def graph_time(graph, rates, gradient=False):
    coefficients = np.ascontiguousarray(rates, dtype=np.float64)
    derivative = np.empty(len(coefficients), dtype=np.float64) if gradient else None
    value = graph_engine().fa_latency_eval(
        len(graph["a"]),
        len(graph["templates"]),
        len(coefficients),
        graph["root"],
        graph["a"].ctypes.data,
        graph["b"].ctypes.data,
        graph["which"].ctypes.data,
        graph["templates"].ctypes.data,
        coefficients.ctypes.data,
        derivative.ctypes.data if gradient else None,
    )
    if not math.isfinite(value):
        raise RuntimeError("source timing graph produced a non-finite duration")
    return (value, derivative) if gradient else value


def evaluate(graph, rates):
    durations = graph["templates"] @ np.asarray(rates)
    extra = np.zeros((len(durations), len(PHASES)))
    for index, phase in enumerate(graph["phases"]):
        extra[index, PHASES.index(phase)] = durations[index]
    encoded = {**graph, "templates": np.column_stack((graph["templates"], extra))}
    value, derivative = graph_time(
        encoded, np.append(rates, np.zeros(len(PHASES))), True
    )
    result = dict(zip(PHASES, map(float, derivative[-len(PHASES) :])))
    if not math.isclose(sum(result.values()), value, rel_tol=1e-9, abs_tol=1e-8):
        raise RuntimeError(
            "phase critical-path contributions do not sum to kernel latency"
        )
    return value, derivative[: len(rates)], result
