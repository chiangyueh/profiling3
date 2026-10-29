from __future__ import annotations

import csv
import ctypes
import gc
import hashlib
import heapq
import inspect
import itertools
import json
import math
import os
import re
import subprocess
import statistics
from datetime import datetime, timezone
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import numpy as np

import fa_latency_inputs as cost
import fit_fa_latency as legacy


VERSION = "fa_source_pipeline_v5"
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "fa_source_model.json"
SOURCE = "attention/flash_attention_score/"
PIPES = {
    "cube": ("aic_cube_active_sum_us", ("flops_m", "tiles", "output_tiles")),
    "mte1": ("aic_mte1_active_sum_us", ("l0_kb", "tiles")),
    "mte2": ("aic_mte2_active_sum_us", ("read_kb", "loads", "address_gap_kb")),
    "fixpipe": ("aic_fixpipe_active_sum_us", ("nd_write_kb", "nz_write_kb", "stores", "calls")),
    "vector": ("aiv_vec_active_sum_us", ("softmax_k", "reduction_k", "cast_k", "nz2nd_k", "nd2nz_k", "online_k", "rescale_k", "normalize_k", "drop_k", "unpack_k", "chunks")),
    "vmte2": ("aiv_mte2_active_sum_us", ("read_kb", "alignment_read_kb", "loads")),
    "vmte3": ("aiv_mte3_active_sum_us", ("write_kb", "stores", "address_gap_kb")),
    "cscalar": ("aic_scalar_active_sum_us", ("workers", "batch_entries", "commands")),
    "vscalar": ("aiv_scalar_active_sum_us", ("workers", "batch_entries", "chunks", "drop_chunks", "adapter_workers")),
}
PHASES = ("startup", "drop_adapter", "bmm1", "vec1", "bmm2", "vec2")


def source_hash():
    functions = (schedule, mm_requests, matmul_geometry, vector_job, cube_job, adapter_jobs, reuse_fake_pairs, workload, features, atomic_features, cost_vertices, service, execution_class, modeled_path, cost_group, phase_features, parameter_names, graph_engine, timing_graph, graph_time, critical_phases, elapsed, adapter, predict, predict_design)
    return cost.sha256({"version": VERSION, "pipes": PIPES, "phases": PHASES, "python": [inspect.getsource(function) for function in functions],
                        "host": (ROOT / "fa_matmul_tiling.cpp").read_text(),
                        "graph": (ROOT / "fa_latency_eval.cpp").read_text(),
                        "sameab_geometry": inspect.getsource(legacy.sameab_workload)})


def schedule(inputs, hardware):
    p = cost.normalize(inputs)
    if p["dtype"] not in ("FP16", "BF16") or p["layout"] not in ("BNSD", "SBH", "BSND", "TND"):
        raise ValueError("source model supports dense FP16/BF16 BNSD/SBH/BSND and uniform-length TND")
    if int(hardware["aiv_cores"]) != 2 * int(hardware["aic_cores"]):
        raise ValueError("source model requires two Vector cores per Cube")
    try:
        work = legacy.sameab_workload(p, hardware)
        s = dict(work["schedule"])
        s.update(reuse=False, l1=work["l1_carry"], nz=work["bmm1_template_format"] == "NZ",
                 implementation=work["implementation"], policy=work["matmul_policy"])
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
            best = 128 if b * q <= 10000 or k >= 1024 else 256 if b >= 5 or k >= 512 else 512
            base, ratio = min(best, aq), min(8, ceil(k, min(128, ak)))
        elif k > 1024:
            family, priority = "general", 96
            base = min(128 if b * n * (n // nk) * ceil(q, min(64, aq)) > aiv else 64, aq)
            ratio = min(5 if 1025 <= k <= 1040 else 6 if d == 64 and p["layout"] == "BNSD" and 2049 <= k <= 2304 else 8, ceil(k, min(128, ak)))
            reuse = d <= 128 and not (d == 64 and p["layout"] == "BNSD" and k % 256 and k not in (4032, 42192))
            nz = bool(k % 64 and d != 64)
        else:
            bytes1, bytes2 = n * ((aq + ak) * ad + ak) * 2, n * (aq + ad) * ak * 2
            is_b = ak <= 128 and bytes1 < int(hardware["l1_bytes"]) and bytes2 < int(hardware["l1_bytes"]) and n * aq * ak <= 65536
            if is_b:
                family, priority = ("b_drop" if p["has_drop_mask"] else "b"), 98
                base, ratio = min(aq, 256, 8192 // ak // 16 * 16), 0
            else:
                family, priority = "s1", 97
                initial = min(128, aq)
                l1 = ad == 64 and 512 <= ak <= 1024 and q <= 3840
                reuse = not l1 and ((k >= 512 and b * n > 64 and d <= 128) or (b * n >= 4800 and k == 256 and d == 144))
                nz = bool(k % 64 or k == 64 or k >= 704 and k % 128 and b * n * q > 20480)
                maximum = 1 if l1 and ak == 512 or ak < 128 else 4
                if nz and d <= 256:
                    maximum = 4
                ratio, total = 1, b * n * ceil(q, initial)
                while ratio < maximum and total > (ratio - 1) * aiv // ratio and initial * ratio < aq and initial * ratio * ak <= 131072:
                    ratio += 1
                    total = b * n * ceil(ceil(q, initial), ratio)
                while total <= (ratio - 1) * aiv // ratio or initial * ratio > aq or initial * ratio * ak > 131072:
                    ratio -= 1
                    if ratio <= 0:
                        raise ValueError("official S1 block geometry is not representable")
                base = initial * ratio
        outer = ceil(q, base)
        total = b if family in ("b", "b_drop") else b * n * outer
        split = ceil(total, min(total, aiv)) * (2 if reuse else 1)
        cores = min(total, aiv)
        if family in ("s1", "b", "b_drop"):
            cores = min(aiv, ceil(total, split // (2 if reuse else 1)))
            if reuse and cores % 2:
                cores = min(aiv, cores + 1)
        s = {"family": family, "priority": priority, "same_ab": False, "reuse": reuse, "l1": l1, "nz": nz,
             "s1_base": base, "s2_base": ak if family in ("b", "b_drop") else min(128, ak), "n_ratio": ratio,
             "s1_outer": outer, "total_tasks": total, "split_factor": split, "core_num": cores}
        s.update(policy="NORMAL")
        if p["layout"] == "TND":
            s["l1"] = s["s1_base"] <= 128 and max(p["D"], p["DV"]) <= 128
            s["nz"] = True
            s["implementation"] = "TND_GENERAL_L1" if s["l1"] else "TND_GENERAL_GM"
        else:
            s["implementation"] = ("B_BATCH" if s["family"] in ("b", "b_drop") else "S1" if s["family"] == "s1" else "GENERAL")
            s["implementation"] += "_NZ" if s["nz"] else "_ND"
            s["implementation"] += "_L1" if s["l1"] else "_REUSE" if s["reuse"] else "_GM"
    s["drop_adapter"] = bool(p["has_drop_mask"] and (p["S2"] % 8 or p["layout"] == "TND" and p["B"] > 1))
    return s


def cann_root():
    choices = [os.environ.get(name, "") for name in ("ASCEND_CANN_PACKAGE_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")]
    choices += ["/usr/local/Ascend/cann-8.5.0", "/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/cann-8.5.2"]
    for raw in choices:
        p = Path(raw)
        headers = any((p / prefix / "tiling/matmul/matmul_tiling.h").is_file() for prefix in ("include", "include/ascendc/host_api"))
        library = any((p / "lib64" / name).is_file() for name in ("libtiling_api.so", "libtiling_api.a"))
        if raw and headers and library:
            return p.resolve()
    raise RuntimeError("CANN Host Matmul tiling API headers/library are missing")


@lru_cache(maxsize=1)
def tiler():
    cann = cann_root()
    source = ROOT / "fa_matmul_tiling.cpp"
    library = next(cann / "lib64" / name for name in ("libtiling_api.so", "libtiling_api.a") if (cann / "lib64" / name).is_file())
    signature = cost.sha256({"source": source.read_text(), "cann": str(cann),
                             "api": hashlib.sha256(library.read_bytes()).hexdigest()})
    folder = ROOT / "out" / "fa_latency_host"
    folder.mkdir(parents=True, exist_ok=True)
    binary, marker = folder / "matmul_tiling", folder / "signature"
    if not binary.is_file() or not marker.is_file() or marker.read_text() != signature:
        command = ["g++", "-O0", "-std=c++17", str(source), "-I", str(cann / "include"),
                   "-I", str(cann / "include/ascendc/host_api"), "-L", str(cann / "lib64"),
                   "-Wl,-rpath," + str(cann / "lib64"), "-ltiling_api", "-lplatform", "-lregister", "-lascendalog",
                   "-lgraph_base", "-ldl", "-lc_sec", "-o", str(binary) + ".tmp"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("CPU Matmul tiling helper build failed:\n" + result.stdout[-4000:])
        Path(str(binary) + ".tmp").replace(binary)
        marker.write_text(signature)
    return str(binary), signature


def mm_requests(p, s, hardware):
    align = cost.align
    base, s2base = s["s1_base"], s["s2_base"]
    same, batch = s["same_ab"], s["family"] in ("b", "b_drop")
    window = min(s2base * s["n_ratio"], p["S2"]) if same or s["family"] in ("general", "varlen") else p["S2"]
    qstride = p["D"] * (1 if p["layout"] == "BNSD" else p["N1"] * (p["B"] if p["layout"] == "SBH" else 1))
    kstride = p["D"] * (1 if p["layout"] == "BNSD" else p["N2"] * (p["B"] if p["layout"] == "SBH" else 1))
    vstride = kstride // p["D"] * p["DV"]
    l1 = int(hardware["l1_bytes"]) - (align(p["S2"]) * align(p["D"]) * 4 if s["family"] == "s1" and s["l1"] else 0)
    requests = []
    for stage in (1, 2):
        m, n, k = min(base, p["S1"]), window if stage == 1 else p["DV"], p["D"] if stage == 1 else window
        om, on, ka, kb = p["S1"], s2base * s["n_ratio"], qstride, kstride
        fm = fn = fk = -1
        if stage == 2:
            on, ka, kb = vstride, window, vstride
        if batch:
            m = p["S1"]
            om, on = p["S1"], p["S2"] if stage == 1 else p["DV"]
            ka, kb = (qstride, kstride) if stage == 1 else (p["S2"], p["DV"])
            fm, fn = base, s2base if stage == 1 else min(align(p["DV"]), (131072 // (base * 4)) // 16 * 16)
        elif s["family"] == "s1":
            om, on, ka, kb = (p["S1"], p["S2"], qstride, kstride) if stage == 1 else (p["S1"], vstride, p["S2"], vstride)
            fm, fn = min(128, align(p["S1"])), min(128, align(p["S2"] if stage == 1 else p["DV"]))
            if stage == 2 and p["S1"] == p["S2"] and p["S2"] >= 928 and p["D"] == 64:
                fm, fn, fk = 64, 64, 256
        elif p["layout"] == "TND":
            if stage == 2 and s["l1"]:
                fm, fn = align(min(base, p["S1"])), -1 if same else align(p["DV"])
        elif stage == 1 and (64 < p["D"] <= 128 or s["policy"] == "UNSPLITK"):
            fm, fn, fk = min(128 if same else base, align(p["S1"])), 128, align(p["D"]) if same else p["D"]
        elif stage == 2:
            if same and (64 < p["DV"] <= 128 or s["policy"] == "UNSPLITK"):
                fm, fn = min(128, align(p["S1"])), align(p["DV"])
            if p["DV"] == 64 and p["layout"] == "BNSD" and m >= 64:
                fm, fn, fk = 64, 64, 256
        requests.append((m, n, k, om, on, ka, kb, fm, fn, fk, l1, 131072, int(hardware["ub_bytes"]),
                         2 if p["dtype"] == "BF16" else 0, int(batch and stage == 2), int(s["family"] == "s1" and stage == 2),
                         p["B"], p["S1"], p["S2"], p["N2"], p["N1"] // p["N2"], stage, int(batch), int(same), int(s["policy"] == "UNSPLITK")))
    return requests


@lru_cache(maxsize=8192)
def matmul_geometry(requests):
    binary, signature = tiler()
    result = subprocess.run([binary], input="\n".join(" ".join(map(str, r)) for r in requests) + "\n",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
    lines = result.stdout.strip().splitlines()
    if result.returncode or len(lines) != 2 or any(len(line.split()) != 8 for line in lines):
        raise RuntimeError("official CPU Matmul tiler rejected inputs: " + (result.stdout + result.stderr)[-1000:])
    names = ("base_m", "base_n", "base_k", "depth_a1", "depth_b1", "db_l0c", "step_m", "step_n")
    return [dict(zip(names, map(int, line.split()))) for line in lines]


def vector_job(p, s, rows, columns, first, last):
    align, ceil = cost.align, cost.ceil_div
    batch = s["family"] in ("b", "b_drop")
    windowed = s["same_ab"] or s["family"] in ("general", "varlen")
    padded = align(columns, 64) if s["family"] == "s1" or s["same_ab"] else align(columns)
    v1base = min(rows, s["s1_base"] if batch else max(1, 1024 // padded * 8))
    if p["layout"] == "TND" and columns < 8:
        v1base = rows
    if windowed:
        v2base = max(1, 1024 // align(p["D"]) * 8) if align(p["D"]) > 64 else min(128, rows) if s["same_ab"] or p["layout"] == "TND" else rows
    else:
        v2base = min(align(p["S1"]), max(1, 8192 // align(p["D"]) // 16 * 16 if align(p["D"]) <= 512 else 8192 // align(p["D"])))
    c1, c2 = ceil(rows, v1base), ceil(rows, v2base)
    soft_rows = sum(align(min(v1base, rows - pos), 8) for pos in range(0, rows, v1base))
    soft_elements = soft_rows * padded
    nz1 = s["nz"] and (s["family"] == "s1" or columns % 64 != 0)
    probability_nz = s["same_ab"] or (not batch and s["nz"])
    nz2 = s["same_ab"] or p["layout"] == "TND" or (s["family"] in ("s1", "b", "b_drop") and p["DV"] % 16 != 0)
    width = align(p["DV"])
    stats = rows * 32
    drop_bytes = rows * (align(columns, 32) if s["drop_adapter"] else ceil(align(columns, 32), 8)) if p["has_drop_mask"] else 0
    stage1 = {"softmax_k": soft_elements / 1000, "reduction_k": soft_rows * math.ceil(math.log2(max(2, padded))) / 1000,
              "cast_k": rows * padded / 1000, "format_k": rows * padded * (int(nz1) + int(probability_nz)) / 1000,
              "online_k": (0 if first else rows * 8) / 1000, "drop_k": rows * padded / 1000 if p["has_drop_mask"] else 0,
              "chunks": c1, "calls": 1, "read_kb": (4 * rows * align(columns) + drop_bytes) / 1024,
              "write_kb": (2 * rows * align(columns) + (stats if last else 0) if windowed else 2 * rows * align(columns) + 2 * stats) / 1024,
              "loads": c1 * (1 + int(bool(p["has_drop_mask"]))), "stores": c1 + int(last) if windowed else c1 * 3,
              "vec1_calls": c1, "softmax_initial_calls": c1 if first else 0, "softmax_update_calls": 0 if first else c1}
    stage1.update(nz2nd_k=rows * padded * int(nz1) / 1000, nd2nz_k=rows * padded * int(probability_nz) / 1000,
                  drop_chunks=c1 if p["has_drop_mask"] else 0,
                  alignment_read_kb=4 * rows * (padded - align(columns)) / 1024 if nz1 else 0)
    output_elements = rows * width
    stage2 = {"softmax_k": 0, "reduction_k": 0, "cast_k": output_elements / 1000 if last else 0,
              "format_k": output_elements / 1000 if nz2 else 0, "online_k": 0,
              "rescale_k": output_elements * int(not first) * 2 / 1000,
              "normalize_k": output_elements * int(last) / 1000,
              "drop_k": 0, "chunks": c2, "calls": 1,
              "read_kb": (4 * output_elements + (0 if windowed else stats) + (0 if first else 4 * output_elements)) / 1024,
              "write_kb": ((2 * rows * p["DV"] if last else 4 * output_elements) + (stats if windowed and last else 0)) / 1024,
              "loads": c2 * (1 + int(not windowed) + int(not first)), "stores": c2 + ((1 if s["same_ab"] else c2) if windowed and last else 0),
              "vec2_calls": c2, "accumulator_read_bytes": 0 if first else 4 * output_elements,
              "accumulator_write_bytes": 0 if last else 4 * output_elements,
              "final_output_bytes": 2 * rows * p["DV"] if last else 0}
    output_stride = p["DV"] * (p["N1"] * (p["B"] if p["layout"] == "SBH" else 1) if p["layout"] != "BNSD" else 1)
    stage2["address_gap_kb"] = max(0, (rows - 1) * (output_stride - p["DV"]) * 2) / 1024 if last else 0
    stage2.update(nz2nd_k=output_elements / 1000 if nz2 else 0, nd2nz_k=0)
    return stage1, stage2


def cube_job(p, s, geometry, stage, rows, columns, heads, kv_heads, shared, carried, q_carried=False):
    align, ceil = cost.align, cost.ceil_div
    m, n, k = rows, columns if stage == 0 else p["DV"], p["D"] if stage == 0 else columns
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
    kvstride = (p["D"] if stage == 0 else p["DV"]) * (p["N2"] * factor if p["layout"] != "BNSD" else 1)
    gap = heads * max(0, m - 1) * max(0, qstride - p["D"]) * 2 if stage == 0 and not q_carried else 0
    if not shared and not carried:
        gap += kv_heads * max(0, columns - 1) * max(0, kvstride - (p["D"] if stage == 0 else p["DV"])) * 2
    output_width = 4
    nz = (s["nz"] and (s["family"] == "s1" or columns % 64 != 0)) if stage == 0 else (
        s["same_ab"] or p["layout"] == "TND" or s["family"] in ("s1", "b", "b_drop") and p["DV"] % 16 != 0)
    write_kb = heads * m * align(n) * output_width / 1024
    loads = heads * ceil(mt * kt, geometry["depth_a1"]) + (0 if shared or carried else kv_heads * ceil(nt * kt, geometry["depth_b1"]))
    calls, k_slices = 1, []
    if p["layout"] == "TND" and s["same_ab"] and s["l1"]:
        if stage == 0:
            calls = ceil(columns, 256)
            loads = (0 if q_carried else heads) + kv_heads * calls
        else:
            parts, nloops = [min(192, columns - start) for start in range(0, columns, 192)], ceil(p["DV"], 128)
            total_k = sum(align(part) for part in parts)
            k_slices = [align(part) / total_k for part in parts]
            a_bytes *= nloops
            calls = nloops * (len(parts) + 1)
            loads = (heads + kv_heads) * nloops * len(parts)
    k_parts = [align(min(bk, k - start)) for start in range(0, k, bk)] if s['policy'] == 'NORMAL' else []
    return {"flops_m": 2 * heads * align(m) * align(n) * align(k) / 1e6, "tiles": tiles,
            "output_tiles": heads * mt * nt,
            "l0_kb": heads * 2 * (align(m) * align(k) * nt + align(n) * align(k) * mt) / 1024,
            "read_kb": (a_bytes + b_bytes) / 1024,
            "address_gap_kb": gap / 1024,
            "write_kb": write_kb, "nd_write_kb": 0 if nz else write_kb, "nz_write_kb": write_kb if nz else 0,
            "loads": loads, "stores": heads * mt * nt, "calls": calls, "commands": calls + tiles,
            "k_slices": k_slices,
            "pipeline_k_slices": [part / sum(k_parts) for part in k_parts] if len(k_parts) > 1 else [],
            "pipeline_k_buffers": min(max(1, geometry['depth_a1'] // max(1, geometry.get('step_m', 1))),
                                      max(1, geometry['depth_b1'] // max(1, geometry.get('step_n', 1)))),
            "query_rows": heads * rows}


def adapter_jobs(p, s, hardware):
    if not s["drop_adapter"]:
        return []
    count = p["B"] * p["N1"] * p["S1"] * p["S2"]
    aligned = cost.align(count, 2048)
    single = cost.ceil_div(cost.ceil_div(count, 2048), int(hardware["aiv_cores"])) * 2048
    base = min(single, cost.ceil_div(int(hardware["ub_bytes"]), 6) // 2048 * 2048)
    factor, total = cost.ceil_div(single, base), cost.ceil_div(aligned, base)
    jobs = []
    for core in range(int(hardware["aiv_cores"])):
        chunks = max(0, min(factor, total - core * factor))
        elements = max(0, min(chunks * base, count - core * factor * base))
        if elements:
            jobs.append({"unpack_k": elements / 1000, "chunks": chunks, "calls": chunks,
                         "read_kb": cost.ceil_div(elements, 8) / 1024, "write_kb": elements / 1024,
                         "loads": chunks, "stores": chunks})
    return jobs


def reuse_fake_pairs(p, s):
    if not s["reuse"]:
        return 0
    if s["family"] == "s1":
        return s["total_tasks"] % 2
    count, outer = 0, s["s1_outer"]
    for start in range(0, s["total_tasks"], s["split_factor"]):
        end = min(start + s["split_factor"], s["total_tasks"])
        for head in range(start // outer, (end - 1) // outer + 1):
            count += (min(end, (head + 1) * outer) - max(start, head * outer)) % 2
    return count


def workload(inputs, hardware, calibration_geometry=None, calibration_schedule=None):
    p = cost.normalize(inputs)
    s = schedule(p, hardware)
    if calibration_schedule is not None:
        s.update(calibration_schedule)
        s["s1_outer"] = cost.ceil_div(p["S1"], s["s1_base"])
        if p["layout"] == "TND" and s["same_ab"]:
            s["l1"] = p["D"] in (80, 88, 128, 192) and s["s1_base"] <= 256
    if calibration_geometry is None:
        mm = matmul_geometry(tuple(mm_requests(p, s, hardware)))
    elif any("db_l0c" not in geometry for geometry in calibration_geometry):
        inferred = matmul_geometry(tuple(mm_requests(p, s, hardware)))
        mm = [{**inferred[index], **geometry} for index, geometry in enumerate(calibration_geometry)]
    else:
        mm = calibration_geometry
    static = p["dtype"] == "BF16" and not p["has_drop_mask"] and p["layout"] in ("BSND", "SBH") and not s["same_ab"] and s["family"] in ("b", "s1")
    static = static and mm[0]["base_m"] == mm[1]["base_m"] and mm[0]["base_n"] == mm[1]["base_k"] and mm[0]["base_k"] == mm[1]["base_n"]
    static = static and ((mm[0]["base_m"], mm[0]["base_n"], mm[0]["base_k"]) in ((16, 16, 80), (32, 32, 80), (48, 48, 80), (64, 64, 80), (16, 16, 96), (32, 32, 96)) if s["family"] == "b" else mm[0]["base_m"] == 128 and mm[0]["base_n"] == 128 and mm[0]["base_k"] in (80, 96, 128))
    static = static and (p["layout"] == "BSND" or s["family"] == "s1" and mm[0]["base_k"] == 96)
    s["static_matmul"] = bool(static)
    if static:
        s["implementation"] += "_STATIC"
    batch = s["family"] in ("b", "b_drop")
    windowed = s["same_ab"] or s["family"] in ("general", "varlen")
    window = s["s2_base"] * s["n_ratio"] if windowed else p["S2"]
    workers, jobs, totals = [], [], Counter()
    cube_cache, vector_cache, unit_cache, unit_indices, unit_counts = {}, {}, {}, {}, Counter()
    for start in range(0, s["total_tasks"], s["split_factor"]):
        clients = [[], []] if s["reuse"] else [[]]
        previous = {}
        for task in range(start, min(start + s["split_factor"], s["total_tasks"])):
            rows = p["S1"] if batch else min(s["s1_base"], p["S1"] - task % s["s1_outer"] * s["s1_base"])
            head = 0 if batch else task // s["s1_outer"] % p["N1"]
            batch_id = task if batch else task // (s["s1_outer"] * p["N1"])
            client = (task - start) % 2 if s["reuse"] else 0
            if s["reuse"] and s["family"] == "general":
                client = (task - max(start, task // s["s1_outer"] * s["s1_outer"])) % 2
            heads, kv_heads = (p["N1"], p["N2"]) if batch else (1, 1)
            for offset in range(0, p["S2"], window):
                columns = min(window, p["S2"] - offset)
                key = batch_id, head // (p["N1"] // p["N2"]), offset
                cube = []
                cube_keys = []
                for stage in (0, 2):
                    shared = s["reuse"] and client == 1 and previous.get(("shared", stage, offset)) == key
                    carry_api = s["l1"] and (stage == 2 or p["layout"] == "TND")
                    carried = carry_api and previous.get(("carried", stage, client)) == key
                    q_carried = p["layout"] == "TND" and s["l1"] and stage == 0 and offset > 0
                    if p["layout"] == "TND" and s["l1"]:
                        carried = False
                    cube_identity = stage, rows, columns, heads, kv_heads, shared, carried, q_carried
                    if cube_identity not in cube_cache:
                        cube_cache[cube_identity] = cube_job(p, s, mm[stage // 2], stage, rows, columns, heads, kv_heads, shared, carried, q_carried)
                    cube.append(cube_cache[cube_identity])
                    cube_keys.append(cube_identity)
                    previous["shared", stage, offset], previous["carried", stage, client] = key, key
                vector = []
                vector_keys = []
                for owned in ((cost.ceil_div(rows, 2), rows // 2) if s["same_ab"] else (rows,)):
                    if owned == 0:
                        vector.append(({}, {}))
                        vector_keys.append((0,))
                        continue
                    vector_identity = owned, columns, offset == 0, offset + columns == p["S2"]
                    if vector_identity not in vector_cache:
                        v1, v2 = vector_job(p, s, owned, columns, offset == 0, offset + columns == p["S2"])
                        if batch:
                            for v in (v1, v2):
                                for name in v:
                                    v[name] *= heads
                        vector_cache[vector_identity] = v1, v2
                    v1, v2 = vector_cache[vector_identity]
                    vector.append((v1, v2))
                    vector_keys.append(vector_identity)
                    for v in (v1, v2):
                        totals.update({name: v.get(name, 0) for name in ("vec1_calls", "vec2_calls", "accumulator_read_bytes", "accumulator_write_bytes", "final_output_bytes")})
                unit_identity = tuple(cube_keys), tuple(vector_keys)
                if unit_identity not in unit_cache:
                    unit_indices[unit_identity] = len(unit_cache)
                    unit_cache[unit_identity] = {"cube": cube, "vector": vector, "columns": columns, "rows": rows}
                unit = unit_cache[unit_identity]
                unit_counts[unit_indices[unit_identity]] += 1
                clients[client].append(unit)
                jobs.append(unit)
        workers.extend(clients)
    cube_pairs = len(workers) if s["same_ab"] else cost.ceil_div(len(workers), 2)
    aiv_workers = 2 * len(workers) if s["same_ab"] else len(workers)
    totals.update(cube_workers=int(hardware["aic_cores"]) if s["drop_adapter"] else cube_pairs,
                  vector_workers=int(hardware["aiv_cores"]) if s["drop_adapter"] else aiv_workers,
                  bmm1_calls=len(jobs), bmm2_calls=len(jobs))
    return {"inputs": p, "schedule": s, "matmul": mm, "workers": workers, "jobs": jobs, "totals": dict(totals),
            "adapter_jobs": adapter_jobs(p, s, hardware), "unpaired_reuse_segments": reuse_fake_pairs(p, s),
            "units": list(unit_cache.values()), "unit_counts": [unit_counts[index] for index in range(len(unit_cache))]}


def features(design, pipe):
    if "pipe_features" in design:
        return design["pipe_features"][pipe]
    result = Counter()
    cube = pipe in ("cube", "mte1", "mte2", "fixpipe", "cscalar")
    units = zip(design["units"], design["unit_counts"]) if "units" in design else ((job, 1) for job in design["jobs"])
    for job, multiplicity in units:
        for item in job["cube"] if cube else (item for pair in job["vector"] for item in pair):
            result.update({name: item.get(name, 0) * multiplicity for name in PIPES[pipe][1] if name != "workers"})
    if not cube:
        for job in design["adapter_jobs"]:
            result.update({name: job.get(name, 0) for name in PIPES[pipe][1]})
    if "workers" in PIPES[pipe][1]:
        result["workers"] = design["totals"]["cube_workers" if cube else "vector_workers"]
        result["batch_entries"] = result["workers"] * design["inputs"]["B"] if design["inputs"]["layout"] == "TND" else 0
    if pipe == "vscalar":
        result["adapter_workers"] = design["totals"]["vector_workers"] if design["schedule"]["drop_adapter"] else 0
    return result


def atomic_features(design, pipe):
    cube = pipe in ("cube", "mte1", "mte2", "fixpipe", "cscalar")
    names = PIPES[pipe][1]
    values = set()
    for job in design.get("units", design["jobs"]):
        for item in job["cube"] if cube else (item for pair in job["vector"] for item in pair):
            values.add(tuple(float(item.get(name, 0)) for name in names))
    if not cube:
        values.update(tuple(float(job.get(name, 0)) for name in names) for job in design["adapter_jobs"])
    if "workers" in names:
        startup = {"workers": 1, "batch_entries": design["inputs"]["B"] if design["inputs"]["layout"] == "TND" else 0,
                   "adapter_workers": int(design["schedule"]["drop_adapter"]) if pipe == "vscalar" else 0}
        values.add(tuple(float(startup.get(name, 0)) for name in names))
    return sorted(values)


def service(job, pipe, model):
    identity = pipe, tuple(job.get(name, 0) for name in PIPES[pipe][1])
    cache = model.setdefault("_service_cache", {})
    if identity in cache:
        return cache[identity]
    if model.get("strict_classes"):
        if not any(job.get(name, 0) for name in PIPES[pipe][1]):
            return 0.0
        fitted = model["pipes"].get(pipe)
        if fitted is None or fitted.get("status") == "unavailable":
            raise ValueError(f"uncalibrated cost: {model['class_id']} {pipe}")
        if fitted.get("status") == "joint_pipeline_calibration":
            if any(job.get(name, 0) for name in fitted["inactive_feature_rates"]):
                raise ValueError(f"unmeasured operation cost: {model['class_id']} {pipe}")
            value = sum(value * job.get(name, 0) for name, value in fitted["coefficients"].items())
            cache[identity] = value
            return value
        if model.get("cost_mode") in ("lower", "upper"):
            if any(job.get(name, 0) for name in fitted["inactive_feature_rates"]):
                raise ValueError(f"unmeasured operation cost: {model['class_id']} {pipe}")
            vector = np.asarray([job.get(name, 0) for name in PIPES[pipe][1]])
            values = np.asarray(fitted["feasible_rate_vertices"]) @ vector
            value = float(np.min(values) if model["cost_mode"] == "lower" else np.max(values))
            cache[identity] = value
            return value
        vector = np.asarray([job.get(name, 0) / scale for name, scale in zip(PIPES[pipe][1], fitted["feature_scales"])])
        basis = np.asarray(fitted["row_basis"])
        projected = vector @ basis.T @ basis
        residual = float(np.linalg.norm(vector - projected) / max(np.linalg.norm(vector), 1e-30))
        if residual > 1e-7:
            raise ValueError(f"unidentified stage cost: {model['class_id']} {pipe}; relative_rowspace_residual={residual:.6g}")
    value = sum(value * job.get(name, 0) for name, value in model["pipes"][pipe]["coefficients"].items())
    cache[identity] = value
    return value


def cost_vertices(pipe, fitted):
    names = PIPES[pipe][1]
    scale, basis = np.asarray(fitted["feature_scales"]), np.asarray(fitted["row_basis"])
    total = basis @ (np.asarray([fitted["coefficients"][name] for name in names]) * scale)
    active = [index for index, name in enumerate(names) if name not in fitted["inactive_feature_rates"]]
    vertices = []
    tolerance = 1e-7 * max(1.0, float(np.linalg.norm(total)))
    for selected in itertools.combinations(active, len(basis)):
        matrix = basis[:, selected]
        if np.linalg.matrix_rank(matrix) < len(basis):
            continue
        solution = np.linalg.solve(matrix, total)
        if float(np.min(solution)) < -tolerance:
            continue
        vector = np.zeros(len(names))
        vector[list(selected)] = np.maximum(solution, 0)
        if float(np.linalg.norm(basis @ vector - total)) <= tolerance:
            vertices.append((vector / scale).tolist())
    if not vertices:
        raise ValueError(f"no feasible nonnegative cost decomposition: {pipe}")
    return vertices


def execution_class(design):
    p, s = design["inputs"], design["schedule"]
    windowed = s["same_ab"] or s["family"] in ("general", "varlen")
    width = s["s2_base"] * s["n_ratio"] if windowed else p["S2"]
    columns = [min(width, p["S2"] - offset) for offset in range(0, p["S2"], width)]
    if s["family"] == "s1":
        formats = ["NZ" if s["nz"] else "ND"]
    else:
        formats = sorted({"NZ" if s["nz"] and count % 64 else "ND" for count in columns})
    mm2_format = "NZ" if s["same_ab"] or p["layout"] == "TND" or s["family"] in ("s1", "b", "b_drop") and p["DV"] % 16 else "ND"
    family = "tnd_same_ab" if s["same_ab"] and p["layout"] == "TND" else "tnd_general" if p["layout"] == "TND" else "b" if s["family"] in ("b", "b_drop") else s["family"]
    q_blocks = [min(s["s1_base"], p["S1"])]
    if p["S1"] % s["s1_base"]:
        q_blocks.append(p["S1"] % s["s1_base"])
    vector_rows = [r for q in q_blocks for r in ((cost.ceil_div(q, 2), q // 2) if s["same_ab"] else (q,)) if r]
    return {"kernel": family, "dtype": p["dtype"], "layout": p["layout"],
            "matmul": "STATIC" if s.get("static_matmul") else s["policy"],
            "mm1_template": "NZ" if s["nz"] else "ND", "mm1_windows": "+".join(formats), "mm2_output": mm2_format,
            "operand_path": "TSCM" if s["policy"] == "UNSPLITK" else "L1_CARRY" if s["l1"] else "IBSHARE" if s["reuse"] else "GM",
            "reuse_fake_peer": bool(design["unpaired_reuse_segments"]),
            "probability_format": "NZ" if s["same_ab"] or s["nz"] else "ND",
            "drop_mask": "BYTE_ADAPTER" if s["drop_adapter"] else "BIT" if p["has_drop_mask"] else "NONE",
            "softmax_reduce": 1 if s["same_ab"] and p["layout"] != "TND" and s["s1_base"] > 256 or p["layout"] == "TND" and not s["same_ab"] and s["s1_base"] > 128 else 8,
            "online_update": bool(windowed and len(columns) > 1),
            "q_tail": bool(p["S1"] % s["s1_base"]), "k_tail_mask": bool(any(count % 64 for count in columns)),
            "softmax_row_padding": bool(any(rows % 8 for rows in vector_rows)),
            "tiny_k": bool(p["layout"] == "TND" and any(count < 8 for count in columns)),
            "d_tail": bool(p["D"] % 16), "dv_tail": bool(p["DV"] % 16)}


def modeled_path(design):
    p, s = design["inputs"], design["schedule"]
    return bool(s["same_ab"])


def validation_inputs(hardware):
    result = []
    for query, key in ((384, 1536), (640, 2176), (1152, 3200), (2304, 3456), (4608, 5504), (8960, 9216)):
        for batch, nq, nkv, dv in ((1, 16, 16, 128), (1, 64, 64, 128), (2, 16, 16, 128),
                                    (1, 24, 8, 128), (1, 16, 16, 96), (1, 16, 16, 136)):
            inputs = cost.normalize(dict(zip(cost.INPUT_FIELDS, (batch, nq, nkv, query, key, 192, dv, "BF16", "BNSD", 0))))
            if not modeled_path({"inputs": inputs, "schedule": schedule(inputs, hardware)}):
                raise RuntimeError("a validation input does not select the modeled SameAB path")
            result.append(inputs)
    return result


def cost_group(design):
    if modeled_path(design):
        path = execution_class(design)
        fields = ("kernel", "dtype", "layout", "matmul", "operand_path", "softmax_reduce")
        return "|".join(f"{name}={path[name]}" for name in fields)
    return "|".join(f"{name}={value}" for name, value in execution_class(design).items())


def phase_features(design):
    result = {stage: {pipe: Counter() for pipe in pipes} for stage, pipes in
              (("bmm1", ("cube", "mte1", "mte2", "fixpipe", "cscalar")),
               ("bmm2", ("cube", "mte1", "mte2", "fixpipe", "cscalar")),
               ("vec1", ("vector", "vmte2", "vmte3", "vscalar")),
               ("vec2", ("vector", "vmte2", "vmte3", "vscalar")))}
    units = zip(design["units"], design["unit_counts"]) if "units" in design else ((job, 1) for job in design["jobs"])
    for job, multiplicity in units:
        for index, stage in enumerate(("bmm1", "bmm2")):
            for pipe in result[stage]:
                result[stage][pipe].update({name: job["cube"][index].get(name, 0) * multiplicity for name in PIPES[pipe][1]})
        for pair in job["vector"]:
            for index, stage in enumerate(("vec1", "vec2")):
                for pipe in result[stage]:
                    result[stage][pipe].update({name: pair[index].get(name, 0) * multiplicity for name in PIPES[pipe][1]})
    batches = design["inputs"]["B"] if design["inputs"]["layout"] == "TND" else 0
    result["startup"] = {"cscalar": {"workers": design["totals"]["cube_workers"], "batch_entries": batches * design["totals"]["cube_workers"]},
                         "vscalar": {"workers": design["totals"]["vector_workers"], "batch_entries": batches * design["totals"]["vector_workers"],
                                     "adapter_workers": design["totals"]["vector_workers"] if design["schedule"]["drop_adapter"] else 0}}
    if design["adapter_jobs"]:
        result["drop_adapter"] = {pipe: Counter() for pipe in ("vector", "vmte2", "vmte3", "vscalar")}
        for job in design["adapter_jobs"]:
            for pipe in result["drop_adapter"]:
                result["drop_adapter"][pipe].update({name: job.get(name, 0) for name in PIPES[pipe][1]})
    return result


def source_evidence():
    root = Path(os.environ.get("OPS_TRANSFORMER_ROOT", str(ROOT.parent / "ops-transformer-official-8.5.0")))
    kernels = {"b": ["flash_attention_score_bn2gs1s2_b.h"], "s1": ["flash_attention_score_s1_bn2gs1.h"],
               "general": ["flash_attention_score_s1s2_bn2gs1.h"], "same_ab": ["flash_attention_score_s1s2_bn2gs1_sab.h"],
               "tnd_general": ["flash_attention_var_len_score.h", "flash_attention_score_s1s2_bn2gs1.h"],
               "tnd_same_ab": ["flash_attention_var_len_score_sab.h", "flash_attention_score_s1s2_bn2gs1_sab.h"]}
    files = {SOURCE + "op_kernel/arch32/" + name for names in kernels.values() for name in names}
    files.update(SOURCE + suffix for suffix in ("op_host/arch32/flash_attention_score_tiling_general.cpp",
                 "op_kernel/arch32/flash_attention_score_template_tiling_key.h", "op_kernel/arch32/flash_attention_score_drop_mask_adapter.h"))
    return {"kernel_files": {name: [SOURCE + "op_kernel/arch32/" + item for item in names] for name, names in kernels.items()},
            "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(files)}}


def parameter_names():
    return tuple((pipe, name) for pipe, (_, names) in PIPES.items() for name in names)


@lru_cache(maxsize=1)
def graph_engine():
    source = ROOT / "fa_latency_eval.cpp"
    signature = cost.sha256({"source": source.read_text(), "machine": os.uname().machine})
    folder = ROOT / "out" / "fa_latency_host"
    folder.mkdir(parents=True, exist_ok=True)
    library, marker = folder / "latency_eval.so", folder / "latency_eval_signature"
    if not library.is_file() or not marker.is_file() or marker.read_text() != signature:
        result = subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library) + ".tmp"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("CPU latency evaluator build failed:\n" + result.stdout[-2000:])
        Path(str(library) + ".tmp").replace(library)
        marker.write_text(signature)
    engine = ctypes.CDLL(str(library))
    engine.fa_latency_order.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    engine.fa_latency_order.restype = ctypes.c_int
    engine.fa_latency_eval.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 6
    engine.fa_latency_eval.restype = ctypes.c_double
    return engine


def timing_graph(design, finite_buffers=True):
    if not design["schedule"]["same_ab"]:
        raise ValueError("the compiled timing graph supports SameAB implementations")
    names = parameter_names()
    dimensions = len(names)
    columns = {pair: index for index, pair in enumerate(names)}
    templates, template_phases = [tuple([0.0] * dimensions)], ["startup"]
    template_ids = {("startup", tuple([0.0] * dimensions)): 0}
    left, right, labels = [], [], []

    def template(pipe, job, divisor=1, phase="startup"):
        values = [0.0] * dimensions
        for name in PIPES[pipe][1]:
            values[columns[pipe, name]] = float(job.get(name, 0)) / divisor
        identity = phase, tuple(values)
        if identity not in template_ids:
            template_ids[identity] = len(templates)
            templates.append(tuple(values))
            template_phases.append(phase)
        return template_ids[identity]

    def node(preds=(), label=0):
        preds = list(dict.fromkeys(p for p in preds if p >= 0))
        while len(preds) > 2:
            preds = [node(preds[:2])] + preds[2:]
        index = len(left)
        left.append(preds[0] if preds else -1)
        right.append(preds[1] if len(preds) > 1 else -1)
        labels.append(label)
        return index

    def resolve(index, preds):
        dependency = node(preds)
        left[index], right[index] = dependency, -1

    def flow(job, pipes, count, launch, tails, depth=0, phase="startup"):
        encoded = {pipe: template(pipe, job, count, phase) for pipe in pipes}
        fixed = []
        end_fix = tails.get("fixpipe", -1) if design.get("matmul_end_fences") else -1
        end_load = tails.get("mte1", -1) if design.get("matmul_end_fences") else -1
        for chunk in range(count):
            slices = job.get("k_slices") or (job.get('pipeline_k_slices') if design.get('pipeline_k_streaming') else [])
            if slices:
                buffers = 2 if job.get('k_slices') else job['pipeline_k_buffers']
                partials = []
                for part, fraction in enumerate(slices):
                    previous = launch
                    for pipe in pipes[:-1]:
                        dependencies = [previous, tails.get(pipe, -1)]
                        if pipe == "cube":
                            dependencies.append(end_fix)
                        if pipe == "mte2":
                            dependencies.append(end_load)
                        if pipe == "mte2" and part >= buffers:
                            dependencies.append(partials[part - buffers])
                        if pipe == "cube" and depth and chunk >= depth and part == 0:
                            dependencies.append(fixed[chunk - depth])
                        current = {name: job.get(name, 0) * fraction / count for name in PIPES[pipe][1]}
                        if pipe == "mte2":
                            current["loads"] = job["loads"] / count / len(slices)
                        previous = node(dependencies, template(pipe, current, phase=phase))
                        tails[pipe] = previous
                    partials.append(previous)
                previous = node([partials[-1], tails.get("fixpipe", -1)], encoded["fixpipe"])
                tails["fixpipe"] = previous
                fixed.append(previous)
                continue
            previous = launch
            for pipe in pipes:
                dependencies = [previous, tails.get(pipe, -1)]
                if pipe == "cube":
                    dependencies.append(end_fix)
                if pipe == "mte2":
                    dependencies.append(end_load)
                if pipe == "cube" and depth and chunk >= depth:
                    dependencies.append(fixed[chunk - depth])
                previous = node(dependencies, encoded[pipe])
                tails[pipe] = previous
            fixed.append(previous)
        return fixed[-1]

    finished, seen = [], set()
    init = {"workers": 0 if design.get('isolated_phase') else 1, "batch_entries": design["inputs"]["B"] if design["inputs"]["layout"] == "TND" else 0,
            "adapter_workers": int(design["schedule"]["drop_adapter"])}
    prefix = node()
    if design["adapter_jobs"]:
        adapter_finished = []
        for job in {cost.sha256(job): job for job in design["adapter_jobs"]}.values():
            adapter_finished.append(flow(job, ("vscalar", "vmte2", "vector", "vmte3"), max(1, int(job["chunks"])), prefix, {}, phase="drop_adapter"))
        prefix = node(adapter_finished)
    startup = node([node([prefix], template("cscalar", init)), node([prefix], template("vscalar", init))])
    unit_ids = {id(unit): index for index, unit in enumerate(design.get("units", []))}
    for tasks in design["workers"]:
        identity = tuple(unit_ids[id(job)] for job in tasks) if unit_ids else cost.sha256(tasks)
        if identity in seen:
            continue
        seen.add(identity)
        count = len(tasks)
        if not count:
            continue
        cube_launch = {(task, stage): node() for task in range(count) for stage in (0, 2)}
        cube_done = {(task, stage): node() for task in range(count) for stage in (0, 2)}
        vec_done = {(owner, task, stage): node() for owner in (0, 1) for task in range(count) for stage in (1, 3)}
        tails, previous = {}, startup
        for step in range(count + 1):
            for task, stage in ((step, 0), (step - 1, 2)):
                if not 0 <= task < count:
                    continue
                dependencies = [previous]
                if stage == 2:
                    dependencies.extend(vec_done[owner, task, 1] for owner in (0, 1))
                launch = cube_launch[task, stage]
                resolve(launch, dependencies)
                job = tasks[task]["cube"][stage // 2]
                depth = int(design["matmul"][stage // 2].get("db_l0c", 2)) if finite_buffers else 0
                end = flow(job, ("cscalar", "mte2", "mte1", "cube", "fixpipe"), max(1, int(job.get("stores", 0))), launch, tails, depth, "bmm1" if stage == 0 else "bmm2")
                resolve(cube_done[task, stage], [end])
                previous = launch
        for owner in (0, 1):
            previous, tails = startup, {}
            for step in range(count + 1):
                for task, stage in ((step, 1), (step - 1, 3)):
                    if not 0 <= task < count:
                        continue
                    launch = node([previous, cube_done[task, 0 if stage == 1 else 2]])
                    job = tasks[task]["vector"][owner][stage // 2]
                    end = flow(job, ("vscalar", "vmte2", "vector", "vmte3"), max(1, int(job.get("chunks", 0))), launch, tails, phase="vec1" if stage == 1 else "vec2")
                    resolve(vec_done[owner, task, stage], [end])
                    previous = vec_done[owner, task, stage]
            finished.append(previous)
    root = node(finished)
    a, b, which = np.asarray(left, dtype=np.int32), np.asarray(right, dtype=np.int32), np.asarray(labels, dtype=np.int32)
    order = np.empty(len(a), dtype=np.int32)
    if graph_engine().fa_latency_order(len(a), a.ctypes.data, b.ctypes.data, order.ctypes.data) != len(a):
        raise RuntimeError("source timing graph contains a cyclic dependency")
    inverse = np.empty(len(a), dtype=np.int32)
    inverse[order] = np.arange(len(a), dtype=np.int32)
    return {"a": np.where(a[order] >= 0, inverse[np.maximum(a[order], 0)], -1).astype(np.int32),
            "b": np.where(b[order] >= 0, inverse[np.maximum(b[order], 0)], -1).astype(np.int32),
            "which": which[order].copy(), "templates": np.asarray(templates, dtype=np.float64), "root": int(inverse[root]),
            "finite_l0c_buffers": finite_buffers, "phases": template_phases}


def graph_time(graph, rates, gradient=False):
    coefficients = np.ascontiguousarray(rates, dtype=np.float64)
    derivative = np.empty(len(coefficients), dtype=np.float64) if gradient else None
    value = graph_engine().fa_latency_eval(len(graph["a"]), len(graph["templates"]), len(coefficients), graph["root"],
                graph["a"].ctypes.data, graph["b"].ctypes.data, graph["which"].ctypes.data,
                graph["templates"].ctypes.data, coefficients.ctypes.data, derivative.ctypes.data if gradient else None)
    if not math.isfinite(value):
        raise RuntimeError("source timing graph produced a non-finite duration")
    return (value, derivative) if gradient else value


def critical_phases(graph, rates):
    durations = graph["templates"] @ np.asarray(rates)
    extra = np.zeros((len(durations), len(PHASES)))
    for index, phase in enumerate(graph["phases"]):
        extra[index, PHASES.index(phase)] = durations[index]
    encoded = {**graph, "templates": np.column_stack((graph["templates"], extra))}
    value, derivative = graph_time(encoded, np.append(rates, np.zeros(len(PHASES))), True)
    result = dict(zip(PHASES, map(float, derivative[-len(PHASES):])))
    if not math.isclose(sum(result.values()), value, rel_tol=1e-9, abs_tol=1e-8):
        raise RuntimeError("phase critical-path contributions do not sum to kernel latency")
    return result


def elapsed(design, model):
    same = design["schedule"]["same_ab"]
    paired_l1 = design["inputs"]["layout"] == "TND" and design["schedule"]["l1"] and not same
    controlled_cube = same or paired_l1
    width = 1 if same else 2
    pair_results, seen = [], set()
    for first in range(0, len(design["workers"]), width):
        workers = design["workers"][first:first + width]
        if same:
            workers = [workers[0], workers[0]]
        identity = cost.sha256(workers)
        if identity in seen:
            continue
        seen.add(identity)
        programs = []
        for worker, tasks in enumerate(workers):
            program = []
            for step in range(len(tasks) + 2):
                if 0 <= step - 1 < len(tasks):
                    program.append(("wait", step - 1, 0))
                if step < len(tasks) and not controlled_cube:
                    program.append(("launch", step, 0))
                if 0 <= step - 1 < len(tasks):
                    program.append(("vector", step - 1, 1))
                    if paired_l1 and len(workers) == 2 and step - 1 < len(workers[1 - worker]):
                        program.append(("peer_wait", step - 1, 0))
                if 0 <= step - 2 < len(tasks):
                    program.append(("wait", step - 2, 2))
                if 0 <= step - 1 < len(tasks) and not controlled_cube:
                    program.append(("launch", step - 1, 2))
                if 0 <= step - 2 < len(tasks):
                    program.append(("vector", step - 2, 3))
            programs.append(program)
        ready, positions, blocked = {}, [0] * len(workers), {}
        pending, events, finished, vector_done = deque(), [], {}, set()
        resources, busy, flows, groups = {}, set(), {}, {}
        serial, next_flow = 0, 0
        totals = Counter()
        cube_program, cube_position = [], 0
        if controlled_cube:
            cube_workers = workers[:1] if same else workers
            for step in range(max(map(len, cube_workers)) + 1):
                for client, tasks in enumerate(cube_workers):
                    if step < len(tasks):
                        cube_program.append((client, step, 0))
                for client, tasks in enumerate(cube_workers):
                    if 0 <= step - 1 < len(tasks):
                        cube_program.append((client, step - 1, 2))

        def event(time, flow):
            nonlocal serial
            serial += 1
            heapq.heappush(events, (time, serial, flow))

        def completion_key(worker, task, stage):
            return (0 if same else worker), task, stage

        def start_resource(resource, time):
            if resource in busy or not resources.get(resource):
                return
            flow = resources[resource].popleft()
            item = flows[flow]
            busy.add(resource)
            event(time + item["steps"][item["position"]][1], flow)

        def enqueue(flow, time):
            item = flows[flow]
            resource = item["steps"][item["position"]][0]
            resources.setdefault(resource, deque()).append(flow)
            start_resource(resource, time)

        def launch(worker, task, stage, time):
            nonlocal next_flow
            cube = stage in (0, 2)
            job = workers[worker][task]["cube"][stage // 2] if cube else workers[worker][task]["vector"][worker if same else 0][stage // 2]
            count = max(1, int(job.get("stores" if cube else "chunks", 0)))
            pipes = ("cscalar", "mte2", "mte1", "cube", "fixpipe") if cube else ("vscalar", "vmte2", "vector", "vmte3")
            group = worker, task, stage
            groups[group] = count
            durations = [service(job, pipe, model) / count for pipe in pipes]
            totals[("bmm1_us", "vec1_us", "bmm2_us", "vec2_us")[stage]] += max(durations) * count
            steps = [((pipe, 0 if cube else worker), duration) for pipe, duration in zip(pipes, durations)]
            for _ in range(count):
                next_flow += 1
                flows[next_flow] = {"steps": steps, "position": 0, "group": group}
                enqueue(next_flow, time)

        def start_cube(time):
            nonlocal cube_position
            if controlled_cube:
                while cube_position < len(cube_program):
                    client, task, stage = cube_program[cube_position]
                    required = range(2) if same else (client,)
                    if stage == 2 and any((v, task) not in vector_done for v in required):
                        return
                    launch(client, task, stage, time)
                    cube_position += 1
                return
            while pending:
                worker, task, stage = pending[0]
                if same and stage == 2 and any((v, task) not in vector_done for v in range(2)):
                    return
                pending.popleft()
                launch(worker, task, stage, time)

        def advance(worker, time):
            while positions[worker] < len(programs[worker]):
                kind, task, stage = programs[worker][positions[worker]]
                key = completion_key(1 - worker if kind == "peer_wait" else worker, task, stage)
                if kind in ("wait", "peer_wait") and key not in ready:
                    blocked[worker] = key
                    return
                positions[worker] += 1
                if kind == "launch":
                    pending.append((worker, task, stage))
                    start_cube(time)
                elif kind == "vector":
                    launch(worker, task, stage, time)
                    return
            finished[worker] = time

        init = {"workers": 1, "batch_entries": design["inputs"]["B"] if design["inputs"]["layout"] == "TND" else 0}
        startup = max(service(init, "cscalar", model), service(init, "vscalar", model))
        if controlled_cube:
            start_cube(startup)
        for worker in range(len(workers)):
            advance(worker, startup)
        while events:
            time, _, flow = heapq.heappop(events)
            item = flows[flow]
            resource = item["steps"][item["position"]][0]
            busy.remove(resource)
            start_resource(resource, time)
            item["position"] += 1
            if item["position"] < len(item["steps"]):
                enqueue(flow, time)
                continue
            group = item["group"]
            flows.pop(flow)
            groups[group] -= 1
            if groups[group]:
                continue
            groups.pop(group)
            worker, task, stage = group
            if stage in (0, 2):
                key = completion_key(worker, task, stage)
                ready[key] = time
                for client in list(blocked):
                    if blocked[client] == key:
                        blocked.pop(client)
                        advance(client, time)
            else:
                if stage == 1:
                    vector_done.add((worker, task))
                    start_cube(time)
                advance(worker, time)
        if len(finished) != len(workers) or pending or blocked or flows or busy or groups or controlled_cube and cube_position != len(cube_program):
            raise RuntimeError("source pipeline has unresolved Cube/Vector dependencies")
        pair_results.append({"pipeline_us": max(finished.values()), "critical_cube": first // width,
                             "stage_service_us": dict(totals), "startup_us": startup})
    return max(pair_results, key=lambda item: item["pipeline_us"])


def adapter(design, model):
    p, s, h = design["inputs"], design["schedule"], model["hardware"]
    if not s["drop_adapter"]:
        return {"adapter_us": 0.0, "adapter_passes": 0}
    duration, passes = 0.0, 0
    for job in design["adapter_jobs"]:
        pipes = [service(job, pipe, model) for pipe in ("vector", "vmte2", "vmte3")]
        current = max(max(pipes), service(job, "vscalar", model)) + (sum(pipes) - max(pipes)) / job["chunks"]
        duration, passes = max(duration, current), max(passes, job["chunks"])
    return {"adapter_us": duration, "adapter_passes": passes}


def predict(inputs, model):
    if model["version"] != VERSION or model["source_sha256"] != source_hash():
        raise RuntimeError("source model and frozen predictor differ; native caches are unaffected")
    design = workload(inputs, model["hardware"])
    return predict_design(design, model)


def predict_design(design, model, geometry_origin="cpu_matmul_tiler"):
    if not modeled_path(design):
        raise ValueError("outside the modeled dense/TND SameAB implementations")
    group = cost_group(design)
    entry = model["classes"].get(group)
    if entry is None:
        raise ValueError(f"unmeasured execution class: {group}")
    if design["unpaired_reuse_segments"]:
        raise ValueError("unmodeled paired-Matmul empty-message control cost")
    if entry["parameter_names"] != [list(pair) for pair in parameter_names()]:
        raise RuntimeError("source primitive dimensions differ from the calibrated model")
    rates = {"hardware": model["hardware"], "pipes": entry["pipes"], "strict_classes": True, "class_id": group}
    active = {phase: {pipe: service(job, pipe, rates) for pipe, job in pipes.items()} for phase, pipes in phase_features(design).items()}
    graph = timing_graph({**design, 'pipeline_k_streaming': bool(entry.get('pipeline_k_streaming')),
                          'matmul_end_fences': bool(entry.get('matmul_end_fences'))})
    value = graph_time(graph, entry["rates"])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("invalid source-model latency")
    return {"pipeline_us": value, "adapter_us": 0.0, "predicted_us": value, "input_schedule": design["schedule"],
            "matmul": design["matmul"], "work": design["totals"], "cost_policy": group,
            "identification_range_us": [None, None], "identification_width_percent": None,
            "phase_pipe_active_sum_us": active, "phase_pipe_active_sum_range_us": {},
            "critical_path_phase_us": critical_phases(graph, entry["rates"]),
            "range_scope": "Stage costs are jointly calibrated, not independently measured; no identified interval or accuracy guarantee is claimed",
            "point_rule": "Critical-path evaluation of source operations, dependencies, engine queues and finite L0C buffers",
            "prediction_quality": "source_operation_pipeline_model", "model_version": VERSION,
            "host_api_sha256": tiler()[1] if geometry_origin == "cpu_matmul_tiler" else geometry_origin,
            "qualification": "experimental_accuracy_reported_by_execution_class",
            "unidentified_costs": list(PIPES),
            "transferred_pipe_rates": [], "execution_class": execution_class(design),
            "out_of_calibration_range": [name for name, bounds in entry["input_ranges"].items() if not bounds[0] <= design["inputs"][name] <= bounds[1]]}


@lru_cache(maxsize=1)
def key_schema():
    path = Path(os.environ.get("OPS_TRANSFORMER_ROOT", str(ROOT.parent / "ops-transformer-official-8.5.0")))
    path /= SOURCE + "op_kernel/arch32/flash_attention_score_template_tiling_key.h"
    text = path.read_text().split("ASCENDC_TPL_SEL(", 1)[0]
    return [(name, int(width), tuple(map(int, re.findall(r"\d+", values)))) for name, width, values in
            re.findall(r"ASCENDC_TPL_UINT_DECL\(\s*(\w+)\s*,\s*ASCENDC_TPL_(\d+)_BW\s*,\s*ASCENDC_TPL_UI_LIST\s*,([^)]*)\)", text)]


def verify_class_trace(raw, design):
    value, shift, decoded = int(raw["tiling_key"]), 0, {}
    schema = key_schema()
    if not schema:
        raise ValueError("official template key schema is missing")
    for name, width, values in schema:
        index = value >> shift & ((1 << width) - 1)
        if index >= len(values):
            raise ValueError("Host key contains an invalid template index")
        decoded[name] = values[index]
        shift += width
    s, p = design["schedule"], design["inputs"]
    ub0, ub1, block = (9, 9, 0) if s["family"] in ("b", "b_drop") else (3, 9, 9) if s["same_ab"] else (3, 5, 9) if s["family"] == "s1" else (3, 4, 9)
    expected = {"UB0": ub0, "UB1": ub1, "Block": block, "DataType": 2 if p["dtype"] == "BF16" else 3,
                "KernelTypeKey": 0, "Sparse": 1 if s["family"] in ("b", "b_drop") else 2, "BigDoubleBuffer": 2,
                "Layout": {"BSND": 1, "SBH": 2, "BNSD": 3, "TND": 4}[p["layout"]],
                "Bmm1Format": 0 if p["layout"] == "TND" else int(s["nz"]),
                "Bmm2Source": int(s["family"] == "s1" and s["l1"]), "EnableL1Reuse": int(s["reuse"]),
                "MatmulPolicyType": int(s["policy"] == "UNSPLITK"), "HasDropOut": p["has_drop_mask"],
                "HasAttenMask": 0, "HasPse": 0, "HasRope": 0, "ImplMode": 0}
    if not s["static_matmul"]:
        expected.update(S1TemplateType=0, S2TemplateType=0, DTemplateType=0)
    return {name: [expected[name], decoded.get(name)] for name in expected if expected[name] != decoded.get(name)}


def measurement_corpus(paths, hardware, header_schemas=None):
    measurements, inventory, content_seen = {}, [], {}
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_limit, 32 * 1024 * 1024))
    for path in map(Path, paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        audit = {"name": path.name, "sha256": digest, "rows": 0, "accepted": 0, "rejected": Counter(), "statuses": Counter(), "non_successful_shapes": set()}
        if digest in content_seen:
            audit["duplicate_file_of"] = content_seen[digest]
            audit["non_successful_shapes"] = []
            inventory.append(audit)
            continue
        content_seen[digest] = path.name
        schema = (header_schemas or {}).get(path.name)
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as source:
            lines = source.readlines()
            reader = csv.reader(lines)
            while True:
                start_line = reader.line_num
                try:
                    cells = next(reader)
                except StopIteration:
                    break
                except csv.Error as error:
                    audit["rejected"]["csv_error: " + str(error)] += 1
                    continue
                if "duration_us" in cells and "status" in cells and ("inputs" in cells or all(name in cells for name in cost.INPUT_FIELDS[:7])):
                    schema = cells
                    continue
                if schema is None or len(cells) != len(schema):
                    try:
                        raw = json.loads("".join(lines[start_line:reader.line_num]))
                        if not isinstance(raw, dict) or "status" not in raw or "inputs" not in raw:
                            raise ValueError("not a measurement object")
                    except (ValueError, json.JSONDecodeError):
                        audit["rejected"]["not_a_measurement_row"] += 1
                        continue
                else:
                    raw = dict(zip(schema, cells))
                audit["rows"] += 1
                audit["statuses"][raw.get("status", "missing")] += 1
                if raw.get("status") != "ok":
                    audit["rejected"]["non_successful_status"] += 1
                    try:
                        rejected_inputs = (dict(raw["inputs"]) if isinstance(raw.get("inputs"), dict) else json.loads(raw["inputs"])) if raw.get("inputs") else {name: raw[name] for name in cost.INPUT_FIELDS}
                        rejected_inputs["has_drop_mask"] = {"false": 0, "true": 1}.get(str(rejected_inputs["has_drop_mask"]).lower(), rejected_inputs["has_drop_mask"])
                        audit["non_successful_shapes"].add(cost.sha256(cost.normalize(rejected_inputs)))
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                        pass
                    continue
                try:
                    inputs = (dict(raw["inputs"]) if isinstance(raw.get("inputs"), dict) else json.loads(raw["inputs"])) if raw.get("inputs") else {name: raw[name] for name in cost.INPUT_FIELDS}
                    flag = str(inputs["has_drop_mask"]).lower()
                    inputs["has_drop_mask"] = {"false": 0, "true": 1}.get(flag, inputs["has_drop_mask"])
                    inputs = cost.normalize(inputs)
                    duration = float(raw["duration_us"])
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError("invalid measured duration")
                    if any(str(raw[name]) != str(hardware[name]) for name in hardware if raw.get(name)):
                        raise ValueError("different hardware profile")
                    effective = (raw["effective_tiling"] if isinstance(raw.get("effective_tiling"), dict) else json.loads(raw["effective_tiling"])) if raw.get("effective_tiling") else {}
                    normalized = {**raw, **inputs, "duration_us": duration}
                    for name in ("s1_base", "s2_base", "n_ratio", "core_num", "total_tasks", "split_factor"):
                        value = raw.get(name) or raw.get("effective_" + name) or effective.get("FA_" + name.upper())
                        if value not in (None, ""):
                            normalized[name] = int(value)
                    for stage in (1, 2):
                        for name in ("base_m", "base_n", "base_k", "depth_a1", "depth_b1", "db_l0c", "step_m", "step_n"):
                            field = f"bmm{stage}_{name}"
                            value = raw.get(field) or effective.get("FA_" + field.upper())
                            if value not in (None, ""):
                                normalized[field] = int(value)
                    preset = raw.get("effective_bmm_preset") or effective.get("FA_BMM_PRESET", 0)
                    normalized["bmm_preset"] = int(preset)
                    attrs = {name: str(raw.get(name, "0")).lower() for name in ("has_pse", "has_attention_mask")}
                    if any(value not in ("", "0", "false") for value in attrs.values()) or str(raw.get("sparse_mode", "0")) not in ("", "0"):
                        raise ValueError("unsupported operator attributes")
                    route = schedule(inputs, hardware)
                    normalized["family"] = raw.get("family") or route["family"]
                    normalized["official_tiling"] = int(preset) == 0 and all(normalized.get(name) == route[name] for name in ("s1_base", "s2_base", "n_ratio", "core_num", "total_tasks", "split_factor"))
                    correctness = (raw["correctness"] if isinstance(raw.get("correctness"), dict) else json.loads(raw["correctness"])) if raw.get("correctness") else {}
                    correctness = correctness if isinstance(correctness, dict) else {}
                    checked = int(raw.get("checked_elements") or correctness.get("checked_elements") or 0)
                    normalized["checked_elements"] = checked
                    normalized["correctness_basis"] = "independent_reference" if checked > 0 else "baseline_comparison" if raw.get("correct") == "True" else "runner_success_only"
                    normalized["timing_protocol"] = raw.get("timing_source") or "msprof_task_duration"
                    normalized["source_files"] = [path.name]
                    normalized["shape_identity"] = cost.sha256(inputs)
                    normalized["same_ab"] = bool(route["same_ab"])
                    normalized["stable"] = str(raw.get("stable", "True")).lower() == "true"
                    rounds = (raw["round_medians_us"] if isinstance(raw.get("round_medians_us"), list) else json.loads(raw["round_medians_us"])) if raw.get("round_medians_us") else [float(raw[name]) for name in ("round1_us", "round2_us", "round3_us") if raw.get(name)]
                    normalized["round_values_us"] = rounds
                    normalized["round_spread_ratio"] = max(rounds) / min(rounds) if rounds and min(rounds) > 0 else None
                    if raw.get("correct") == "False":
                        raise ValueError("numerical comparison failed")
                    configuration = {name: normalized.get(name) for name in ("tiling_key", "s1_base", "s2_base", "n_ratio", "core_num", "total_tasks", "split_factor", "bmm_preset")}
                    configuration.update({f"bmm{stage}_{name}": normalized.get(f"bmm{stage}_{name}") for stage in (1, 2) for name in ("base_m", "base_n", "base_k", "depth_a1", "depth_b1")})
                    normalized["configuration_identity"] = cost.sha256({"inputs": inputs, "tiling": configuration, "protocol": normalized["timing_protocol"]})
                    identity = cost.sha256({"configuration": normalized["configuration_identity"], "duration_us": round(duration, 3),
                                            "rounds": [round(value, 3) for value in rounds], "trace": raw.get("latency_trace_sha256") or ""})
                    normalized["measurement_identity"] = identity
                    if identity in measurements:
                        previous = measurements[identity]
                        sources = sorted(set(previous["source_files"] + [path.name]))
                        merged = dict(previous)
                        for name, value in normalized.items():
                            if value not in (None, "", []) or name not in merged:
                                merged[name] = value
                        merged["source_files"] = sources
                        measurements[identity] = merged
                    else:
                        measurements[identity] = normalized
                    audit["accepted"] += 1
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    audit["rejected"][str(error)[:120]] += 1
        audit["rejected"], audit["statuses"] = dict(audit["rejected"]), dict(audit["statuses"])
        audit["non_successful_shapes"] = sorted(audit["non_successful_shapes"])
        inventory.append(audit)
    csv.field_size_limit(previous_limit)
    return {"measurements": list(measurements.values()), "inventory": inventory}


def pipeline_samples(corpus, hardware, graph_store=None):
    grouped, excluded, designs = {}, [], {}
    graph_signature = cost.sha256({"source": source_hash(), "host_api": tiler()[1], "hardware": hardware}) if graph_store is not None else None
    for row in corpus["measurements"]:
        if not row["same_ab"]:
            continue
        grouped.setdefault(row["configuration_identity"], []).append(row)
    samples = []
    for index, (identity, observations) in enumerate(grouped.items(), 1):
        raw = max(observations, key=lambda row: sum(bool(row.get(field)) for field, _ in PIPES.values()))
        inputs = cost.normalize({name: raw[name] for name in cost.INPUT_FIELDS})
        try:
            required = ("s1_base", "s2_base", "n_ratio", "core_num", "total_tasks", "split_factor")
            if any(name not in raw for name in required):
                raise ValueError("actual tiling schedule is missing")
            actual = {name: int(raw[name]) for name in required}
            actual["priority"] = int(raw["priority"])
            geometry = [{name: int(raw[f"bmm{stage}_{name}"]) for name in
                         ("base_m", "base_n", "base_k", "depth_a1", "depth_b1", "db_l0c", "step_m", "step_n")
                         if raw.get(f"bmm{stage}_{name}") not in (None, "")} for stage in (1, 2)]
            if any(not all(name in mm for name in ("base_m", "base_n", "base_k", "depth_a1", "depth_b1")) for mm in geometry):
                raise ValueError("actual Matmul geometry is missing")
            design_id = cost.sha256({"inputs": inputs, "schedule": actual, "geometry": geometry})
            if design_id not in designs:
                design = workload(inputs, hardware, calibration_geometry=geometry, calibration_schedule=actual)
                mismatch = verify_class_trace(raw, design)
                if mismatch:
                    raise ValueError("operator attributes or template key differ from the modeled source: " + json.dumps(mismatch))
                if graph_store is not None:
                    folder = Path(graph_store) / design_id
                    folder.mkdir(parents=True, exist_ok=True)
                    marker = folder / "graph.json"
                    saved = json.loads(marker.read_text()) if marker.is_file() else {}
                    if saved.get("signature") == graph_signature:
                        graph = {**saved, **{field: np.load(folder / (field + ".npy"), mmap_mode="r") for field in ("a", "b", "which", "templates")}}
                    else:
                        graph = timing_graph(design)
                        for field in ("a", "b", "which", "templates"):
                            target = folder / (field + ".npy")
                            np.save(target, graph[field])
                            graph[field] = np.load(target, mmap_mode="r")
                        marker.write_text(json.dumps({"signature": graph_signature, "root": graph["root"], "finite_l0c_buffers": True, "phases": graph["phases"]}))
                else:
                    graph = timing_graph(design)
                designs[design_id] = {"graph": graph, "features": {pipe: dict(features(design, pipe)) for pipe in PIPES},
                                      "phase_features": {phase: {pipe: dict(value) for pipe, value in pipes.items()} for phase, pipes in phase_features(design).items()},
                                      "class_id": cost_group(design), "path": execution_class(design), "geometry": design["matmul"], "schedule": design["schedule"]}
                del design
                gc.collect()
            structure = designs[design_id]
            duration = statistics.median(float(row["duration_us"]) for row in observations)
            targets = {pipe: statistics.median(float(row[field]) for row in observations if row.get(field) and float(row[field]) > 0)
                       for pipe, (field, _) in PIPES.items() if any(row.get(field) and float(row[field]) > 0 for row in observations)}
            core_times = [float(row[field]) for row in observations for field in ("aic_time_max_us", "aiv_time_max_us") if row.get(field) and float(row[field]) > 0]
            profile_ratio = max(core_times, default=duration) / duration
            spreads = [math.log(float(row["round_spread_ratio"])) for row in observations if row.get("round_spread_ratio")]
            noise = max(spreads, default=0.0)
            samples.append({**structure, "identity": identity, "shape_identity": raw["shape_identity"], "inputs": inputs,
                            "duration_us": duration, "targets": targets, "timing_protocol": raw["timing_protocol"],
                            "official_tiling": all(row["official_tiling"] for row in observations),
                            "sources": sorted({name for row in observations for name in row["source_files"]}),
                            "measurement_count": len(observations), "stable": all(row["stable"] for row in observations),
                            "case": raw.get("case") or raw["shape_identity"], "checked_elements": max(int(row["checked_elements"]) for row in observations),
                            "pmu_timing_ratio": profile_ratio, "pmu_weight": 1 / math.sqrt(1 + (math.log(profile_ratio) / 0.03) ** 2),
                            "timing_weight": 1 / math.sqrt(1 + (noise / 0.03) ** 2), "round_spread_ratio": math.exp(noise),
                            "configuration": actual})
        except (ValueError, KeyError, RuntimeError) as error:
            excluded.append({"case": raw.get("case"), "shape_identity": raw["shape_identity"], "sources": raw["source_files"], "reason": str(error)})
        if index % 32 == 0 or index == len(grouped):
            print(f"SOURCE GRAPHS: {index}/{len(grouped)} configurations; all supplied shapes retained in the audit", flush=True)
    return samples, excluded


def fit_pipeline_group(samples, hardware, phase_targets=(), phase_graph_targets=(), transport_counters_are_upper_bounds=False):
    from scipy.optimize import least_squares, nnls
    names = parameter_names()
    position = {pair: index for index, pair in enumerate(names)}
    dimensions = len(names)
    initial, scales = np.zeros(dimensions), np.ones(dimensions)
    prior_rows, prior_values, prior_weights, prior_scalar = [], [], [], []
    for pipe, (_, fields) in PIPES.items():
        observations = [row for row in samples if pipe in row["targets"]]
        if not observations:
            continue
        matrix = np.asarray([[row["features"][pipe].get(name, 0) for name in fields] for row in observations], dtype=float)
        values = np.asarray([row["targets"][pipe] for row in observations])
        confidence = np.sqrt(np.asarray([row["pmu_weight"] for row in observations]))
        column_scale = np.maximum(np.max(abs(matrix), axis=0), 1)
        relative = matrix / values[:, None] / column_scale
        solution, _ = nnls(relative * confidence[:, None], confidence, maxiter=10000)
        fitted = solution / column_scale
        budget = statistics.median(values) / max(1, np.count_nonzero(np.max(matrix, axis=0)))
        for j, name in enumerate(fields):
            at = position[pipe, name]
            scales[at] = max(float(fitted[j]), budget / max(1, float(np.max(matrix[:, j]))), 1e-12)
            initial[at] = fitted[j]
        for row, value in zip(observations, values):
            vector = np.zeros(dimensions)
            for name in fields:
                vector[position[pipe, name]] = row["features"][pipe].get(name, 0)
            prior_rows.append(vector / value)
            prior_values.append(value)
            prior_weights.append(math.sqrt(0.25 * row["pmu_weight"]))
            prior_scalar.append(pipe not in ("cube", "vector") if transport_counters_are_upper_bounds else pipe in ("cscalar", "vscalar"))
    phase_replicas = Counter(row.get('case', '') for row in phase_targets)
    for row in phase_targets:
        vector = np.zeros(dimensions)
        for name in PIPES[row['pipe']][1]:
            vector[position[row['pipe'], name]] = row['features'].get(name, 0)
        prior_rows.append(vector / row['target_us'])
        prior_weights.append(1 / math.sqrt(phase_replicas[row.get('case', '')]))
        prior_scalar.append(row['pipe'] in ('cscalar', 'vscalar'))
    prior = np.asarray(prior_rows).reshape((-1, dimensions))
    confidence = np.asarray(prior_weights)
    upper = np.asarray(prior_scalar, dtype=bool)
    exposed = np.asarray([any(np.any(row["graph"]["templates"][:, j]) for row in samples) for j in range(dimensions)])
    active = np.flatnonzero(exposed)
    event = np.asarray([row["timing_protocol"] == "aclrtEventElapsedTime" for row in samples], dtype=float)
    event_parameter = bool(np.any(event))
    start = initial[active] / scales[active]
    limits = np.full(len(active), np.inf)
    for row in phase_targets:
        if row.get('phase') != 'startup':
            continue
        for at, index in enumerate(active):
            pipe, field = names[index]
            count = row['features'].get(field, 0) if pipe == row['pipe'] else 0
            if count > 0:
                limits[at] = min(limits[at], row.get('target_high_us', row['target_us']) / count / scales[index])
    start = np.minimum(start, limits * 0.95)
    if event_parameter:
        start = np.append(start, 0.0)
        limits = np.append(limits, np.inf)
    target = np.asarray([row["duration_us"] for row in samples])
    weights = np.sqrt(np.asarray([row["timing_weight"] for row in samples]))
    replicas = Counter(row["shape_identity"] for row in samples)
    weights /= np.sqrt(np.asarray([replicas[row["shape_identity"]] for row in samples]))
    last = {}
    phase_graph_replicas = Counter(row['case'] for row in phase_graph_targets)

    def residual(parameters):
        if "parameters" in last and np.array_equal(parameters, last["parameters"]):
            return last["residual"]
        rates = np.zeros(dimensions)
        rates[active] = parameters[:len(active)] * scales[active]
        predictions, derivatives = [], []
        for row in samples:
            value, derivative = graph_time(row["graph"], rates, True)
            predictions.append(value)
            derivatives.append(derivative[active] * scales[active])
        prediction = np.asarray(predictions)
        jacobian = np.asarray(derivatives)
        if event_parameter:
            prediction += event * parameters[-1]
            jacobian = np.column_stack((jacobian, event))
        times = (prediction / target - 1) * weights
        time_jacobian = jacobian / target[:, None] * weights[:, None]
        for row in phase_graph_targets:
            value, derivative = graph_time(row['graph'], rates, True)
            weight = 1 / math.sqrt(phase_graph_replicas[row['case']])
            times = np.append(times, (value / row['target_us'] - 1) * weight)
            current = derivative[active] * scales[active] / row['target_us'] * weight
            if event_parameter:
                current = np.append(current, 0.0)
            time_jacobian = np.vstack((time_jacobian, current))
        counter = prior @ rates - 1
        counter_jacobian = prior[:, active] * scales[active]
        disabled = upper & (counter <= 0)
        counter[disabled] = 0
        counter_jacobian[disabled] = 0
        if event_parameter:
            counter_jacobian = np.column_stack((counter_jacobian, np.zeros(len(counter))))
        values = np.concatenate((times, counter * confidence, 0.001 * (parameters - start)))
        matrix = np.vstack((time_jacobian, counter_jacobian * confidence[:, None], 0.001 * np.eye(len(parameters))))
        last.update(parameters=parameters.copy(), residual=values, jacobian=matrix)
        return values

    optimum = least_squares(residual, start, jac=lambda x: (residual(x), last["jacobian"])[1], bounds=(0, limits),
                            loss="soft_l1", f_scale=0.03, max_nfev=160, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    rates = np.zeros(dimensions)
    rates[active] = optimum.x[:len(active)] * scales[active]
    pipes = {pipe: {"coefficients": {name: float(rates[position[pipe, name]]) for name in fields},
                    "status": "joint_pipeline_calibration", "samples": sum(pipe in row["targets"] for row in samples),
                    "inactive_feature_rates": [name for name in fields if not exposed[position[pipe, name]]]}
             for pipe, (_, fields) in PIPES.items()}
    errors = []
    for i, row in enumerate(samples):
        value = graph_time(row["graph"], rates) + (float(optimum.x[-1]) if event_parameter and event[i] else 0)
        errors.append(abs(value / row["duration_us"] - 1) * 100)
    return {"path": samples[0]["path"], "pipes": pipes, "rates": rates.tolist(), "parameter_names": [list(pair) for pair in names],
            "event_bracket_overhead_us": float(optimum.x[-1]) if event_parameter else 0,
            "measured_shapes": len({row["shape_identity"] for row in samples}), "configurations": len(samples),
            "input_ranges": {name: [min(row["inputs"][name] for row in samples), max(row["inputs"][name] for row in samples)] for name in cost.INPUT_FIELDS[:7]},
            "calibration_shapes": sorted({row["shape_identity"] for row in samples}), "optimization_success": bool(optimum.success),
            "optimization_evaluations": int(optimum.nfev), "calibration_mape_percent": statistics.mean(errors),
            "calibration_max_ape_percent": max(errors), "stage_costs_independently_measured": False,
            "calibration": "Shared nonnegative source-operation rates within this implementation, fitted jointly to kernel times and PMU constraints. Scalar active time is an upper bound, since wait/control is already represented in the graph. Entire shapes and all their tilings must stay in the same validation fold."}


def pipeline_model(samples, hardware, inventory=()):
    classes = {}
    for group in sorted({row['class_id'] for row in samples}):
        selected = [row for row in samples if row['class_id'] == group]
        print(f"SOURCE COST FIT: {group}; shapes={len({row['shape_identity'] for row in selected})}; configurations={len(selected)}", flush=True)
        classes[group] = fit_pipeline_group(selected, hardware)
    return {'version': VERSION, 'source_sha256': source_hash(), 'hardware': hardware, 'classes': classes,
            'training_shapes': sorted({row['shape_identity'] for row in samples}),
            'exposed_shapes': sorted({row['shape_identity'] for row in samples}),
            'source_evidence': source_evidence(), 'development_data': list(inventory),
            'scope': 'Existing dense and uniform-TND SameAB implementations; FP16/BF16, observed layouts, NORMAL/UNSPLITK, with and without dropout. No fallback to another kernel implementation.',
            'calibration': 'Nonnegative primitive costs fitted jointly to measured kernel latency and PMU constraints; costs are shared within actual source implementations. Actual latency is a training target, never a prediction input.',
            'target': 'FlashAttentionScore kernel Task Duration(us)', 'global_fallback': False,
            'phase_costs': 'Stage resource demands are reconstructed, not independently timed. PMU Scalar busy time includes waits and is only an upper bound. Stage resource sums are not stage wall times.',
            'qualification': 'experimental_accuracy_reported_by_execution_class',
            'prospective_pass': False}


def pipeline_case(row, model, input_only=False, graph_cache=None):
    result = {'case': row['case'], 'identity': row['identity'], 'shape_identity': row['shape_identity'],
              'class_id': row['class_id'], 'inputs': row['inputs'], 'sources': row['sources'],
              'actual_us': row['duration_us'], 'stable': row['stable'], 'timing_protocol': row['timing_protocol'],
              'official_tiling': row['official_tiling'], 'measurement_count': row['measurement_count'],
              'checked_elements': row['checked_elements'], 'round_spread_ratio': row['round_spread_ratio'],
              'pmu_timing_ratio': row['pmu_timing_ratio'], 'prediction_scope': 'declared_tiling_replay',
              'predicted_us': None, 'ape_percent': None, 'pass_10_percent': False}
    try:
        entry = model['classes'].get(row['class_id'])
        if entry is None:
            raise ValueError('unmeasured execution class')
        graph = row['graph']
        if input_only:
            if not row['official_tiling']:
                raise ValueError('candidate tiling is not the official input-derived schedule')
            cache = graph_cache if graph_cache is not None else {}
            identity = row['shape_identity']
            if identity not in cache:
                inferred_schedule = schedule(row['inputs'], model['hardware'])
                design = {'inputs': row['inputs'], 'schedule': inferred_schedule, 'unpaired_reuse_segments': 0}
                cache[identity] = {'geometry': matmul_geometry(tuple(mm_requests(row['inputs'], inferred_schedule, model['hardware']))),
                                   'schedule': inferred_schedule, 'class_id': cost_group(design)}
            expected = cache[identity]
            bad = [f'bmm{stage}_{name}' for stage, (actual, inferred) in enumerate(zip(row['geometry'], expected['geometry']), 1)
                   for name in actual if actual[name] != inferred.get(name)]
            bad.extend(name for name in row['configuration'] if row['configuration'][name] != expected['schedule'].get(name))
            result['geometry_schedule_mismatch'] = bad
            result['prediction_scope'] = 'input_only_with_separate_event_protocol_adjustment' if row['timing_protocol'] == 'aclrtEventElapsedTime' else 'input_only_kernel_duration'
            if expected['class_id'] != row['class_id']:
                raise ValueError('input-derived source class differs from the recorded Host class')
            if bad:
                if 'graph' not in expected:
                    expected['graph'] = timing_graph(workload(row['inputs'], model['hardware']))
                graph = expected['graph']
        for pipe, fitted in entry['pipes'].items():
            if any(row['features'][pipe].get(name, 0) for name in fitted['inactive_feature_rates']):
                raise ValueError(f'unmeasured primitive operation: {pipe}')
        kernel = graph_time(graph, entry['rates'])
        critical = critical_phases(graph, entry['rates'])
        phase_demands = {phase: {pipe: sum(value.get(name, 0) * coefficient for name, coefficient in entry['pipes'][pipe]['coefficients'].items())
                                  for pipe, value in pipes.items()} for phase, pipes in row['phase_features'].items()}
        pipe_counters = {pipe: {'predicted_active_sum_us': sum(row['features'][pipe].get(name, 0) * coefficient for name, coefficient in entry['pipes'][pipe]['coefficients'].items()),
                                'measured_active_sum_us': row['targets'].get(pipe),
                                'scope': 'measured_scalar_includes_waits' if pipe in ('cscalar', 'vscalar') else 'whole_kernel_resource_counter'} for pipe in PIPES}
        overhead = entry['event_bracket_overhead_us'] if row['timing_protocol'] == 'aclrtEventElapsedTime' else 0.0
        prediction = kernel + overhead
        error = abs(prediction / row['duration_us'] - 1) * 100
        result.update(kernel_predicted_us=kernel, protocol_adjustment_us=overhead, predicted_us=prediction,
                      ape_percent=error, pass_10_percent=error <= 10 and not result.get('geometry_schedule_mismatch'),
                      signed_error_percent=100 * (prediction / row['duration_us'] - 1), absolute_error_us=abs(prediction - row['duration_us']),
                      critical_path_phase_us=critical, phase_resource_active_sum_us=phase_demands, resource_counters=pipe_counters,
                      active_cube_cores=row['schedule']['core_num'] // 2, available_cube_cores=int(model['hardware']['aic_cores']),
                      query_rows_per_task=min(row['schedule']['s1_base'], row['inputs']['S1']),
                      key_windows=cost.ceil_div(row['inputs']['S2'], row['schedule']['s2_base'] * row['schedule']['n_ratio']),
                      phase_attribution_scope='Predicted critical-path contributions, not separately measured phase wall times; whole-kernel PMU cannot uniquely assign an error to one stage')
    except (ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    return result


def error_metrics(rows):
    errors = [row['ape_percent'] for row in rows if row.get('ape_percent') is not None]
    passed = sum(bool(row.get('pass_10_percent', row['ape_percent'] <= 10)) for row in rows if row.get('ape_percent') is not None)
    def percentile(values, quantile):
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        left = int(position)
        return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)
    absolute = [abs(row['predicted_us'] - row['actual_us']) for row in rows if row.get('predicted_us') is not None]
    signed = [100 * (row['predicted_us'] / row['actual_us'] - 1) for row in rows if row.get('predicted_us') is not None]
    return {'count': len(rows), 'predicted': len(errors), 'unavailable': len(rows) - len(errors),
            'pass_10_percent': passed, 'fail_10_percent_including_unavailable': len(rows) - passed,
            'mape_percent': statistics.mean(errors) if errors else None, 'max_ape_percent': max(errors, default=None),
            'median_ape_percent': percentile(errors, 0.5), 'p90_ape_percent': percentile(errors, 0.9), 'p95_ape_percent': percentile(errors, 0.95),
            'mean_signed_error_percent': statistics.mean(signed) if signed else None,
            'median_absolute_error_us': percentile(absolute, 0.5), 'p90_absolute_error_us': percentile(absolute, 0.9),
            'error_bands': {'at_most_10': sum(value <= 10 for value in errors), '10_to_15': sum(10 < value <= 15 for value in errors),
                            '15_to_20': sum(15 < value <= 20 for value in errors), '20_to_30': sum(20 < value <= 30 for value in errors),
                            'above_30': sum(value > 30 for value in errors)},
            'all_pass': bool(rows) and passed == len(rows)}


def pipeline_validation(samples, hardware, folds=5):
    outcomes, cache = [], {}
    for fold in range(folds):
        held = [row for row in samples if int(cost.sha256({name: row['inputs'][name] for name in ('S1', 'S2')})[:8], 16) % folds == fold]
        train = [row for row in samples if int(cost.sha256({name: row['inputs'][name] for name in ('S1', 'S2')})[:8], 16) % folds != fold]
        if not held:
            continue
        assert not {row['shape_identity'] for row in held} & {row['shape_identity'] for row in train}
        needed = {row['class_id'] for row in held}
        model = pipeline_model([row for row in train if row['class_id'] in needed], hardware)
        for row in held:
            result = pipeline_case(row, model, input_only=row['official_tiling'], graph_cache=cache)
            result.update(fold=fold, prediction_origin='development_sequence_group_holdout', fold_model_sha256=cost.sha256(model))
            outcomes.append(result)
        print(f"SOURCE HOLDOUT: fold={fold + 1}/{folds}; held configurations={len(held)}; every shape's tilings and every S1/S2 pair stay together", flush=True)
    return outcomes


def pipeline_summary(outcomes):
    official = [row for row in outcomes if row['official_tiling']]
    return {'all_configurations': error_metrics(outcomes), 'official_input_only': error_metrics(official),
            'official_msprof_kernel_duration': error_metrics([row for row in official if row['timing_protocol'] == 'msprof_task_duration']),
            'official_event_bracket': error_metrics([row for row in official if row['timing_protocol'] == 'aclrtEventElapsedTime']),
            'stable_official': error_metrics([row for row in official if row['stable']]),
            'by_class': {group: {'all': error_metrics([row for row in outcomes if row['class_id'] == group]),
                                 'official': error_metrics([row for row in official if row['class_id'] == group])}
                         for group in sorted({row['class_id'] for row in outcomes})}}


def write_pipeline_csv(path, samples, outcomes, origin, append=False):
    indexed = {row['identity']: row for row in samples}
    rows = []
    for result in outcomes:
        sample = indexed[result['identity']]
        omitted = ('inputs', 'sources', 'critical_path_phase_us', 'phase_resource_active_sum_us', 'resource_counters', 'phase_attribution_scope')
        row = {**{name: value for name, value in result.items() if name not in omitted},
               **result['inputs'], 'sources': '|'.join(result['sources']), 'status': 'ok', 'evaluation': origin,
               **sample['configuration'],
               **{f'bmm{stage}_{name}': value for stage, geometry in enumerate(sample['geometry'], 1) for name, value in geometry.items()},
               **{f'critical_{phase}_us': result.get('critical_path_phase_us', {}).get(phase) for phase in PHASES},
               **{field: sample['targets'].get(pipe) for pipe, (field, _) in PIPES.items()},
               **{f'predicted_{pipe}_resource_sum_us': result.get('resource_counters', {}).get(pipe, {}).get('predicted_active_sum_us') for pipe in PIPES},
               **{f'{phase}_source_gm_to_l1_kb': sample['phase_features'][phase]['mte2'].get('read_kb', 0) for phase in ('bmm1', 'bmm2')},
               'independently_measured_phase_times': False}
        for name, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                row[name] = json.dumps(value, sort_keys=True, separators=(',', ':'))
        rows.append(row)
    path = Path(path)
    if append and path.is_file():
        with path.open(newline='') as source:
            rows = list(csv.DictReader(source)) + rows
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(dict.fromkeys(name for row in rows for name in row)), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def assess_costs(paths, previous_model, output, fieldnames=None, prepared=None):
    import tempfile
    hardware = previous_model['hardware']
    corpus = measurement_corpus(paths, hardware, header_schemas=fieldnames)
    with tempfile.TemporaryDirectory(prefix='fa_latency_graphs_') as temporary:
        samples, excluded = prepared if prepared is not None else pipeline_samples(corpus, hardware, temporary)
        if not samples:
            raise ValueError('no SameAB measurements have the required actual Host schedule and geometry')
        model = pipeline_model(samples, hardware, corpus['inventory'])
        outcomes = pipeline_validation(samples, hardware)
        model['assessment'] = pipeline_summary(outcomes)
        model['assessment'].update(measured_shapes=len({row['shape_identity'] for row in samples}),
                                   configurations=len(samples), source_classes=len(model['classes']),
                                   excluded_missing_or_inconsistent_metadata=excluded,
                                   prospective_pass=False, evaluation='development_sequence_grouped_5_fold_cross_validation')
        model['exposed_shapes'] = sorted({row['shape_identity'] for row in corpus['measurements']})
        model['qualification'] = 'experimental_accuracy_reported_by_execution_class'
        write_pipeline_csv(output, samples, outcomes, 'development_sequence_group_holdout')
        model['assessment']['csv_sha256'] = hashlib.sha256(Path(output).read_bytes()).hexdigest()
        return model


def evaluate(paths, model, input_only=True, fieldnames=None, output=None):
    import tempfile
    if model['version'] != VERSION or model['source_sha256'] != source_hash():
        raise RuntimeError('source model and frozen predictor differ')
    corpus = measurement_corpus(paths, model['hardware'], header_schemas=fieldnames)
    exposed = set(model['exposed_shapes'])
    with tempfile.TemporaryDirectory(prefix='fa_latency_holdout_') as temporary:
        samples, excluded = pipeline_samples(corpus, model['hardware'], temporary)
        cache, outcomes = {}, []
        for row in samples:
            result = pipeline_case(row, model, input_only=input_only and row['official_tiling'], graph_cache=cache)
            result['previously_exposed'] = row['shape_identity'] in exposed
            outcomes.append(result)
        repeated = [row for row in outcomes if row['previously_exposed']]
        unseen = [row for row in outcomes if not row['previously_exposed']]
        result = {'summary': {'all': pipeline_summary(outcomes), 'unseen_shapes': pipeline_summary(unseen), 'repeated_shapes': pipeline_summary(repeated)},
                  'shape_counts': {'all': len({row['shape_identity'] for row in outcomes}), 'unseen': len({row['shape_identity'] for row in unseen}),
                                   'repeated': len({row['shape_identity'] for row in repeated})},
                  'cases': outcomes, 'excluded_missing_or_inconsistent_metadata': excluded, 'inventory': corpus['inventory'],
                  'new_data_used_for_fit': False, 'prospective_pass': False, 'frozen_model_sha256': cost.sha256(model),
                  'evaluation': 'Frozen-model unseen-file validation; repetitions are reported separately. Measured before forecast, so this is not a prospective NPU experiment.'}
        if output is not None:
            write_pipeline_csv(output, samples, outcomes, 'frozen_file_holdout', append=True)
        return result


def freeze_predictions(payload, plan, records, snapshot, hardware):
    artifact = json.loads(MODEL.read_text())
    if artifact["source_sha256"] != source_hash():
        raise RuntimeError("frozen source model does not match the predictor")
    if any(str(artifact["hardware"][name]) != str(hardware[name]) for name in artifact["hardware"]):
        raise RuntimeError("source latency model belongs to another hardware profile")
    api = tiler()[1]
    if "source_model" in payload and (cost.sha256(payload["source_model"]) != cost.sha256(artifact) or payload.get("source_host_api_sha256", api) != api):
        previous = payload["source_model"]
        previous_sha = cost.sha256(previous)
        measured = {name: item for name, item in payload.get("source_predictions", {}).items()
                    if records.get(name, {}).get("model_sha256") == previous_sha}
        if measured:
            history_key = previous_sha + "/" + payload.get("source_host_api_sha256", "unknown")
            payload.setdefault("source_history", {})[history_key] = {"model": previous, "predictions": measured}
        payload["source_predictions"] = {}
    payload["source_model"] = artifact
    forecasts = payload.setdefault("source_predictions", {})
    payload["source_host_api_sha256"] = api
    identity = cost.sha256(artifact)
    for index, case in enumerate(plan, 1):
        name = case["name"]
        if name in forecasts:
            if forecasts[name]["input_sha256"] != cost.sha256(case["inputs"]):
                raise RuntimeError("a previously forecast input was changed")
            continue
        exposed = name in records or cost.sha256(case["inputs"]) in set(artifact["exposed_shapes"])
        item = {"kind": "source", "model_sha256": identity, "input_sha256": cost.sha256(case["inputs"]),
                "frozen_utc": datetime.now(timezone.utc).isoformat(), "previously_exposed": exposed}
        try:
            item.update(status="predicted", result=predict(case["inputs"], artifact))
        except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
            item.update(status="unavailable", error=str(error))
        forecasts[name] = item
        if index % 8 == 0 or index == len(plan):
            cost.save_json(snapshot, payload)
            print(f"CPU FORECAST: {index}/{len(plan)}; predictions only, no NPU measurement", flush=True)
    cost.save_json(snapshot, payload)


def report(path, model, forecasts, plan=()):
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        fields, rows = list(reader.fieldnames or []), list(reader)
    planned = {item["name"]: item for item in plan}
    rows = [row for row in rows if row.get("status") != "pending" or row.get("case") in planned]
    existing = {row.get("case") for row in rows}
    for name, case in planned.items():
        if name not in existing:
            rows.append({"case": name, "split": "external", "family": case["family"], "coverage": case.get("coverage", ""),
                         **case["inputs"], **model["hardware"], "status": "pending"})
    columns = ("source_predicted_us", "source_ape_percent", "source_pass_10_percent", "source_evaluation", "source_error",
               "source_model_sha256", "source_host_api_sha256", "source_implementation", "source_cost_policy",
               "source_execution_class", "source_class_cost_status",
               "source_identification_min_us", "source_identification_max_us", "source_identification_width_percent",
               "source_phase_active_sum_ranges_us",
               "source_critical_startup_us", "source_critical_drop_adapter_us", "source_critical_bmm1_us", "source_critical_vec1_us", "source_critical_bmm2_us", "source_critical_vec2_us",
               "source_transferred_pipe_rates", "source_pipeline_us", "source_adapter_us", "source_vec1_calls", "source_vec2_calls",
               "source_schedule_mismatch", "source_geometry_mismatch")
    fields.extend(name for name in columns if name not in fields)
    identity = cost.sha256(model)
    replay_cache = getattr(report, "_prediction_cache", {})
    if replay_cache.get("identity") != (identity, tiler()[1]):
        replay_cache = {"identity": (identity, tiler()[1]), "values": {}}
        report._prediction_cache = replay_cache
    outcomes, prospective, retrospective = [], [], []
    for row in rows:
        row.update({name: "" for name in columns})
        inputs = None
        try:
            inputs = cost.normalize({name: row[name] for name in cost.INPUT_FIELDS})
            route = schedule(inputs, model["hardware"])
            if not modeled_path({"inputs": inputs, "schedule": route}):
                row.update(source_evaluation="outside_model_scope", source_implementation=route["implementation"])
                continue
            item = forecasts.get(row.get("case"))
            if item is not None and item["input_sha256"] != cost.sha256(inputs):
                raise RuntimeError("frozen prediction input differs from measurement")
            if item is not None and item["status"] != "predicted":
                raise RuntimeError(item["error"])
            if item:
                value = item["result"]
            else:
                key = cost.sha256(inputs)
                if key not in replay_cache["values"]:
                    replay_cache["values"][key] = predict(inputs, model)
                value = replay_cache["values"][key]
            row.update(source_predicted_us=format(value["predicted_us"], ".7g"), source_model_sha256=identity,
                       source_host_api_sha256=value["host_api_sha256"], source_implementation=value["input_schedule"]["implementation"],
                       source_cost_policy=value["cost_policy"], source_transferred_pipe_rates="|".join(value["transferred_pipe_rates"]),
                       source_pipeline_us=format(value["pipeline_us"], ".7g"), source_adapter_us=format(value["adapter_us"], ".7g"),
                       source_vec1_calls=value["work"]["vec1_calls"], source_vec2_calls=value["work"]["vec2_calls"])
            row["source_execution_class"] = json.dumps(value["execution_class"], sort_keys=True, separators=(",", ":"))
            row.update(source_identification_min_us=value["identification_range_us"][0], source_identification_max_us=value["identification_range_us"][1],
                       source_identification_width_percent=value["identification_width_percent"],
                       source_phase_active_sum_ranges_us=json.dumps(value["phase_pipe_active_sum_range_us"], sort_keys=True, separators=(",", ":")))
            row.update({f"source_critical_{phase}_us": value.get("critical_path_phase_us", {}).get(phase, "") for phase in PHASES})
            entry = model["classes"].get(value["cost_policy"], {})
            row["source_class_cost_status"] = "|".join(f"{pipe}:{item['status']}" for pipe, item in entry.get("pipes", {}).items())
            derived = value["input_schedule"]
            schedule_bad = [name for name in ("priority", "s1_base", "s2_base", "n_ratio", "total_tasks", "core_num", "split_factor")
                            if row.get(name) not in (None, "") and int(row[name]) != derived[name]]
            geometry_bad = [f"bmm{stage}_{name}" for stage, mm in enumerate(value["matmul"], 1)
                            for name in ("base_m", "base_n", "base_k", "depth_a1", "depth_b1")
                            if row.get(f"bmm{stage}_{name}") not in (None, "") and int(row[f"bmm{stage}_{name}"]) != mm[name]]
            row.update(source_schedule_mismatch="|".join(schedule_bad), source_geometry_mismatch="|".join(geometry_bad))
            before = item and not item["previously_exposed"] and row.get("model_sha256") == identity and str(row.get("prediction_before_measurement")).lower() == "true"
            evaluation = "prospective" if before else "training_replay" if cost.sha256(inputs) in model["training_shapes"] else "previously_exposed_check"
            row["source_evaluation"] = evaluation if row.get("status") == "ok" else "pending" if row.get("status") == "pending" else "not_numerically_validated"
            if row.get("status") == "ok":
                error = abs(value["predicted_us"] / float(row["duration_us"]) - 1) * 100
                passed = error <= 10 and not schedule_bad and not geometry_bad
                row.update(source_ape_percent=format(error, ".7g"), source_pass_10_percent=passed)
                outcome = {"case": row.get("case"), "coverage": row.get("coverage") or row["family"],
                           "evaluation": evaluation, "ape_percent": error, "pass": passed,
                           "predicted_us": value["predicted_us"], "actual_us": float(row["duration_us"]),
                           "stable": str(row.get("stable")).lower() == "true"}
                outcomes.append(outcome)
                (prospective if before else retrospective).append(outcome)
        except (ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            row["source_error"] = str(error)
            if inputs is not None:
                try:
                    design = workload(inputs, model["hardware"])
                    group = cost_group(design)
                    entry = model["classes"].get(group, {})
                    row.update(source_model_sha256=identity, source_implementation=design["schedule"]["implementation"],
                               source_cost_policy=group, source_execution_class=json.dumps(execution_class(design), sort_keys=True, separators=(",", ":")),
                               source_class_cost_status="|".join(f"{pipe}:{item['status']}" for pipe, item in entry.get("pipes", {}).items()) or "unmeasured_class")
                except (ValueError, RuntimeError, subprocess.SubprocessError):
                    pass
    temporary = Path(path).with_suffix(".source.tmp")
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fields} for row in rows)
    temporary.replace(path)

    def metrics(items):
        return error_metrics([{**item, "pass_10_percent": item["pass"]} for item in items])

    targets = {case.get("coverage", case["family"]) for case in plan}
    covered = {item["coverage"] for item in prospective}
    exposed = set(model["exposed_shapes"])
    fresh = {name for name, case in planned.items() if
             (not forecasts[name]["previously_exposed"] if name in forecasts else cost.sha256(case["inputs"]) not in exposed)}
    passed = {item["case"] for item in prospective if item["pass"]}
    prospective_metrics = metrics(prospective)
    prospective_metrics.update(predicted_and_measured=prospective_metrics["count"], count=len(fresh),
                               failed_or_missing=len(fresh - passed), all_pass=bool(fresh) and fresh <= passed)
    summary = {"version": VERSION, "model_sha256": identity, "prospective": metrics(prospective),
               "retrospective": metrics(retrospective), "by_path": {name: metrics([item for item in outcomes if item["coverage"] == name]) for name in sorted({item["coverage"] for item in outcomes})},
               "fresh_blind_pass": bool(fresh) and fresh <= passed and targets <= covered,
               "prospective_expected": len(fresh), "prospective_failed_or_missing": len(fresh - passed),
               "planned_predictions": len(forecasts), "pending": sum(row.get("status") == "pending" for row in rows),
               "missing_forecasts": sum(bool(row.get("source_error")) for row in rows),
               "qualification": "Accuracy is reported as a distribution and signed error per execution class; 10% is a reference target. Existing measurements are retrospective, not a new blind test"}
    summary["prospective"] = prospective_metrics
    cost.save_json(Path(path).parent / "source_model_report.json", summary)
    print(f"SOURCE MODEL: prospective={summary['prospective']}; retrospective={summary['retrospective']}; pending={summary['pending']}; qualified={summary['fresh_blind_pass']}", flush=True)
    return summary


def stage_observations(rows, hardware):
    columns = {'cube': 'aic_cube_sum_us', 'mte1': 'aic_mte1_sum_us', 'mte2': 'aic_mte2_sum_us',
               'fixpipe': 'aic_fixpipe_sum_us', 'vector': 'aiv_vec_sum_us', 'vmte2': 'aiv_mte2_sum_us',
               'vmte3': 'aiv_mte3_sum_us', 'cscalar': 'aic_scalar_sum_us', 'vscalar': 'aiv_scalar_sum_us'}
    designs, result = {}, []
    for row in rows:
        phase = row.get('phase')
        if phase not in ('bmm1', 'bmm2', 'vec1', 'vec2'):
            continue
        if row.get('status') != 'ok' or str(row.get('gating_verified')).lower() != 'true' or int(row.get('checked_elements') or 0) <= 0:
            raise ValueError(f'unverified stage observation: {row.get("case")} {phase}')
        inputs = cost.normalize({name: row[name] for name in cost.INPUT_FIELDS})
        identity = cost.sha256(inputs)
        if identity not in designs:
            designs[identity] = workload(inputs, hardware)
        design = designs[identity]
        if row['query_task'] not in ('first', 'last') or row['key_window'] not in ('single', 'first', 'last'):
            raise ValueError(f'unknown selected stage window: {row["case"]}')
        mismatch = verify_class_trace(row, design)
        if mismatch:
            raise ValueError(f'stage template differs from source model: {row["case"]}: {mismatch}')
        if any(int(row[name]) != int(design['schedule'][name]) for name in
               ('s1_base', 's2_base', 'n_ratio', 'core_num', 'total_tasks', 'split_factor')):
            raise ValueError(f'input-derived schedule differs from stage observation: {row["case"]}')
        windows = cost.ceil_div(inputs['S2'], design['schedule']['s2_base'] * design['schedule']['n_ratio'])
        stage = 0 if phase in ('bmm1', 'vec1') else 1
        selected = []
        for worker in design['workers']:
            offset = 0 if row['query_task'] == 'first' else len(worker) - windows
            selected.append(worker[offset + (windows - 1 if row['key_window'] == 'last' else 0)])
        cube = phase.startswith('bmm')
        jobs = [job['cube'][stage] for job in selected] if cube else [pair[stage] for job in selected for pair in job['vector']]
        pipes = ('cube', 'mte1', 'mte2', 'fixpipe', 'cscalar') if cube else ('vector', 'vmte2', 'vmte3', 'vscalar')
        features = {pipe: {field: sum(job.get(field, 0) for job in jobs) for field in PIPES[pipe][1]} for pipe in pipes}
        targets, intervals = {}, {}
        for pipe in pipes:
            column = columns[pipe]
            samples = [float(value) for value in json.loads(row[column.replace('_us', '_samples_us')])]
            if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
                raise ValueError(f'missing positive replay samples: {row["case"]} {phase} {pipe}')
            targets[pipe] = statistics.median(samples)
            intervals[pipe] = [min(samples), max(samples)]
        read_kb = sum(job.get('read_kb', 0) + job.get('alignment_read_kb', 0) for job in jobs)
        write_kb = sum(job.get('write_kb', 0) for job in jobs)
        result.append({'case': row['case'], 'shape_identity': identity, 'inputs': inputs, 'phase': phase,
                       'class_id': cost_group(design), 'window': row['key_window'], 'query_task': row['query_task'],
                       'active_cube_cores': len(design['workers']), 'active_vector_cores': 2 * len(design['workers']),
                       'features': features, 'targets': targets, 'replay_intervals': intervals,
                       'read_source_kb': read_kb, 'write_source_kb': write_kb,
                       'read_measured_kb': float(row['gm_to_l1_kb' if cube else 'gm_to_ub_kb']),
                       'write_measured_kb': float(row['l0c_to_gm_kb' if cube else 'ub_to_gm_kb']),
                       'other_engine_execution_stopped': False,
                       'gated_core_elapsed_us': float(row['aic_own_table_time_max_us' if cube else 'aiv_own_table_time_max_us']),
                       'runtime_bmm1_output': 'NZ' if design['schedule']['nz'] and selected[0]['columns'] % 64 else 'ND'})
    return result


def stage_model_signature():
    return cost.sha256({'source': source_hash(), 'functions': [inspect.getsource(function) for function in
                       (stage_observations, stage_component_key, stage_feature_groups, stage_model_signature,
                        fit_stage_model, evaluate_stage_features, predict_stage_activity)]})


def stage_component_key(phase, pipe, output_format=None):
    key = phase + '|' + pipe
    return key + '|' + output_format if phase == 'bmm1' and pipe == 'cube' and output_format is not None else key


def stage_feature_groups(design):
    groups = {phase: {pipe: {stage_component_key(phase, pipe): values} for pipe, values in pipes.items()}
              for phase, pipes in phase_features(design).items() if phase in ('bmm1', 'bmm2', 'vec1', 'vec2')}
    groups['bmm1']['cube'] = {}
    units = zip(design['units'], design['unit_counts']) if 'units' in design else ((job, 1) for job in design['jobs'])
    for job, count in units:
        cube = job['cube'][0]
        key = stage_component_key('bmm1', 'cube', 'NZ' if cube['nz_write_kb'] else 'ND')
        values = groups['bmm1']['cube'].setdefault(key, Counter())
        values.update({name: cube.get(name, 0) * count for name in PIPES['cube'][1]})
    return groups


def fit_stage_model(rows, hardware, calibration_cases):
    from scipy.optimize import linprog
    verified, excluded = [], []
    for row in rows:
        if row.get('phase') not in ('bmm1', 'bmm2', 'vec1', 'vec2'):
            continue
        if row.get('status') == 'ok' and str(row.get('gating_verified')).lower() == 'true' and int(row.get('checked_elements') or 0) > 0:
            verified.append(row)
        else:
            excluded.append({'case': row.get('case'), 'phase': row.get('phase'),
                             'reason': row.get('error') or 'gated counters or numerical verification unavailable'})
    observations = stage_observations(verified, hardware)
    calibration_cases = set(calibration_cases)
    selected = [row for row in observations if row['case'] in calibration_cases]
    if not calibration_cases or {row['case'] for row in selected} != calibration_cases:
        raise ValueError('calibration case selection is missing stage observations')
    classes = sorted({row['class_id'] for row in selected})
    if len(classes) != 1:
        raise ValueError('independent stage calibration requires one source execution class at a time')
    components = {}
    for phase, pipe, key in sorted({(row['phase'], pipe, stage_component_key(row['phase'], pipe, row['runtime_bmm1_output']))
                                    for row in selected for pipe in row['features']}):
        records = [row for row in selected if pipe in row['features'] and
                   stage_component_key(row['phase'], pipe, row['runtime_bmm1_output']) == key]
        names = list(PIPES[pipe][1])
        matrix = np.asarray([[row['features'][pipe][name] for name in names] for row in records], dtype=float)
        targets = np.asarray([row['targets'][pipe] for row in records])
        scales = np.maximum(np.max(abs(matrix), axis=0), 1.0)
        normalized = matrix / scales / targets[:, None]
        objective = np.append(np.zeros(len(names)), 1.0)
        relative_constraints = np.vstack((np.column_stack((normalized, -np.ones(len(records)))),
                                          np.column_stack((-normalized, -np.ones(len(records))))))
        solution = linprog(objective, A_ub=relative_constraints,
                           b_ub=np.concatenate((np.ones(len(records)), -np.ones(len(records)))),
                           bounds=(0, None), method='highs')
        if not solution.success:
            raise RuntimeError(f'independent stage fit failed: {phase} {pipe}: {solution.message}')
        intervals = np.asarray([row['replay_intervals'][pipe] for row in records])
        feasibility = linprog(np.zeros(len(names)), A_ub=np.vstack((matrix / scales, -matrix / scales)),
                              b_ub=np.concatenate((intervals[:, 1], -intervals[:, 0])), bounds=(0, None), method='highs')
        if feasibility.status not in (0, 2):
            raise RuntimeError(f'could not determine stage fit consistency: {phase} {pipe}: {feasibility.message}')
        point = solution.x[:-1]
        if feasibility.success:
            identity = np.eye(len(names))
            projection = linprog(np.concatenate((np.zeros(len(names)), np.ones(len(names)))),
                A_ub=np.vstack((np.column_stack((matrix / scales, np.zeros_like(matrix))),
                                np.column_stack((-matrix / scales, np.zeros_like(matrix))),
                                np.column_stack((identity, -identity)), np.column_stack((-identity, -identity)))),
                b_ub=np.concatenate((intervals[:, 1], -intervals[:, 0], point, -point)), bounds=(0, None), method='highs')
            if not projection.success:
                raise RuntimeError(f'could not select a repeat-consistent stage cost: {phase} {pipe}')
            point = projection.x[:len(names)]
        _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
        threshold = max(normalized.shape) * np.finfo(float).eps * singular[0] if len(singular) else 0
        rank = int(np.sum(singular > threshold))
        rates = point / scales
        components[key] = {'phase': phase, 'pipe': pipe, 'features': names,
            'runtime_output_format': key.rsplit('|', 1)[-1] if phase == 'bmm1' and pipe == 'cube' else 'shared',
            'coefficients': {name: float(value) for name, value in zip(names, rates)},
            'feature_scales': scales.tolist(), 'row_basis': vt[:rank].tolist(), 'rank': rank,
            'active_features': int(np.sum(np.any(matrix != 0, axis=0))), 'observations': len(records),
            'minimum_possible_max_relative_error_percent': float(solution.x[-1] * 100),
            'point_max_calibration_error_percent': float(np.max(abs(matrix @ rates / targets - 1)) * 100),
            'consistent_with_replay_intervals': bool(feasibility.success),
            'status': 'repeat_consistent' if feasibility.success else 'work_features_or_execution_context_incomplete',
            'scalar_includes_waits': pipe in ('cscalar', 'vscalar'),
            'maximum_replay_span_percent': float(np.max((intervals[:, 1] - intervals[:, 0]) / targets) * 100)}
    return {'version': 'fa_independent_stage_v2', 'feature_source_sha256': stage_model_signature(),
            'hardware': hardware, 'execution_classes': classes, 'calibration_cases': sorted(calibration_cases),
            'components': components, 'excluded_observations': excluded, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
            'uses_total_kernel_latency_for_stage_fit': False, 'uses_holdout_targets_for_fit': False,
            'other_engine_execution_stopped': False, 'standalone_phase_wall_times_measured': False,
            'full_latency_qualified': False,
            'scope': 'In-context gated engine active counters; phase costs are fitted independently and never adjusted to cancel total-latency error'}


def evaluate_stage_features(features, component):
    names = component['features']
    scaled = np.asarray([features.get(name, 0) / scale for name, scale in zip(names, component['feature_scales'])])
    basis = np.asarray(component['row_basis'])
    projected = scaled @ basis.T @ basis if basis.size else np.zeros(len(names))
    unidentified = float(np.linalg.norm(scaled - projected) / max(np.linalg.norm(scaled), 1e-30))
    prediction = sum(features.get(name, 0) * component['coefficients'][name] for name in names)
    return {'predicted_active_sum_us': prediction, 'unidentified_feature_fraction': unidentified,
            'feature_combination_identified': unidentified <= 1e-7,
            'fit_consistent_with_repeats': component['consistent_with_replay_intervals'],
            'scalar_includes_waits': component['scalar_includes_waits'],
            'use_as_calibrated_cost': unidentified <= 1e-7 and component['consistent_with_replay_intervals']
                                      and not component['scalar_includes_waits'],
            'component_status': component['status']}


def predict_stage_activity(inputs, model):
    if model.get('version') != 'fa_independent_stage_v2' or model['feature_source_sha256'] != stage_model_signature():
        raise RuntimeError('independent stage model and feature source differ')
    design = workload(inputs, model['hardware'])
    if cost_group(design) not in model['execution_classes']:
        raise ValueError('execution class has no independent stage calibration')
    phases = {}
    for phase, pipes in stage_feature_groups(design).items():
        phases[phase] = {}
        for pipe, groups in pipes.items():
            forecasts = {key: evaluate_stage_features(features, model['components'][key]) if key in model['components'] else
                         {'predicted_active_sum_us': None, 'component_status': 'missing_stage_calibration',
                          'use_as_calibrated_cost': False} for key, features in groups.items()}
            complete = all(item['predicted_active_sum_us'] is not None for item in forecasts.values())
            phases[phase][pipe] = {'components': forecasts,
                'predicted_active_sum_us': sum(item['predicted_active_sum_us'] for item in forecasts.values()) if complete else None,
                'use_as_calibrated_cost': all(item['use_as_calibrated_cost'] for item in forecasts.values())}
    return {'inputs': design['inputs'], 'schedule': design['schedule'], 'phases': phases,
            'full_latency_qualified': False, 'total_latency_us': None,
            'scope': 'Whole-kernel phase active sums inferred from inputs; no total latency is inferred from incomplete stage costs'}


def audit_stage_model(rows, model, previous_model=None):
    verified = [row for row in rows if row.get('status') == 'ok' and str(row.get('gating_verified')).lower() == 'true'
                and int(row.get('checked_elements') or 0) > 0]
    observations = stage_observations(verified, model['hardware'])
    output = []
    for row in observations:
        split = 'calibration' if row['case'] in model['calibration_cases'] else 'holdout'
        shared = {'case': row['case'], 'phase': row['phase'], 'split': split, 'key_window': row['window'],
                  'query_task': row['query_task'], 'active_cube_cores': row['active_cube_cores'],
                  'active_vector_cores': row['active_vector_cores'], 'runtime_bmm1_output': row['runtime_bmm1_output'],
                  'validation_is_new_blind_test': False,
                  'other_engine_execution_stopped': False, 'full_latency_qualified': False}
        for pipe, features in row['features'].items():
            component_key = stage_component_key(row['phase'], pipe, row['runtime_bmm1_output'])
            component = model['components'].get(component_key)
            if component is None:
                output.append({**shared, 'record_type': 'independent_stage_activity', 'pipe': pipe,
                               'component_status': 'missing_stage_calibration', 'actual_us': row['targets'][pipe]})
                continue
            forecast = evaluate_stage_features(features, component)
            prediction, actual = forecast['predicted_active_sum_us'], row['targets'][pipe]
            record = {**shared, 'record_type': 'independent_stage_activity', 'pipe': pipe,
                      'component_key': component_key,
                      'actual_us': actual, 'candidate_us': prediction,
                      'candidate_ape_percent': 100 * abs(prediction / actual - 1),
                      'signed_error_percent': 100 * (prediction / actual - 1), 'error_us': prediction - actual,
                      'replay_min_us': row['replay_intervals'][pipe][0], 'replay_max_us': row['replay_intervals'][pipe][1],
                      'minimum_possible_calibration_error_percent': component['minimum_possible_max_relative_error_percent'],
                      'fit_consistent_with_repeats': forecast['fit_consistent_with_repeats'],
                      'rank': component['rank'], 'active_features': component['active_features'],
                      'identified_feature_combination': forecast['feature_combination_identified'],
                      'use_as_calibrated_cost': forecast['use_as_calibrated_cost'],
                      'component_status': forecast['component_status'],
                      'features_json': json.dumps(features, sort_keys=True, separators=(',', ':')),
                      'validation_scope': 'selected_query_task_engine_active_sum; total_kernel_time_never_used_to_fit_stage_costs'}
            if previous_model is not None:
                previous = previous_model['classes'][row['class_id']]['pipes'][pipe]['coefficients']
                value = sum(features.get(name, 0) * coefficient for name, coefficient in previous.items())
                record.update(previous_us=value, previous_ape_percent=100 * abs(value / actual - 1))
            output.append(record)
        for direction in ('read', 'write'):
            actual, predicted = row[direction + '_measured_kb'], row[direction + '_source_kb']
            output.append({**shared, 'record_type': 'independent_stage_work_volume', 'pipe': direction,
                           'actual_kb': actual, 'source_kb': predicted,
                           'volume_error_percent': 100 * abs(predicted / actual - 1) if actual else None,
                           'validation_scope': 'source_bytes_checked_against_gated_memory_counters'})
    output.extend({'record_type': 'excluded_stage_observation', **row} for row in model['excluded_observations'])
    output.extend(audit_stage_replays(rows, model))
    return output


def audit_stage_replays(rows, model, accuracy_budget_percent=10.0):
    budget = accuracy_budget_percent / 100
    if not 0 < budget < 1:
        raise ValueError('repeatability accuracy budget must be between zero and one hundred percent')
    verified = [row for row in rows if row.get('status') == 'ok' and str(row.get('gating_verified')).lower() == 'true'
                and int(row.get('checked_elements') or 0) > 0]
    observations = {(row['case'], row['phase'], row['window']): row
                    for row in stage_observations(verified, model['hardware'])}
    columns = {'cube': 'aic_cube', 'mte1': 'aic_mte1', 'mte2': 'aic_mte2', 'fixpipe': 'aic_fixpipe',
               'cscalar': 'aic_scalar', 'vector': 'aiv_vec', 'vmte2': 'aiv_mte2', 'vmte3': 'aiv_mte3', 'vscalar': 'aiv_scalar'}
    output = []
    for saved in rows:
        observation = observations.get((saved['case'], saved['phase'], saved['key_window']))
        metrics = {'task_duration': json.loads(saved.get('task_duration_samples_us') or '[]')}
        if observation is not None:
            metrics.update({pipe: json.loads(saved[columns[pipe] + '_sum_samples_us']) for pipe in observation['features']})
        for pipe, samples in metrics.items():
            if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
                continue
            intervals = [(value / (1 + budget), value / (1 - budget)) for value in samples]
            support = max(sum(low <= point <= high for low, high in intervals) for point in sorted({end for pair in intervals for end in pair}))
            record = {'record_type': 'repeatability_check', 'case': saved['case'], 'phase': saved['phase'],
                'key_window': saved['key_window'], 'pipe': pipe, 'repeat_count': len(samples),
                'samples_us_json': json.dumps(samples, separators=(',', ':')), 'median_us': statistics.median(samples),
                'minimum_us': min(samples), 'maximum_us': max(samples),
                'repeat_span_percent': 100 * (max(samples) - min(samples)) / statistics.median(samples),
                'accuracy_budget_percent': accuracy_budget_percent, 'largest_consistent_repeat_count': support,
                'majority_consistent_with_accuracy_budget': support > len(samples) / 2,
                'discarded_repeats': 0, 'phase_cost_fit_uses_this_metric': observation is not None and
                    saved['case'] in model['calibration_cases'] and pipe != 'task_duration',
                'validation_scope': 'measurement-only interval consistency; no sample is deleted using prediction error'}
            if observation is not None and pipe != 'task_duration':
                key = stage_component_key(observation['phase'], pipe, observation['runtime_bmm1_output'])
                component = model['components'].get(key)
                if component is not None:
                    prediction = evaluate_stage_features(observation['features'][pipe], component)['predicted_active_sum_us']
                    deletion_medians = [statistics.median(samples[:index] + samples[index + 1:]) for index in range(len(samples))] if len(samples) > 1 else samples
                    errors = [100 * abs(prediction / value - 1) for value in deletion_medians]
                    record.update(candidate_us=prediction, candidate_ape_percent=100 * abs(prediction / statistics.median(samples) - 1),
                        best_single_repeat_ape_percent=min(100 * abs(prediction / value - 1) for value in samples),
                        leave_one_repeat_out_min_ape_percent=min(errors), leave_one_repeat_out_max_ape_percent=max(errors),
                        sensitivity_uses_refitted_coefficients=False)
            output.append(record)
    return output


def audit_stage_counters(rows, model):
    observations = stage_observations(rows, model['hardware'])
    raw = {(row['case'], row['phase'], row['key_window']): row for row in rows}
    metric_columns = {name for saved in rows for columns in json.loads(saved.get('raw_metric_column_names') or '{}').values()
                      for name in columns}
    calibration = [row for row in observations if row['case'] in model['calibration_cases']]
    output = []
    for row in observations:
        if row['phase'] not in ('bmm1', 'bmm2'):
            continue
        saved = raw[row['case'], row['phase'], row['window']]
        counters = json.loads(saved.get('counter_diagnostics_json') or '{}').get('ArithmeticUtilization/aic', {})
        issued = counters.get('aic_cube_total_instr_number', {}).get('core_sum')
        pmu_fops = counters.get('aic_cube_fops', {}).get('core_sum')
        design = workload(row['inputs'], model['hardware'])
        peers = [peer for peer in calibration if peer['phase'] == row['phase']]
        write_features = row['features']['fixpipe']
        equal_writes = [peer for peer in peers if peer['features']['fixpipe'] == write_features]
        intervals = [peer['replay_intervals']['fixpipe'] for peer in equal_writes]
        medians = [peer['targets']['fixpipe'] for peer in equal_writes]
        repeat_overlap = max(pair[0] for pair in intervals) <= min(pair[1] for pair in intervals) if intervals else None
        output.append({'record_type': 'source_counter_diagnosis', 'case': row['case'], 'phase': row['phase'],
            'split': 'calibration' if row['case'] in model['calibration_cases'] else 'holdout',
            'key_window': row['window'], 'query_task': row['query_task'],
            'runtime_bmm1_output': row['runtime_bmm1_output'], 'active_cube_cores': row['active_cube_cores'],
            'source_mmad_instructions': row['features']['cube']['tiles'], 'pmu_mmad_instructions': issued,
            'mmad_instruction_count_matches': issued == row['features']['cube']['tiles'] if issued is not None else None,
            'source_flops': row['features']['cube']['flops_m'] * 1e6, 'pmu_fops_field': pmu_fops,
            'source_to_pmu_fops_ratio': row['features']['cube']['flops_m'] * 1e6 / pmu_fops if pmu_fops else None,
            'matmul_geometry_json': json.dumps(design['matmul'][0 if row['phase'] == 'bmm1' else 1], sort_keys=True, separators=(',', ':')),
            'calibration_query_tasks_json': json.dumps(sorted({peer['query_task'] for peer in peers})),
            'calibration_active_cube_cores_json': json.dumps(sorted({peer['active_cube_cores'] for peer in peers})),
            'query_position_seen_in_calibration': any(peer['query_task'] == row['query_task'] for peer in peers),
            'active_core_count_seen_in_calibration': any(peer['active_cube_cores'] == row['active_cube_cores'] for peer in peers),
            'equal_write_feature_observations': len(equal_writes), 'equal_write_repeat_intervals_overlap': repeat_overlap,
            'equal_write_minimum_max_error_percent': 100 * (max(medians) - min(medians)) / (max(medians) + min(medians)) if medians else None,
            'direct_wait_stall_counter_present': any(re.search(r'wait|stall|conflict|cflt', name, re.I) for name in metric_columns) if metric_columns else None,
            'shared_global_memory_queue_in_timing_graph': False,
            'validation_scope': 'source_work_and_counter_consistency; no cost or holdout target is fitted'})
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, help="B,Nq,Nkv,S1,S2,Dqk,Dv")
    parser.add_argument("--dtype", choices=("FP16", "BF16"), default="FP16")
    parser.add_argument("--layout", choices=("BNSD", "SBH", "BSND", "TND"), default="BNSD")
    parser.add_argument("--drop-mask", action="store_true")
    parser.add_argument("--model", type=Path, default=ROOT / "fa_sameab_minimum.json")
    args = parser.parse_args()
    try:
        shape = tuple(map(int, args.shape.split(",")))
        if len(shape) != 7:
            raise ValueError("shape requires seven integers: B,Nq,Nkv,S1,S2,Dqk,Dv")
        inputs = dict(zip(cost.INPUT_FIELDS[:7], shape))
        inputs.update(dtype=args.dtype, layout=args.layout, has_drop_mask=int(args.drop_mask))
        model = json.loads(args.model.read_text())
        result = predict(inputs, model)
        print(json.dumps({"inputs": cost.normalize(inputs), **result,
                          "validation": model.get("validation"),
                          "accuracy_guaranteed": False}, ensure_ascii=False, separators=(",", ":")))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
