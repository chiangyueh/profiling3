from __future__ import annotations

import csv
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


VERSION = "fa_source_pipeline_v3"
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "fa_source_model.json"
SOURCE = "attention/flash_attention_score/"
PIPES = {
    "cube": ("aic_cube_active_sum_us", ("flops_m", "tiles")),
    "mte1": ("aic_mte1_active_sum_us", ("l0_kb", "tiles")),
    "mte2": ("aic_mte2_active_sum_us", ("read_kb", "loads")),
    "fixpipe": ("aic_fixpipe_active_sum_us", ("write_kb", "stores")),
    "vector": ("aiv_vec_active_sum_us", ("softmax_k", "reduction_k", "cast_k", "format_k", "online_k", "drop_k", "chunks")),
    "vmte2": ("aiv_mte2_active_sum_us", ("read_kb", "loads")),
    "vmte3": ("aiv_mte3_active_sum_us", ("write_kb", "stores")),
    "cscalar": ("aic_scalar_active_sum_us", ("workers", "batch_entries", "commands")),
    "vscalar": ("aiv_scalar_active_sum_us", ("workers", "batch_entries", "chunks")),
}


def source_hash():
    functions = (schedule, mm_requests, matmul_geometry, vector_job, cube_job, adapter_jobs, reuse_fake_pairs, workload, features, atomic_features, cost_vertices, service, execution_class, modeled_path, cost_group, phase_features, elapsed, adapter, predict, predict_design)
    return cost.sha256({"version": VERSION, "pipes": PIPES, "python": [inspect.getsource(function) for function in functions],
                        "host": (ROOT / "fa_matmul_tiling.cpp").read_text(),
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
    output_elements = rows * width
    stage2 = {"softmax_k": 0, "reduction_k": 0, "cast_k": output_elements / 1000 if last else 0,
              "format_k": output_elements / 1000 if nz2 else 0, "online_k": output_elements * (int(not first) * 2 + int(last)) / 1000,
              "drop_k": 0, "chunks": c2, "calls": 1,
              "read_kb": (4 * output_elements + (0 if windowed else stats) + (0 if first else 4 * output_elements)) / 1024,
              "write_kb": ((2 * rows * p["DV"] if last else 4 * output_elements) + (stats if windowed and last else 0)) / 1024,
              "loads": c2 * (1 + int(not windowed) + int(not first)), "stores": c2 + ((1 if s["same_ab"] else c2) if windowed and last else 0),
              "vec2_calls": c2, "accumulator_read_bytes": 0 if first else 4 * output_elements,
              "accumulator_write_bytes": 0 if last else 4 * output_elements,
              "final_output_bytes": 2 * rows * p["DV"] if last else 0}
    return stage1, stage2


def cube_job(p, s, geometry, stage, rows, columns, heads, kv_heads, shared, carried):
    align, ceil = cost.align, cost.ceil_div
    m, n, k = rows, columns if stage == 0 else p["DV"], p["D"] if stage == 0 else columns
    bm, bn, bk = (geometry[name] for name in ("base_m", "base_n", "base_k"))
    mt, nt, kt = ceil(m, bm), ceil(n, bn), ceil(k, bk)
    tiles = heads * mt * nt * kt
    a_bytes = heads * m * align(k, 16) * 2
    b_bytes = kv_heads * columns * align(p["D"] if stage == 0 else p["DV"], 16) * 2
    if shared or carried:
        b_bytes = 0
    output_width = 4
    return {"flops_m": 2 * heads * align(m) * align(n) * align(k) / 1e6, "tiles": tiles,
            "l0_kb": heads * 2 * (align(m) * align(k) * nt + align(n) * align(k) * mt) / 1024,
            "read_kb": (a_bytes + b_bytes) / 1024,
            "write_kb": heads * m * align(n) * output_width / 1024,
            "loads": heads * ceil(mt * kt, geometry["depth_a1"]) + (0 if shared or carried else kv_heads * ceil(nt * kt, geometry["depth_b1"])),
            "stores": heads * mt * nt, "calls": 1, "commands": 1 + tiles,
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
            jobs.append({"cast_k": elements / 1000, "drop_k": elements / 1000, "chunks": chunks, "calls": chunks,
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


def workload(inputs, hardware, calibration_geometry=None):
    p = cost.normalize(inputs)
    s = schedule(p, hardware)
    mm = matmul_geometry(tuple(mm_requests(p, s, hardware))) if calibration_geometry is None else calibration_geometry
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
                for stage in (0, 2):
                    shared = s["reuse"] and client == 1 and previous.get(("shared", stage, offset)) == key
                    carry_api = s["l1"] and (stage == 2 or p["layout"] == "TND")
                    carried = carry_api and previous.get(("carried", stage, client)) == key
                    cube.append(cube_job(p, s, mm[stage // 2], stage, rows, columns, heads, kv_heads, shared, carried))
                    previous["shared", stage, offset], previous["carried", stage, client] = key, key
                vector = []
                for owned in ((cost.ceil_div(rows, 2), rows // 2) if s["same_ab"] else (rows,)):
                    if owned == 0:
                        vector.append(({}, {}))
                        continue
                    v1, v2 = vector_job(p, s, owned, columns, offset == 0, offset + columns == p["S2"])
                    if batch:
                        for v in (v1, v2):
                            for name in v:
                                v[name] *= heads
                    vector.append((v1, v2))
                    for v in (v1, v2):
                        totals.update({name: v.get(name, 0) for name in ("vec1_calls", "vec2_calls", "accumulator_read_bytes", "accumulator_write_bytes", "final_output_bytes")})
                unit = {"cube": cube, "vector": vector, "columns": columns, "rows": rows}
                clients[client].append(unit)
                jobs.append(unit)
        workers.extend(clients)
    cube_pairs = len(workers) if s["same_ab"] else cost.ceil_div(len(workers), 2)
    aiv_workers = 2 * len(workers) if s["same_ab"] else len(workers)
    totals.update(cube_workers=int(hardware["aic_cores"]) if s["drop_adapter"] else cube_pairs,
                  vector_workers=int(hardware["aiv_cores"]) if s["drop_adapter"] else aiv_workers,
                  bmm1_calls=len(jobs), bmm2_calls=len(jobs))
    return {"inputs": p, "schedule": s, "matmul": mm, "workers": workers, "jobs": jobs, "totals": dict(totals),
            "adapter_jobs": adapter_jobs(p, s, hardware), "unpaired_reuse_segments": reuse_fake_pairs(p, s)}


def features(design, pipe):
    if "pipe_features" in design:
        return design["pipe_features"][pipe]
    result = Counter()
    cube = pipe in ("cube", "mte1", "mte2", "fixpipe", "cscalar")
    for job in design["jobs"]:
        for item in job["cube"] if cube else (item for pair in job["vector"] for item in pair):
            result.update({name: item.get(name, 0) for name in PIPES[pipe][1] if name != "workers"})
    if not cube:
        for job in design["adapter_jobs"]:
            result.update({name: job.get(name, 0) for name in PIPES[pipe][1]})
    if "workers" in PIPES[pipe][1]:
        result["workers"] = design["totals"]["cube_workers" if cube else "vector_workers"]
        result["batch_entries"] = result["workers"] * design["inputs"]["B"] if design["inputs"]["layout"] == "TND" else 0
    return result


def atomic_features(design, pipe):
    cube = pipe in ("cube", "mte1", "mte2", "fixpipe", "cscalar")
    names = PIPES[pipe][1]
    values = set()
    for job in design["jobs"]:
        for item in job["cube"] if cube else (item for pair in job["vector"] for item in pair):
            values.add(tuple(float(item.get(name, 0)) for name in names))
    if not cube:
        values.update(tuple(float(job.get(name, 0)) for name in names) for job in design["adapter_jobs"])
    if "workers" in names:
        startup = {"workers": 1, "batch_entries": design["inputs"]["B"] if design["inputs"]["layout"] == "TND" else 0}
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
    return s["same_ab"] and p["dtype"] == "BF16" and p["layout"] == "BNSD" and s["policy"] == "UNSPLITK" and not p["has_drop_mask"]


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
        return "same_ab_bf16_bnsd_unsplitk"
    return "|".join(f"{name}={value}" for name, value in execution_class(design).items())


def phase_features(design):
    result = {stage: {pipe: Counter() for pipe in pipes} for stage, pipes in
              (("bmm1", ("cube", "mte1", "mte2", "fixpipe", "cscalar")),
               ("bmm2", ("cube", "mte1", "mte2", "fixpipe", "cscalar")),
               ("vec1", ("vector", "vmte2", "vmte3", "vscalar")),
               ("vec2", ("vector", "vmte2", "vmte3", "vscalar")))}
    for job in design["jobs"]:
        for index, stage in enumerate(("bmm1", "bmm2")):
            for pipe in result[stage]:
                result[stage][pipe].update({name: job["cube"][index].get(name, 0) for name in PIPES[pipe][1]})
        for pair in job["vector"]:
            for index, stage in enumerate(("vec1", "vec2")):
                for pipe in result[stage]:
                    result[stage][pipe].update({name: pair[index].get(name, 0) for name in PIPES[pipe][1]})
    result["startup"] = {"cscalar": {"workers": design["totals"]["cube_workers"], "batch_entries": 0},
                         "vscalar": {"workers": design["totals"]["vector_workers"], "batch_entries": 0}}
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
        raise ValueError("outside the modeled SameAB BF16 BNSD UNSPLITK path")
    group = cost_group(design)
    entry = model["classes"].get(group)
    if entry is None:
        raise ValueError(f"unmeasured execution class: {group}")
    if design["unpaired_reuse_segments"]:
        raise ValueError("unmodeled paired-Matmul empty-message control cost")
    results, costs = [], []
    for mode in ("lower", "upper"):
        rates = {"hardware": model["hardware"], "pipes": entry["pipes"], "strict_classes": True, "class_id": group, "cost_mode": mode}
        results.append(elapsed(design, rates))
        costs.append({stage: {pipe: service(job, pipe, rates) for pipe, job in pipes.items()} for stage, pipes in phase_features(design).items()})
    lower, upper = (result["pipeline_us"] for result in results)
    if lower > upper + 1e-7:
        raise RuntimeError("source timing range is inconsistent")
    value = 2 * lower * upper / (lower + upper)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("invalid source-model latency")
    return {"pipeline_us": value, "adapter_us": 0.0, "predicted_us": value, "input_schedule": design["schedule"],
            "matmul": design["matmul"], "work": design["totals"], "cost_policy": group,
            "identification_range_us": [lower, upper], "identification_width_percent": 100 * (upper - lower) / (upper + lower),
            "phase_pipe_active_sum_range_us": {stage: {pipe: [costs[0][stage][pipe], costs[1][stage][pipe]] for pipe in costs[0][stage]} for stage in costs[0]},
            "range_scope": "Nonnegative decompositions with the same fitted class PMU totals; this is not a confidence interval and excludes counter-fit error and missing physical effects",
            "point_rule": "harmonic mean of the two pipeline limits minimizes worst relative error within that range",
            "prediction_quality": "source_frozen_before_measurement", "model_version": VERSION,
            "host_api_sha256": tiler()[1] if geometry_origin == "cpu_matmul_tiler" else geometry_origin,
            "qualification": "unqualified_until_independent_per_case_error_is_at_most_10_percent",
            "unidentified_costs": [pipe for pipe, fitted in entry["pipes"].items() if fitted.get("unidentified_atomic_jobs")],
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


def load_cost_records(paths, hardware, fieldnames=None):
    unique = {}
    for path in paths:
        with Path(path).open(newline="") as source:
            schema = fieldnames.get(Path(path).name) if isinstance(fieldnames, dict) else fieldnames
            reader = csv.DictReader(source, fieldnames=schema)
            required = set(cost.INPUT_FIELDS) | {"status", "stable", "duration_us", "checked_elements", "tiling_key"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f"measurement CSV has no compatible header; explicit fieldnames are required: {path}")
            for raw in reader:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(f"measurement CSV row does not match its header: {path}, line={reader.line_num}")
                if raw.get("status") != "ok":
                    continue
                inputs = cost.normalize({name: raw[name] for name in cost.INPUT_FIELDS})
                identity = cost.sha256(inputs)
                if any(str(raw[name]) != str(hardware[name]) for name in hardware if raw.get(name)):
                    raise ValueError("calibration measurements use another hardware profile")
                if int(raw.get("checked_elements") or 0) <= 0:
                    continue
                duration = float(raw["duration_us"])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"successful measurement has an invalid latency: {path}, line={reader.line_num}")
                geometry = [{name: int(raw[f"bmm{stage}_{name}"]) for name in ("base_m", "base_n", "base_k", "depth_a1", "depth_b1")} for stage in (1, 2)]
                design = workload(inputs, hardware, calibration_geometry=geometry)
                mismatch = {name: [design["schedule"][name], int(raw[name])]
                            for name in ("priority", "s1_base", "s2_base", "n_ratio", "total_tasks", "core_num", "split_factor")
                            if raw.get(name) and design["schedule"][name] != int(raw[name])}
                mismatch.update(verify_class_trace(raw, design))
                targets = {}
                for pipe, (field, _) in PIPES.items():
                    try:
                        value = float(raw[field])
                        if math.isfinite(value) and value > 0:
                            targets[pipe] = value
                    except (KeyError, TypeError, ValueError):
                        pass
                unique[identity] = {"identity": identity, "inputs": inputs, "geometry": geometry,
                                    "measurement_file": Path(path).name,
                                    "trace": {name: int(raw[name]) for name in ("tiling_key", "priority", "s1_base", "s2_base", "n_ratio", "total_tasks", "core_num", "split_factor")},
                                    "checked_elements": int(raw["checked_elements"]),
                                    "case": raw.get("case") or identity, "coverage": raw.get("coverage") or raw["family"],
                                    "stable": str(raw.get("stable")).lower() == "true", "duration_us": duration,
                                    "class_id": cost_group(design), "path": execution_class(design), "schedule_mismatch": mismatch,
                                    "fold": int(cost.sha256({name: inputs[name] for name in ("S1", "S2")})[:8], 16) % 5,
                                    "features": {pipe: dict(features(design, pipe)) for pipe in PIPES},
                                    "atoms": {pipe: atomic_features(design, pipe) for pipe in PIPES}, "targets": targets,
                                    "byte_checks": {"gm_to_l1_kb": [features(design, "mte2").get("read_kb", 0), raw.get("gm_to_l1_kb")],
                                                    "l0c_to_gm_kb": [features(design, "fixpipe").get("write_kb", 0), raw.get("l0c_to_gm_kb")]}}
    return list(unique.values())


def fit_class_costs(records, hardware):
    classes = {}
    for group in sorted({row["class_id"] for row in records}):
        observed = [row for row in records if row["class_id"] == group]
        selected = [row for row in observed if row["stable"] and not row["schedule_mismatch"]]
        entry = {"path": observed[0]["path"], "measured_shapes": len(observed), "stable_shapes": len(selected), "pipes": {},
                 "pipeline_control_modeled": not observed[0]["path"]["reuse_fake_peer"],
                 "sequence_pairs": len({(row["inputs"]["S1"], row["inputs"]["S2"]) for row in selected}),
                 "input_ranges": {name: [min(row["inputs"][name] for row in selected), max(row["inputs"][name] for row in selected)] for name in cost.INPUT_FIELDS[:7]} if selected else {}}
        for pipe, (_, names) in PIPES.items():
            rows = [row for row in selected if pipe in row["targets"]]
            if not rows:
                entry["pipes"][pipe] = {"status": "unavailable", "reason": "no stable counter measurements in this execution class", "samples": 0}
                continue
            matrix = np.asarray([[row["features"][pipe].get(name, 0) for name in names] for row in rows], dtype=float)
            targets = np.asarray([row["targets"][pipe] for row in rows])
            scales = np.maximum(np.max(abs(matrix), axis=0), 1.0)
            _, singular, basis = np.linalg.svd(matrix / scales, full_matrices=False)
            rank = int(np.sum(singular > max(float(singular[0]), 1e-30) * 1e-9))
            basis = basis[:rank]
            if not rank:
                entry["pipes"][pipe] = {"status": "unavailable", "reason": "source work is zero but the measured pipe is active", "samples": len(rows)}
                continue
            atom = np.asarray(sorted({tuple(values) for row in rows for values in row["atoms"][pipe]}), dtype=float)
            normalized = atom / scales
            residuals = np.linalg.norm(normalized - normalized @ basis.T @ basis, axis=1) / np.maximum(np.linalg.norm(normalized, axis=1), 1e-30)
            fitted = legacy.pipeline_nnls(matrix.tolist(), targets.tolist())
            weights = fitted.pop("weights")
            estimates = matrix @ weights
            errors = abs(estimates / targets - 1) * 100
            unidentified = int(np.sum(residuals > 1e-7))
            rate_residuals = np.linalg.norm(np.eye(len(names)) - basis.T @ basis, axis=1)
            entry["pipes"][pipe] = {"status": "identified_stage_costs" if unidentified == 0 else "joint_counter_only",
                                    "coefficients": dict(zip(names, weights)), "feature_scales": scales.tolist(), "row_basis": basis.tolist(),
                                    "identified_feature_rates": [name for name, residual in zip(names, rate_residuals) if residual <= 1e-7],
                                    "unresolved_feature_rates": [name for index, (name, residual) in enumerate(zip(names, rate_residuals)) if residual > 1e-7 and np.any(matrix[:, index])],
                                    "inactive_feature_rates": [name for index, name in enumerate(names) if not np.any(matrix[:, index])],
                                    "samples": len(rows), "sequence_pairs": len({(row["inputs"]["S1"], row["inputs"]["S2"]) for row in rows}),
                                    "rank": rank, "feature_count": len(names), "unidentified_atomic_jobs": unidentified,
                                    "atomic_jobs": len(atom), "calibration_mape_percent": float(np.mean(errors)),
                                    "calibration_max_ape_percent": float(np.max(errors)),
                                    "calibration_shapes": sorted(row["identity"] for row in rows)}
            entry["pipes"][pipe]["feasible_rate_vertices"] = cost_vertices(pipe, entry["pipes"][pipe])
        entry["all_stage_costs_identified"] = all(item["status"] == "identified_stage_costs" for item in entry["pipes"].values())
        classes[group] = entry
    return {"version": VERSION, "source_sha256": source_hash(), "hardware": hardware, "classes": classes,
            "training_shapes": sorted(row["identity"] for row in records if row["stable"] and not row["schedule_mismatch"]),
            "exposed_shapes": sorted(row["identity"] for row in records),
            "source_evidence": source_evidence(),
            "target": "FlashAttentionScore kernel Task Duration(us)", "global_fallback": False,
            "scope": "Dense FP16/BF16 FA; BNSD/SBH/BSND and uniform TND; optional dropout; no sparse/PSE/Rope/sink/FAG",
            "calibration": "Class-specific PMU only; all supplied stable development measurements; no FA latency target is fitted",
            "work_audit": {"s1_bmm2": "Kernel C2 is FP32, including FP16 inputs; Host bmm2OutDtype is not the kernel output width",
                           "general_bmm2": "Dense General outputs ND; TND General and SameAB output NZ",
                           "windowed_stats": "Final K window writes max and sum; Vec2 consumes sum from UB",
                           "l1_reuse": "Paired clients share matching K/V windows; reuse is tracked separately for each S2 window",
                           "fake_peer": "General pairs Q tasks within each head and may send an empty peer message; these classes are separate and their empty-message pipeline cost is not yet modeled",
                           "tnd_sameab": "Its ComputeConstexpr retains reduction width 8; TND General has a separate reduction-width branch",
                           "traffic": "Operand-byte estimates are checked against PMU; API reloads and memory transactions are not assumed proven by a matching Host route"},
            "identifiability": "Within the stated linear workload basis, uniquely determined costs are distinguished from joint counter fits. Non-unique phase costs retain all feasible nonnegative decompositions and are propagated as timing ranges. No rates are borrowed across execution classes. These ranges exclude counter-fit error and missing physical effects; they are not confidence intervals or an accuracy guarantee.",
            "classification": "Official implementation, input dtype/layout, Matmul policy, actual ND/NZ windows, second-Matmul output format, operand path, probability format, dropout representation, reduction mode, online update and padding branches",
            "qualification": "development_calibration_requires_grouped_validation_and_new_frozen_shape_test"}


def fit(paths, previous_model):
    records = load_cost_records(paths, previous_model["hardware"])
    selected = [row for row in records if row["class_id"] == "same_ab_bf16_bnsd_unsplitk"]
    result = fit_class_costs(selected, previous_model["hardware"])
    result["exposed_shapes"] = sorted(row["identity"] for row in records)
    result["scope"] = "SameAB BF16 BNSD UNSPLITK without dropout, sparse mask, PSE, Rope or FAG; tails change counted work rather than selecting a fitted shape exception"
    return result


def evaluate(paths, model, input_only=False, fieldnames=None):
    if model["version"] != VERSION or model["source_sha256"] != source_hash():
        raise RuntimeError("source model and frozen predictor differ")
    records = load_cost_records(paths, model["hardware"], fieldnames=fieldnames)
    outcomes, exposed = [], set(model["exposed_shapes"])
    for index, row in enumerate(records, 1):
        outcome = evaluate_record(row, model)
        outcome.update(previously_exposed=row["identity"] in exposed, known_class=row["class_id"] in model["classes"])
        if input_only:
            result = {"ape_percent": None, "predicted_us": None, "geometry_mismatch": []}
            try:
                design = workload(row["inputs"], model["hardware"])
                result["geometry_mismatch"] = [f"bmm{stage}_{name}" for stage, (actual, expected) in enumerate(zip(design["matmul"], row["geometry"]), 1)
                                               for name in expected if actual[name] != expected[name]]
                predicted = predict_design(design, model)["predicted_us"]
                error = abs(predicted / row["duration_us"] - 1) * 100
                result.update(predicted_us=predicted, ape_percent=error, pass_10_percent=error <= 10 and not result["geometry_mismatch"] and not row["schedule_mismatch"])
            except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
                result["error"] = str(error)
            outcome["input_only"] = result
        outcomes.append(outcome)
        if index % 64 == 0:
            print(f"OFFLINE EXTERNAL CHECK: {index}/{len(records)}; frozen costs, no NPU", flush=True)
    groups = {"all": outcomes, "stable": [row for row in outcomes if row["stable"]],
              "unseen": [row for row in outcomes if not row["previously_exposed"]],
              "unseen_stable": [row for row in outcomes if not row["previously_exposed"] and row["stable"]],
              "repeated": [row for row in outcomes if row["previously_exposed"]]}
    summary = {name: {"host_geometry_replay": error_metrics(rows),
                      "input_only_local_sdk": error_metrics([row["input_only"] for row in rows]) if input_only else None,
                      "unknown_classes": sum(not row["known_class"] for row in rows),
                      "geometry_mismatches": sum(bool(row.get("input_only", {}).get("geometry_mismatch")) for row in rows),
                      "host_trace_mismatches": sum(bool(row["schedule_mismatch"]) for row in rows)} for name, rows in groups.items()}
    return {"summary": summary, "cases": outcomes, "prospective_pass": False,
            "new_data_used_for_fit": False, "frozen_model_sha256": cost.sha256(model),
            "evaluation": "Unseen-file holdout is separate from repeated development inputs; Host-geometry replay and input-only installed-SDK predictions are separate"}


def error_metrics(rows):
    errors = [row["ape_percent"] for row in rows if row.get("ape_percent") is not None]
    passed = sum(row.get("pass_10_percent", row["ape_percent"] <= 10) for row in rows if row.get("ape_percent") is not None)
    return {"count": len(rows), "predicted": len(errors), "unavailable": len(rows) - len(errors),
            "pass_10_percent": passed, "fail_10_percent_including_unavailable": len(rows) - passed,
            "mape_percent": statistics.mean(errors) if errors else None, "max_ape_percent": max(errors, default=None),
            "all_pass": bool(rows) and passed == len(rows)}


def evaluate_record(row, model):
    result = {"case": row["case"], "identity": row["identity"], "class_id": row["class_id"], "stable": row["stable"],
              "actual_us": row["duration_us"], "ape_percent": None, "predicted_us": None,
              "inputs": row["inputs"], "execution_class": row["path"], "geometry": row["geometry"], "trace": row["trace"],
              "checked_elements": row["checked_elements"], "schedule_mismatch": row["schedule_mismatch"], "byte_checks": row["byte_checks"],
              "aggregate_pipes": {}, "geometry_origin": "recorded_v8.5.0_host_tiling"}
    entry = model["classes"].get(row["class_id"])
    for pipe, actual in row["targets"].items():
        counter = {"actual_us": actual, "ape_percent": None, "predicted_us": None}
        try:
            rates = {"class_id": row["class_id"], "strict_classes": True, "pipes": entry["pipes"] if entry else {}}
            value = service(row["features"][pipe], pipe, rates)
            counter.update(predicted_us=value, ape_percent=abs(value / actual - 1) * 100)
        except ValueError as error:
            counter["error"] = str(error)
        result["aggregate_pipes"][pipe] = counter
    try:
        if row["schedule_mismatch"]:
            raise ValueError("source classification differs from the recorded Host trace")
        design = workload(row["inputs"], model["hardware"], calibration_geometry=row["geometry"])
        value = predict_design(design, model, geometry_origin=result["geometry_origin"])["predicted_us"]
        result.update(predicted_us=value, ape_percent=abs(value / row["duration_us"] - 1) * 100)
    except (ValueError, RuntimeError) as error:
        result["error"] = str(error)
    return result


def assess_costs(paths, previous_model, output, fieldnames=None):
    paths = [Path(path) for path in paths]
    hardware = previous_model["hardware"]
    records = load_cost_records(paths, hardware, fieldnames=fieldnames)
    selected = [row for row in records if row["class_id"] == "same_ab_bf16_bnsd_unsplitk"]
    if not selected:
        raise ValueError("no numerically successful measurements of the modeled SameAB path")
    model = fit_class_costs(selected, hardware)
    model["scope"] = "SameAB BF16 BNSD UNSPLITK without dropout, attention mask, PSE, Rope or FAG; shape tails change counted work, not fitted shape exceptions"
    model["exposed_shapes"] = sorted(row["identity"] for row in records)
    model["development_data"] = [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    validation, replay = {}, {}
    groups = sorted({(row["inputs"]["S1"], row["inputs"]["S2"]) for row in selected})
    designs = {row["identity"]: workload(row["inputs"], hardware) for row in selected}
    for index, pair in enumerate(groups, 1):
        train = [row for row in selected if (row["inputs"]["S1"], row["inputs"]["S2"]) != pair]
        held = [row for row in selected if (row["inputs"]["S1"], row["inputs"]["S2"]) == pair]
        folded = fit_class_costs(train, hardware)
        assert not {row["identity"] for row in held} & set(folded["training_shapes"])
        for row in held:
            outcome = {"ape_percent": None, "predicted_us": None, "pass_10_percent": False,
                       "fold_model_sha256": cost.sha256(folded), "stable": row["stable"], "held_sequence_pair": pair}
            try:
                design = designs[row["identity"]]
                bad = [f"bmm{stage}_{name}" for stage, (actual, expected) in enumerate(zip(design["matmul"], row["geometry"]), 1)
                       for name in expected if actual[name] != expected[name]]
                prediction = predict_design(design, folded)
                error = abs(prediction["predicted_us"] / row["duration_us"] - 1) * 100
                outcome.update(prediction, ape_percent=error, pass_10_percent=error <= 10 and not bad and not row["schedule_mismatch"], geometry_mismatch=bad)
            except (ValueError, RuntimeError) as error:
                outcome["error"] = str(error)
            validation[row["identity"]] = outcome
        print(f"SAMEAB COST VALIDATION: held S1/S2={pair}; group={index}/{len(groups)}; input-only CPU prediction, no NPU", flush=True)
    for row in selected:
        replay[row["identity"]] = predict_design(designs[row["identity"]], model)
    metrics = error_metrics(list(validation.values()))
    model["assessment"] = {"evaluation": "development_leave_one_complete_sequence_pair_out; all B/head variants of a sequence pair are held together",
                           "measured_shapes": len(selected), "sequence_groups": len(groups), "input_only": metrics,
                           "geometry_mismatches": sum(bool(item.get("geometry_mismatch")) for item in validation.values()),
                           "host_trace_mismatches": sum(bool(row["schedule_mismatch"]) for row in selected),
                           "host_api_sha256": tiler()[1], "host_api_profile": str(cann_root()),
                           "prospective_pass": False, "new_measurements_required_for_blind_validation": True,
                           "classification_only_shapes": len(records) - len(selected),
                           "classified_successful_shapes": len(records), "execution_combinations": len({row["class_id"] for row in records})}
    model["classification"] = "One SameAB BF16 BNSD UNSPLITK implementation; source tail conditions change counted work. Other kernel paths are classified for diagnostics and have no fitted latency model here."
    model["phase_costs"] = "Partial reconstruction from class-specific PMU constraints, with all feasible nonnegative rate decompositions retained; phase active sums are not isolated phase wall times"
    model["qualification"] = "development_holdout_passed_pending_new_shapes" if metrics["all_pass"] else "development_holdout_failed"
    model["stage_measurement"] = {"available": True, "current_data_contains_isolated_stage_times": False,
                                  "timestamp_api": "https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0282.html",
                                  "async_completion": "https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0641.html",
                                  "interpretation": "Device timestamps must distinguish submission, completion and waiting. Additional barriers or tracing affect the original overlap; isolated stage wall times cannot simply be added as kernel latency."}
    unsuccessful, data_status = {}, {}
    for path in paths:
        schema = fieldnames.get(path.name) if isinstance(fieldnames, dict) else fieldnames
        with path.open(newline="") as source:
            raw_rows = list(csv.DictReader(source, fieldnames=schema))
        data_status[path.name] = dict(Counter(row.get("status", "missing") for row in raw_rows))
        for raw in raw_rows:
            if raw.get("status") in ("ok", "covered", "pending"):
                continue
            inputs = cost.normalize({name: raw[name] for name in cost.INPUT_FIELDS})
            identity = cost.sha256(inputs)
            if identity not in set(model["exposed_shapes"]):
                unsuccessful[identity] = {"case": raw.get("case") or identity, "input_sha256": identity,
                                          "measurement_file": path.name, "status": raw["status"], "model_scope": "classification_only",
                                          "family": raw.get("family", ""),
                                          **inputs, **hardware, "stage": raw.get("stage", ""), "reason": raw.get("error", "")[:240]}
    model["data_status_counts"] = data_status
    model["exposed_shapes"] = sorted(set(model["exposed_shapes"]) | set(unsuccessful))
    model["assessment"]["non_successful_inputs"] = len(unsuccessful)
    future = validation_inputs(hardware)
    assert not set(model["exposed_shapes"]) & {cost.sha256(inputs) for inputs in future}
    model["next_validation"] = {"count": len(future), "input_sha256": [cost.sha256(inputs) for inputs in future],
                                "variations": ["batch", "MHA/GQA", "head count and core waves", "S1/S2", "Q tail", "K window tail", "Dv including unaligned output"],
                                "pass_rule": "Every new numerically valid point must have absolute relative latency error at most 10%; failures and unavailable predictions cannot be dropped"}
    rows = []
    for row in records:
        scoped = row["identity"] in validation
        item = {"case": row["case"], "input_sha256": row["identity"], "measurement_file": row["measurement_file"],
                "status": "ok", "family": row["path"]["kernel"], "model_scope": "modeled_path" if scoped else "classification_only", "stable": row["stable"],
                "checked_elements": row["checked_elements"], **row["inputs"], **hardware, **row["trace"],
                **{f"bmm{stage}_{name}": value for stage, geometry in enumerate(row["geometry"], 1) for name, value in geometry.items()},
                **row["path"], "host_trace_matches": not row["schedule_mismatch"], "duration_us": row["duration_us"]}
        for pipe, target in PIPES.items():
            item[target[0]] = row["targets"].get(pipe)
        for name, (value, measured) in row["byte_checks"].items():
            item[name] = measured
            item[name + "_source"] = value
            item[name + "_ape_percent"] = abs(value / float(measured) - 1) * 100 if measured and float(measured) > 0 else None
        if scoped:
            outcome, fitted = validation[row["identity"]], replay[row["identity"]]
            item.update(prediction_origin="development_sequence_holdout", predicted_us=outcome["predicted_us"], ape_percent=outcome["ape_percent"],
                        pass_10_percent=outcome["pass_10_percent"], held_sequence_pair=str(outcome["held_sequence_pair"]),
                        fold_model_sha256=outcome["fold_model_sha256"], replay_us=fitted["predicted_us"],
                        replay_ape_percent=100 * abs(fitted["predicted_us"] / row["duration_us"] - 1),
                        identification_min_us=outcome.get("identification_range_us", [None, None])[0],
                        identification_max_us=outcome.get("identification_range_us", [None, None])[1],
                        identification_width_percent=outcome.get("identification_width_percent"),
                        phase_pipe_active_sum_range_us=json.dumps(outcome.get("phase_pipe_active_sum_range_us", {}), separators=(",", ":")),
                        reason=outcome.get("error", ""))
        rows.append(item)
    for index, inputs in enumerate(future, 1):
        identity = cost.sha256(inputs)
        prediction = predict(inputs, model)
        rows.append({"case": "fa_branch_same_ab_unsplitk_" + identity[:12], "input_sha256": identity, "status": "pending",
                     "model_scope": "modeled_path", "family": "same_ab", **inputs, **hardware, **prediction["execution_class"],
                     "prediction_origin": "frozen_before_future_npu_measurement", "predicted_us": prediction["predicted_us"],
                     "identification_min_us": prediction["identification_range_us"][0], "identification_max_us": prediction["identification_range_us"][1],
                     "identification_width_percent": prediction["identification_width_percent"],
                     "phase_pipe_active_sum_range_us": json.dumps(prediction["phase_pipe_active_sum_range_us"], separators=(",", ":")),
                     "out_of_calibration_range": "|".join(prediction["out_of_calibration_range"])})
        if index % 12 == 0:
            print(f"SAMEAB CPU FORECAST: {index}/{len(future)} new inputs; no NPU measurement", flush=True)
    rows.extend(unsuccessful.values())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".tmp").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(dict.fromkeys(name for row in rows for name in row)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".tmp").replace(output)
    model["assessment"]["csv_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return model


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
        return {"count": len(items), "pass_10_percent": sum(item["pass"] for item in items),
                "mape_percent": statistics.mean(item["ape_percent"] for item in items) if items else None,
                "max_ape_percent": max((item["ape_percent"] for item in items), default=None),
                "all_pass": bool(items) and all(item["pass"] for item in items)}

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
               "qualification": "Every measured point must pass 10%; existing measurements are retrospective, not a new blind test"}
    summary["prospective"] = prospective_metrics
    cost.save_json(Path(path).parent / "source_model_report.json", summary)
    print(f"SOURCE MODEL: prospective={summary['prospective']}; retrospective={summary['retrospective']}; pending={summary['pending']}; qualified={summary['fresh_blind_pass']}", flush=True)
    return summary
