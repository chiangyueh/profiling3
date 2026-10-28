from __future__ import annotations

import csv
import hashlib
import heapq
import inspect
import json
import math
import os
import subprocess
import statistics
from datetime import datetime, timezone
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import fa_latency_inputs as cost
import fit_fa_latency as legacy


VERSION = "fa_source_pipeline_v2"
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
    "cscalar": ("aic_scalar_active_sum_us", ("workers", "commands")),
    "vscalar": ("aiv_scalar_active_sum_us", ("workers", "chunks")),
}


def source_hash():
    functions = (schedule, mm_requests, matmul_geometry, vector_job, cube_job, workload, features, service, cost_group, elapsed, adapter, predict)
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
    nz2 = windowed or p["DV"] % 16 != 0
    width = align(p["DV"])
    stats = rows * (1 if (s["same_ab"] and s["s1_base"] > 256) or (p["layout"] == "TND" and s["s1_base"] > 128) else 8) * 4
    drop_bytes = rows * (align(columns, 32) if s["drop_adapter"] else ceil(align(columns, 32), 8)) if p["has_drop_mask"] else 0
    stage1 = {"softmax_k": soft_elements / 1000, "reduction_k": soft_rows * math.ceil(math.log2(max(2, padded))) / 1000,
              "cast_k": rows * padded / 1000, "format_k": rows * padded * (int(nz1) + int(probability_nz)) / 1000,
              "online_k": (0 if first else rows * 8) / 1000, "drop_k": rows * padded / 1000 if p["has_drop_mask"] else 0,
              "chunks": c1, "calls": 1, "read_kb": (4 * rows * align(columns) + drop_bytes) / 1024,
              "write_kb": (2 * rows * align(columns) + 2 * stats) / 1024,
              "loads": c1 * (1 + int(bool(p["has_drop_mask"]))), "stores": c1 * 3,
              "vec1_calls": c1, "softmax_initial_calls": c1 if first else 0, "softmax_update_calls": 0 if first else c1}
    output_elements = rows * width
    stage2 = {"softmax_k": 0, "reduction_k": 0, "cast_k": output_elements / 1000 if last and p["dtype"] == "BF16" or last and windowed else 0,
              "format_k": output_elements / 1000 if nz2 else 0, "online_k": output_elements * (int(not first) * 2 + int(last)) / 1000,
              "drop_k": 0, "chunks": c2, "calls": 1,
              "read_kb": ((4 if windowed or p["dtype"] == "BF16" or batch else 2) * output_elements + stats + (0 if first else 4 * output_elements)) / 1024,
              "write_kb": (2 * rows * p["DV"] if last else 4 * output_elements) / 1024,
              "loads": c2 * (2 + int(not first)), "stores": c2,
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
    output_width = 2 if stage == 2 and s["family"] == "s1" else 4
    return {"flops_m": 2 * heads * align(m) * align(n) * align(k) / 1e6, "tiles": tiles,
            "l0_kb": heads * 2 * (align(m) * align(k) * nt + align(n) * align(k) * mt) / 1024,
            "read_kb": (a_bytes + b_bytes) / 1024,
            "write_kb": heads * m * align(n) * output_width / 1024,
            "loads": heads * ceil(mt * kt, geometry["depth_a1"]) + (0 if shared or carried else kv_heads * ceil(nt * kt, geometry["depth_b1"])),
            "stores": heads * mt * nt, "calls": 1, "commands": 1 + tiles,
            "query_rows": heads * rows}


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
            heads, kv_heads = (p["N1"], p["N2"]) if batch else (1, 1)
            for offset in range(0, p["S2"], window):
                columns = min(window, p["S2"] - offset)
                key = batch_id, head // (p["N1"] // p["N2"]), offset
                cube = []
                for stage in (0, 2):
                    shared = s["reuse"] and client == 1 and previous.get(stage) == key
                    carry_api = s["l1"] and (stage == 2 or p["layout"] == "TND")
                    carried = carry_api and previous.get((stage, client)) == key
                    cube.append(cube_job(p, s, mm[stage // 2], stage, rows, columns, heads, kv_heads, shared, carried))
                    previous[stage], previous[stage, client] = key, key
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
    totals.update(cube_workers=cube_pairs, vector_workers=aiv_workers,
                  bmm1_calls=len(jobs), bmm2_calls=len(jobs))
    return {"inputs": p, "schedule": s, "matmul": mm, "workers": workers, "jobs": jobs, "totals": dict(totals)}


def features(design, pipe):
    if "pipe_features" in design:
        return design["pipe_features"][pipe]
    result = Counter()
    cube = pipe in ("cube", "mte1", "mte2", "fixpipe", "cscalar")
    for job in design["jobs"]:
        for item in job["cube"] if cube else (item for pair in job["vector"] for item in pair):
            result.update({name: item.get(name, 0) for name in PIPES[pipe][1] if name != "workers"})
    if "workers" in PIPES[pipe][1]:
        result["workers"] = design["totals"]["cube_workers" if cube else "vector_workers"]
    return result


def service(job, pipe, model):
    return sum(value * job.get(name, 0) for name, value in model["pipes"][pipe]["coefficients"].items())


def cost_group(design):
    p, s = design["inputs"], design["schedule"]
    family = "tnd_same_ab" if s["same_ab"] and p["layout"] == "TND" else "tnd_general" if p["layout"] == "TND" else "b" if s["family"] in ("b", "b_drop") else s["family"]
    return family + "/" + ("UNSPLITK" if s["policy"] == "UNSPLITK" else "IBSHARE" if s["reuse"] else "L1_CARRY" if s["l1"] else "GM") + ("/STATIC" if s.get("static_matmul") else "")


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

        startup = max(model["pipes"]["cscalar"]["coefficients"].get("workers", 0), model["pipes"]["vscalar"]["coefficients"].get("workers", 0))
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
    elements = p["B"] * p["N1"] * p["S1"] * p["S2"]
    single = cost.ceil_div(cost.ceil_div(elements, 2048), int(h["aiv_cores"])) * 2048
    available = cost.ceil_div(int(h["ub_bytes"]), 6) // 2048 * 2048
    base = min(single, available)
    passes = cost.ceil_div(single, base)
    job = {"cast_k": single / 1000, "drop_k": single / 1000, "chunks": passes, "calls": passes,
           "read_kb": single / 8 / 1024, "write_kb": single / 1024, "loads": passes, "stores": passes}
    duration = max(service(job, "vector", model), service(job, "vmte2", model) + service(job, "vmte3", model), service(job, "vscalar", model))
    return {"adapter_us": duration, "adapter_passes": passes}


def predict(inputs, model):
    if model["version"] != VERSION or model["source_sha256"] != source_hash():
        raise RuntimeError("source model and frozen predictor differ; native caches are unaffected")
    design = workload(inputs, model["hardware"])
    group = cost_group(design)
    rates = {**model, "pipes": {**model["pipes"], **model.get("policies", {}).get(group, {})}}
    result = elapsed(design, rates)
    extra = adapter(design, rates)
    value = result["pipeline_us"] + extra["adapter_us"]
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("invalid source-model latency")
    return {**result, **extra, "predicted_us": value, "input_schedule": design["schedule"],
            "matmul": design["matmul"], "work": design["totals"], "cost_policy": group,
            "prediction_quality": "source_frozen_before_measurement", "model_version": VERSION,
            "host_api_sha256": tiler()[1], "qualification": "unqualified_until_independent_per_case_error_is_at_most_10_percent",
            "unidentified_costs": model["unidentified_costs"], "transferred_pipe_rates": sorted(set(PIPES) - set(model.get("policies", {}).get(group, {}))),
            "out_of_calibration_range": [name for name, bounds in model["input_ranges"].items() if not bounds[0] <= design["inputs"][name] <= bounds[1]]}


def fit(paths, previous_model):
    allowed = set(previous_model["training_shapes"])
    selected = {}
    for path in paths:
        with Path(path).open(newline="") as source:
            for row in csv.DictReader(source):
                if row.get("status") != "ok" or str(row.get("stable")).lower() != "true" or not legacy.pipeline_counters_complete(row):
                    continue
                p = cost.normalize({name: row[name] for name in cost.INPUT_FIELDS})
                identity = cost.sha256(p)
                if identity not in allowed:
                    continue
                if any(str(row[name]) != str(previous_model["hardware"][name]) for name in previous_model["hardware"] if row.get(name)):
                    raise ValueError("calibration hardware differs")
                if not all(row.get(name) for name in ("soc", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes")):
                    raise ValueError("calibration platform identity is incomplete")
                if any(not row.get(PIPES[pipe][0]) for pipe in PIPES):
                    continue
                mm = [{name: int(row[f"bmm{stage}_{name}"]) for name in ("base_m", "base_n", "base_k", "depth_a1", "depth_b1")} for stage in (1, 2)]
                design = workload(p, previous_model["hardware"], calibration_geometry=mm)
                if design["schedule"]["drop_adapter"]:
                    continue
                compact = {name: design[name] for name in ("inputs", "schedule", "totals")}
                compact["pipe_features"] = {pipe: features(design, pipe) for pipe in PIPES}
                selected[identity] = row, compact
    if not selected:
        raise ValueError("original frozen PMU calibration rows are missing")
    rows = list(selected.values())
    result = {"version": VERSION, "source_sha256": source_hash(), "hardware": previous_model["hardware"], "pipes": {},
              "training_shapes": sorted(selected), "exposed_shapes": sorted(set(previous_model["exposed_shapes"]) | set(selected)),
              "input_ranges": {name: [min(d["inputs"][name] for _, d in rows), max(d["inputs"][name] for _, d in rows)] for name in cost.INPUT_FIELDS[:7]},
              "target": "FlashAttentionScore kernel Task Duration(us)", "host_api_sha256_at_calibration": tiler()[1],
              "calibration": "Only original v1 training identities with stable, numerically checked PMU; no FA duration targets; no October 28 prospective rows used",
              "calibration_geometry": "Original official Host Matmul geometry from calibration CSV; inference independently invokes CPU Matmul tiling from operator inputs",
              "schedule": "Official Host task partitions, two Vector clients per Cube, shared Matmul FIFO, SameAB paired Vector barriers and two drain iterations",
              "cost_model": "Global nonnegative per-pipe PMU rates; no branch-specific FA latency multiplier, no shape-to-latency table",
              "scope": "Dense FA forward FP16/BF16, BNSD/SBH/BSND, uniform-length TND, optional dropout; no mask/PSE/Rope/sink/FP32/FAG",
              "limitations": ["Aggregate per-pipe service is calibrated, not an instruction-accurate hardware simulator",
                              "L2 contention, memory coalescing and runtime clock/thermal interference are not determined exactly by shape",
                              "Adapter global barrier and Matmul internal overlap are approximated by the resource dependency model",
                              "Static Matmul variants have no numerically successful calibration and remain unqualified"],
              "qualification": "not_validated_to_10_percent"}
    for pipe, (target, names) in PIPES.items():
        matrix = [[features(d, pipe).get(name, 0) for name in names] for _, d in rows]
        observed = [float(r[target]) for r, _ in rows]
        fitted = legacy.pipeline_nnls(matrix, observed)
        coefficients = dict(zip(names, fitted.pop("weights")))
        estimates = [sum(value * coefficients[name] for value, name in zip(line, names)) for line in matrix]
        result["pipes"][pipe] = {"coefficients": coefficients, **fitted,
                                 "calibration_mape_percent": statistics.mean(abs(v / y - 1) * 100 for v, y in zip(estimates, observed))}
    result["unidentified_costs"] = [pipe for pipe, item in result["pipes"].items() if item["rank"] < len(PIPES[pipe][1])]
    result["policies"] = {}
    for group in sorted({cost_group(design) for _, design in rows}):
        subset = [(row, design) for row, design in rows if cost_group(design) == group]
        rates = {}
        if len(subset) < 2:
            continue
        for pipe, (target, names) in PIPES.items():
            matrix = [[features(d, pipe).get(name, 0) for name in names] for _, d in subset]
            fitted = legacy.pipeline_nnls(matrix, [float(row[target]) for row, _ in subset])
            if fitted["rank"] == len(names):
                rates[pipe] = {"coefficients": dict(zip(names, fitted.pop("weights"))), **fitted,
                               "calibration_shapes": sorted(cost.sha256(d["inputs"]) for _, d in subset)}
        result["policies"][group] = rates
    result["cost_model"] = "Per-pipe PMU rates for source Matmul/Vector policies when identifiable; otherwise transferred global rates; no FA latency multiplier or shape-to-latency table"
    return result


def evaluate(paths, model):
    outcomes = []
    for path in paths:
        with Path(path).open(newline="") as source:
            for row in csv.DictReader(source):
                if row.get("status") != "ok":
                    continue
                p = cost.normalize({name: row[name] for name in cost.INPUT_FIELDS})
                identity = cost.sha256(p)
                try:
                    result = predict(p, model)
                    error = abs(result["predicted_us"] / float(row["duration_us"]) - 1) * 100
                    outcomes.append({"case": row.get("case", ""), "identity": identity, "coverage": row.get("coverage") or row["family"],
                                     "evaluation": "training_replay" if identity in model["training_shapes"] else "exposed_structural_check",
                                     "actual_us": float(row["duration_us"]), "predicted_us": result["predicted_us"], "ape_percent": error,
                                     "stable": str(row.get("stable")).lower() == "true"})
                except (ValueError, RuntimeError) as error:
                    outcomes.append({"case": row.get("case", ""), "error": str(error)})
    unique = {r.get("identity", r["case"]): r for r in outcomes}
    outcomes = list(unique.values())
    summaries = {}
    for group in ("all", "training_replay", "exposed_structural_check"):
        rows = [r for r in outcomes if group == "all" or r.get("evaluation") == group]
        errors = [r["ape_percent"] for r in rows if "ape_percent" in r]
        summaries[group] = {"count": len(rows), "predicted": len(errors), "pass_10_percent": sum(e <= 10 for e in errors),
                            "mape_percent": statistics.mean(errors) if errors else None, "max_ape_percent": max(errors, default=None)}
    return {"summary": summaries, "cases": outcomes, "prospective_pass": False}


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
        if index % 32 == 0 or index == len(plan):
            cost.save_json(snapshot, payload)
            print(f"CPU FORECAST: {index}/{len(plan)}; predictions only, no NPU measurement", flush=True)
    cost.save_json(snapshot, payload)


def report(path, model, forecasts, plan=()):
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        fields, rows = list(reader.fieldnames or []), list(reader)
    planned = {item["name"]: item for item in plan}
    existing = {row.get("case") for row in rows}
    for name, case in planned.items():
        if name not in existing:
            rows.append({"case": name, "split": "external", "family": case["family"], "coverage": case.get("coverage", ""),
                         **case["inputs"], **model["hardware"], "status": "pending"})
    columns = ("source_predicted_us", "source_ape_percent", "source_pass_10_percent", "source_evaluation", "source_error",
               "source_model_sha256", "source_host_api_sha256", "source_implementation", "source_cost_policy",
               "source_transferred_pipe_rates", "source_pipeline_us", "source_adapter_us", "source_vec1_calls", "source_vec2_calls",
               "source_schedule_mismatch", "source_geometry_mismatch")
    fields.extend(name for name in columns if name not in fields)
    identity = cost.sha256(model)
    outcomes, prospective, retrospective = [], [], []
    for row in rows:
        row.update({name: "" for name in columns})
        try:
            inputs = cost.normalize({name: row[name] for name in cost.INPUT_FIELDS})
            item = forecasts.get(row.get("case"))
            if item is not None and item["input_sha256"] != cost.sha256(inputs):
                raise RuntimeError("frozen prediction input differs from measurement")
            if item is not None and item["status"] != "predicted":
                raise RuntimeError(item["error"])
            value = item["result"] if item else predict(inputs, model)
            row.update(source_predicted_us=format(value["predicted_us"], ".7g"), source_model_sha256=identity,
                       source_host_api_sha256=value["host_api_sha256"], source_implementation=value["input_schedule"]["implementation"],
                       source_cost_policy=value["cost_policy"], source_transferred_pipe_rates="|".join(value["transferred_pipe_rates"]),
                       source_pipeline_us=format(value["pipeline_us"], ".7g"), source_adapter_us=format(value["adapter_us"], ".7g"),
                       source_vec1_calls=value["work"]["vec1_calls"], source_vec2_calls=value["work"]["vec2_calls"])
            schedule = value["input_schedule"]
            schedule_bad = [name for name in ("priority", "s1_base", "s2_base", "n_ratio", "total_tasks", "core_num", "split_factor")
                            if row.get(name) not in (None, "") and int(row[name]) != schedule[name]]
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
    summary = {"version": VERSION, "model_sha256": identity, "prospective": metrics(prospective),
               "retrospective": metrics(retrospective), "by_path": {name: metrics([item for item in outcomes if item["coverage"] == name]) for name in sorted({item["coverage"] for item in outcomes})},
               "fresh_blind_pass": metrics(prospective)["all_pass"] and targets <= covered,
               "planned_predictions": len(forecasts), "pending": sum(row.get("status") == "pending" for row in rows),
               "missing_forecasts": sum(bool(row.get("source_error")) for row in rows),
               "qualification": "Every measured point must pass 10%; existing measurements are retrospective, not a new blind test"}
    cost.save_json(Path(path).parent / "source_model_report.json", summary)
    print(f"SOURCE MODEL: prospective={summary['prospective']}; retrospective={summary['retrospective']}; pending={summary['pending']}; qualified={summary['fresh_blind_pass']}", flush=True)
    return summary
