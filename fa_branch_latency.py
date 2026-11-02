from __future__ import annotations

import copy
import csv
import hashlib
import heapq
import json
import inspect
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

import fa_latency_model as source


ROOT = Path(__file__).resolve().parent
VERSION = "fa_branch_execution_v1"
MODEL = ROOT / "fa_branch_model.json"
FILES = {"b": ("flash_attention_score_bn2gs1s2_b.h", 544),
         "s1": ("flash_attention_score_s1_bn2gs1.h", 880),
         "general": ("flash_attention_score_s1s2_bn2gs1.h", 770),
         "varlen_gm": ("flash_attention_var_len_score.h", 473),
         "varlen_l1": ("flash_attention_var_len_score.h", 560),
         "varlen_same_ab": ("flash_attention_var_len_score_sab.h", 435),
         "same_ab": ("flash_attention_score_s1s2_bn2gs1_sab.h", 810),
         "drop_adapter": ("flash_attention_score_drop_mask_adapter.h", 83)}


def signature():
    return source.cost.sha256({"version": VERSION, "python": Path(__file__).read_text(),
                              "matmul_trace": (ROOT / "fa_branch_matmul_tiling.cpp").read_text(),
                              "primitives": source.source_hash(), "evidence": source.source_evidence()})


@lru_cache(maxsize=1)
def tiler():
    sdk = source.cann_root()
    cpp = ROOT / "fa_branch_matmul_tiling.cpp"
    library = next(sdk / "lib64" / name for name in ("libtiling_api.so", "libtiling_api.a") if (sdk / "lib64" / name).is_file())
    identity = source.cost.sha256({"source": cpp.read_text(), "api": hashlib.sha256(library.read_bytes()).hexdigest()})
    folder = ROOT / "out/fa_branch_host"
    folder.mkdir(parents=True, exist_ok=True)
    binary, marker = folder / "matmul_tiling", folder / "signature"
    if not binary.is_file() or not marker.is_file() or marker.read_text() != identity:
        command = ["g++", "-O0", "-std=c++17", str(cpp), "-I", str(sdk / "include"),
                   "-I", str(sdk / "include/ascendc/host_api"), "-L", str(sdk / "lib64"),
                   "-Wl,-rpath," + str(sdk / "lib64"), "-ltiling_api", "-lplatform", "-lregister", "-lascendalog",
                   "-lgraph_base", "-ldl", "-lc_sec", "-o", str(binary) + ".tmp"]
        built = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        if built.returncode:
            raise RuntimeError("CPU Matmul trace helper failed: " + built.stdout[-2000:])
        Path(str(binary) + ".tmp").replace(binary)
        marker.write_text(identity)
    return str(binary), identity


@lru_cache(maxsize=8192)
def geometry(requests):
    binary, _ = tiler()
    result = subprocess.run([str(binary)], input="\n".join(" ".join(map(str, row)) for row in requests) + "\n",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
    names = ("base_m", "base_n", "base_k", "depth_a1", "depth_b1", "db_l0c", "step_m", "step_n",
             "db_l0a", "db_l0b", "step_ka", "step_kb", "iterate_order", "trans_length")
    lines = result.stdout.strip().splitlines()
    if result.returncode or len(lines) != 2 or any(len(line.split()) != len(names) for line in lines):
        raise ValueError("CPU Matmul API rejected branch geometry: " + result.stderr[-500:])
    return [dict(zip(names, map(int, line.split()))) for line in lines]


def implementation(p, s):
    if p["layout"] == "TND":
        return "varlen_same_ab" if s["same_ab"] else "varlen_l1" if s["l1"] else "varlen_gm"
    return "b" if s["family"] in ("b", "b_drop") else s["family"]


def cost_class(p, s):
    family = implementation(p, s)
    path = "paired_reuse" if s["reuse"] else "l1_carry" if s["l1"] else "gm"
    return "|".join((family, p["dtype"], p["layout"], path, "NZ" if s["nz"] else "ND",
                     "STATIC" if s.get("static_matmul") else s["policy"]))


def _fake():
    return {"cube": [{"commands": 1}, {"commands": 1}], "vector": [({}, {})],
            "rows": 0, "columns": 0, "fake": True}


class Work:
    def __init__(self, inputs, hardware):
        self.p = source.cost.normalize(inputs)
        self.s = source.schedule(self.p, hardware)
        if self.p["layout"] == "TND" and not self.s["same_ab"]:
            self.s["l1"] = self.s["s1_base"] <= 128 and max(self.p["D"], self.p["DV"]) <= 256
        self.hardware = hardware
        self.mm = geometry(tuple(source.mm_requests(self.p, self.s, hardware)))
        a, b = self.mm
        static = (self.p["dtype"] == "BF16" and not self.p["has_drop_mask"]
                  and self.p["layout"] in ("BSND", "SBH") and self.s["family"] in ("b", "s1"))
        static = static and (a["base_m"], a["base_n"], a["base_k"]) == (b["base_m"], b["base_k"], b["base_n"])
        templates = ((16, 16, 80), (32, 32, 80), (48, 48, 80), (64, 64, 80), (16, 16, 96), (32, 32, 96))
        static = static and ((a["base_m"], a["base_n"], a["base_k"]) in templates if self.s["family"] == "b"
                             else a["base_m"] == a["base_n"] == 128 and a["base_k"] in (80, 96, 128))
        self.s["static_matmul"] = bool(static and (self.p["layout"] == "BSND" or a["base_k"] == 96))
        self.previous = {}
        self.cache = {}
        self.pairs = []

    def job(self, task, offset, client, group):
        p, s = self.p, self.s
        batch_mode = s["family"] in ("b", "b_drop")
        rows = p["S1"] if batch_mode else min(s["s1_base"], p["S1"] - task % s["s1_outer"] * s["s1_base"])
        head = 0 if batch_mode else task // s["s1_outer"] % p["N1"]
        batch = task if batch_mode else task // (s["s1_outer"] * p["N1"])
        heads, kv_heads = (p["N1"], p["N2"]) if batch_mode else (1, 1)
        window = s["s2_base"] * s["n_ratio"] if s["same_ab"] or s["family"] in ("general", "varlen") else p["S2"]
        columns = min(window, p["S2"] - offset)
        cubes, sharing = [], []
        key = batch, head // (p["N1"] // p["N2"]), offset
        for stage in (0, 1):
            shared = bool(s["reuse"] and client == 1 and self.previous.get((group, stage, "pair", offset)) == key)
            carried = bool(s["family"] == "s1" and s["l1"] and stage == 1
                           and self.previous.get((group, client, "value")) == key[:2])
            q_carried = p["layout"] == "TND" and s["l1"] and stage == 0 and offset > 0
            job = source.cube_job(p, s, self.mm[stage], stage * 2, rows, columns, heads, kv_heads,
                                  shared, carried, q_carried)
            if p["layout"] == "TND" and s["l1"] and not s["same_ab"]:
                parts = [min(256, columns - first) for first in range(0, columns, 256)]
                job["loads"] = (0 if q_carried else 1) + len(parts) if stage == 0 else 2 * len(parts)
                job["calls"] = len(parts)
                job["commands"] = len(parts) + job["tiles"]
                if stage == 1:
                    job["k_slices"] = [source.cost.align(part) / sum(map(source.cost.align, parts)) for part in parts]
            if p["layout"] == "TND" and s["l1"] and stage == 0:
                job["output_group_tiles"] = source.cost.ceil_div(rows, self.mm[stage]["base_m"]) * source.cost.ceil_div(256, self.mm[stage]["base_n"])
            cubes.append(job)
            sharing.append((shared, carried, q_carried))
            self.previous[group, stage, "pair", offset] = key
        self.previous[group, client, "value"] = key[:2]
        vectors = []
        for owned in ((source.cost.ceil_div(rows, 2), rows // 2) if s["same_ab"] else (rows,)):
            if not owned:
                vectors.append(({}, {}))
                continue
            v1, v2 = source.vector_job(p, s, owned, columns, offset == 0, offset + columns == p["S2"])
            if batch_mode:
                v1, v2 = ({key: value * heads for key, value in v.items()} for v in (v1, v2))
            vectors.append((v1, v2))
        identity = rows, columns, tuple(sharing), offset == 0, offset + columns == p["S2"]
        if identity not in self.cache:
            self.cache[identity] = {"cube": cubes, "vector": vectors, "rows": rows, "columns": columns,
                                    "fake": False, "heads": heads}
        return self.cache[identity]

    def offsets(self):
        s, p = self.s, self.p
        window = s["s2_base"] * s["n_ratio"] if s["same_ab"] or s["family"] in ("general", "varlen") else p["S2"]
        return range(0, p["S2"], window)


def batch_work(work):
    workers = []
    s = work.s
    for start in range(0, s["total_tasks"], s["split_factor"]):
        workers.append([work.job(batch, 0, 0, start) for batch in range(start, min(start + s["split_factor"], s["total_tasks"]))])
    return [workers[first:first + 2] for first in range(0, len(workers), 2)]


def s1_work(work):
    s, workers = work.s, []
    for start in range(0, s["total_tasks"], s["split_factor"]):
        clients = [[], []] if s["reuse"] else [[]]
        for task in range(start, min(start + s["split_factor"], s["total_tasks"])):
            client = (task - start) % len(clients)
            clients[client].append(work.job(task, 0, client, start))
        if len(clients) == 2 and len(clients[0]) != len(clients[1]):
            clients[1].append(_fake())
        workers.extend(clients)
    return [workers[first:first + 2] for first in range(0, len(workers), 2)]


def general_work(work):
    s, workers = work.s, []
    for start in range(0, s["total_tasks"], s["split_factor"]):
        end = min(start + s["split_factor"], s["total_tasks"])
        clients = [[], []] if s["reuse"] else [[]]
        for head in range(start // s["s1_outer"], (end - 1) // s["s1_outer"] + 1):
            first, last = max(start, head * s["s1_outer"]), min(end, (head + 1) * s["s1_outer"])
            for task in range(first, last):
                client = (task - first) % len(clients)
                clients[client].extend(work.job(task, offset, client, start) for offset in work.offsets())
            if s["reuse"] and (last - first) % 2:
                clients[1].extend(_fake() for _ in work.offsets())
        workers.extend(clients)
    return [workers[first:first + 2] for first in range(0, len(workers), 2)]


def varlen_work(work):
    workers, s = [], work.s
    for start in range(0, s["total_tasks"], s["split_factor"]):
        tasks = []
        for task in range(start, min(start + s["split_factor"], s["total_tasks"])):
            tasks.extend(work.job(task, offset, 0, start) for offset in work.offsets())
        workers.append(tasks)
    width = 1 if s["same_ab"] else 2
    return [workers[first:first + width] for first in range(0, len(workers), width)]


def workload(inputs, hardware):
    work = Work(inputs, hardware)
    name = implementation(work.p, work.s)
    work.pairs = (batch_work(work) if name == "b" else s1_work(work) if name == "s1"
                  else general_work(work) if name == "general" else varlen_work(work))
    if name == "same_ab":
        original = source.workload(work.p, hardware)
        work.pairs = [[tasks] for tasks in original["workers"]]
    return work


def active_features(work):
    totals = {pipe: Counter() for pipe in source.PIPES}
    same = work.s["same_ab"]
    for pair in work.pairs:
        for tasks in pair:
            for task in tasks:
                for job in task["cube"]:
                    for pipe in ("cube", "mte1", "mte2", "fixpipe", "cscalar"):
                        totals[pipe].update({name: job.get(name, 0) for name in source.PIPES[pipe][1]})
                for owner in task["vector"]:
                    for job in owner:
                        for pipe in ("vector", "vmte2", "vmte3", "vscalar"):
                            totals[pipe].update({name: job.get(name, 0) for name in source.PIPES[pipe][1]})
    aiv = sum(2 if same else len(pair) for pair in work.pairs)
    aic = len(work.pairs)
    if work.s["drop_adapter"]:
        aiv, aic = work.hardware["aiv_cores"], work.hardware["aic_cores"]
    totals["cscalar"]["workers"] = aic
    totals["vscalar"]["workers"] = aiv
    for pipe, workers in (("cscalar", aic), ("vscalar", aiv)):
        totals[pipe]["batch_entries"] = work.p["B"] * workers if work.p["layout"] == "TND" else 0
    for job in source.adapter_jobs(work.p, work.s, work.hardware):
        for pipe in ("vector", "vmte2", "vmte3", "vscalar"):
            totals[pipe].update({name: job.get(name, 0) for name in source.PIPES[pipe][1]})
    totals["vscalar"]["adapter_workers"] = aiv if work.s["drop_adapter"] else 0
    return {pipe: dict(values) for pipe, values in totals.items()}


def _stages(count):
    program = []
    for step in range(count + 2):
        if 0 <= step - 1 < count:
            program.append(("wait", step - 1, 0))
        if step < count:
            program.append(("launch", step, 0))
        if 0 <= step - 1 < count:
            program.append(("vector", step - 1, 1))
        if 0 <= step - 2 < count:
            program.append(("wait", step - 2, 2))
        if 0 <= step - 1 < count:
            program.append(("launch", step - 1, 2))
        if 0 <= step - 2 < count:
            program.append(("vector", step - 2, 3))
    return program


def batch_program(tasks):
    return _stages(len(tasks))


def s1_program(tasks):
    return _stages(len(tasks))


def general_program(tasks):
    return _stages(len(tasks))


def varlen_gm_program(tasks):
    return _stages(len(tasks))


def varlen_l1_program(tasks, peer):
    program = []
    for step in range(len(tasks) + 2):
        if 0 <= step - 1 < len(tasks):
            program.extend((("wait", step - 1, 0), ("vector", step - 1, 1)))
        if 0 <= step - 1 < len(peer):
            program.append(("peer_wait", step - 1, 0))
        if 0 <= step - 2 < len(tasks):
            program.extend((("wait", step - 2, 2), ("vector", step - 2, 3)))
        if 0 <= step - 2 < len(peer):
            program.append(("peer_wait", step - 2, 2))
    return program


class PairExecutor:
    def __init__(self, work, workers, costs, start, same=False):
        self.work, self.costs, self.same = work, costs, same
        self.workers = [workers[0], workers[0]] if same else workers
        self.name = implementation(work.p, work.s)
        self.controlled = same or self.name == "varlen_l1"
        self.programs = []
        for owner, tasks in enumerate(self.workers):
            if self.controlled:
                self.programs.append(varlen_l1_program(tasks, self.workers[1 - owner] if len(self.workers) == 2 else []))
            else:
                builder = {"b": batch_program, "s1": s1_program, "general": general_program, "varlen_gm": varlen_gm_program}[self.name]
                self.programs.append(builder(tasks))
        self.position = [0] * len(self.workers)
        self.blocked, self.ready, self.finished = {}, set(), {}
        self.vector_ready = set()
        self.events, self.queues, self.busy, self.flows, self.remaining = [], defaultdict(deque), set(), {}, {}
        self.call_waiting, self.call_remaining = {}, Counter()
        self.serial, self.flow_id = 0, 0
        self.l0c_used = 0
        self.l0ab_used = 0
        self.live_c, self.live_ab = Counter(), Counter()
        self.live_l1 = Counter()
        self.l1_bytes = 0
        self.c_bytes, self.a_bytes, self.b_bytes = 0, 0, 0
        self.l0c_limit = int(work.hardware.get("l0c_bytes", 131072))
        self.ub_bytes, self.ub_live = Counter(), Counter()
        self.max_resource_bytes = Counter()
        self.pipeline = []
        self.pipeline_position = 0
        self.phase_active = Counter()
        self.event_counts = Counter()
        self.service_cache = {}
        if self.controlled:
            owners = (0,) if same else range(len(self.workers))
            for step in range(max(map(len, self.workers)) + 1):
                self.pipeline.extend((owner, step, 0) for owner in owners if step < len(self.workers[owner]))
                self.pipeline.extend((owner, step - 1, 2) for owner in owners if 0 <= step - 1 < len(self.workers[owner]))
        self.start = start

    def service(self, job, pipe):
        key = id(job), pipe
        if key not in self.service_cache:
            self.service_cache[key] = sum(job.get(name, 0) * value for name, value in self.costs[pipe]["coefficients"].items())
        return self.service_cache[key]

    def key(self, owner, task, stage):
        return (0 if self.same else owner), task, stage

    def pump(self, resource, now):
        if resource in self.busy or not self.queues[resource]:
            return
        pipe = resource[0]
        flow = self.queues[resource][0]
        item = self.flows[flow]
        group = item["group"]
        output = item["output"]
        if pipe == "cube" and not output.get("c_allocated") and (self.c_bytes + item["c_bytes"] > self.l0c_limit
                               or self.live_c[group] >= item["db_l0c"]):
            return
        if pipe == "mte1" and (self.a_bytes + item["a_bytes"] > 65536
                               or self.b_bytes + item["b_bytes"] > 65536
                               or self.live_ab[group] >= item["db_l0ab"]):
            return
        owner = resource[1]
        if pipe == "vmte2" and (self.ub_live[owner] >= 2
                                or self.ub_bytes[owner] + item["ub_bytes"] > self.work.hardware["ub_bytes"]):
            return
        if pipe == "mte2" and (self.l1_bytes + item["l1_bytes"] > self.work.hardware["l1_bytes"]
                               or self.live_l1[group] >= item["l1_slots"]):
            return
        self.queues[resource].popleft()
        if pipe == "cube" and not output.get("c_allocated"):
            self.l0c_used += 1
            output["c_allocated"] = True
            self.live_c[group] += 1
            self.c_bytes += item["c_bytes"]
            self.max_resource_bytes["l0c"] = max(self.max_resource_bytes["l0c"], self.c_bytes)
        if pipe == "mte1":
            self.l0ab_used += 1
            self.live_ab[group] += 1
            self.a_bytes += item["a_bytes"]
            self.b_bytes += item["b_bytes"]
            self.max_resource_bytes["l0a"] = max(self.max_resource_bytes["l0a"], self.a_bytes)
            self.max_resource_bytes["l0b"] = max(self.max_resource_bytes["l0b"], self.b_bytes)
        if pipe == "vmte2":
            self.ub_live[owner] += 1
            self.ub_bytes[owner] += item["ub_bytes"]
            self.max_resource_bytes["ub"] = max(self.max_resource_bytes["ub"], self.ub_bytes[owner])
        if pipe == "mte2":
            self.l1_bytes += item["l1_bytes"]
            self.live_l1[group] += 1
            self.max_resource_bytes["l1"] = max(self.max_resource_bytes["l1"], self.l1_bytes)
        self.busy.add(resource)
        self.serial += 1
        duration = item["steps"][item["position"]][1]
        heapq.heappush(self.events, (now + duration, self.serial, flow))

    def enqueue(self, flow, now):
        item = self.flows[flow]
        resource = item["steps"][item["position"]][0]
        self.queues[resource].append(flow)
        self.pump(resource, now)

    def launch(self, owner, task, stage, now):
        cube = stage in (0, 2)
        unit = self.workers[owner][task]
        job = unit["cube"][stage // 2] if cube else unit["vector"][owner if self.same else 0][stage // 2]
        count = max(1, int(job.get("stores" if cube else "chunks", 0)))
        pipes = ("cscalar", "mte2", "mte1", "cube", "fixpipe") if cube else ("vscalar", "vmte2", "vector", "vmte3")
        group = owner, task, stage
        self.remaining[group] = count
        phase = ("bmm1", "vec1", "bmm2", "vec2")[stage]
        durations = {}
        for pipe in pipes:
            duration = self.service(job, pipe) / count
            self.phase_active[phase, pipe] += duration * count
            durations[pipe] = duration
        if unit.get("fake"):
            count = self.remaining[group] = 1
            if cube:
                self.event_counts["fake_matmul_messages"] += 1
        self.event_counts["matmul_calls" if cube else "vector_chunks"] += int(job.get("calls", 1)) if cube else count
        held = defaultdict(list)
        mm = self.work.mm[stage // 2]
        a_bytes, b_bytes = mm["base_m"] * mm["base_k"] * 2, mm["base_n"] * mm["base_k"] * 2
        manual_l1 = self.work.p["layout"] == "TND" and self.work.s["l1"]
        l1_slots = 2 if manual_l1 else max(1, min(mm["depth_a1"], mm["depth_b1"]))
        if self.name == "b":
            l1_slots = max(1, self.work.hardware["l1_bytes"] // (a_bytes + b_bytes))
        common = {"group": group, "c_bytes": mm["base_m"] * mm["base_n"] * 4,
            "a_bytes": a_bytes, "b_bytes": b_bytes, "l1_bytes": a_bytes + b_bytes, "l1_slots": l1_slots,
            "db_l0c": max(1, mm["db_l0c"]), "db_l0ab": max(1, min(mm["db_l0a"], mm["db_l0b"])),
            "ub_bytes": max(32, math.ceil((job.get("read_kb", 0) + job.get("write_kb", 0)) * 1024 / count))}
        for output_index in range(count):
            api_call = output_index // max(1, int(job.get("output_group_tiles", count)))
            output = {"c_allocated": False}
            template = {**common, "output": output, "position": 0, "api_call": api_call}
            if cube and not unit.get("fake"):
                fractions = job.get("k_slices") or job.get("pipeline_k_slices") or [1.0]
                output["remaining_parts"] = len(fractions)
                self.flow_id += 1
                output["fix_flow"] = self.flow_id
                self.flows[self.flow_id] = {**template, "steps": [(("fixpipe", 0), durations["fixpipe"])]}
                pending = []
                for fraction in fractions:
                    self.flow_id += 1
                    steps = [((pipe, 0), durations[pipe] * fraction) for pipe in ("cscalar", "mte2", "mte1", "cube")]
                    self.flows[self.flow_id] = {**template, "steps": steps, "cube_part": True}
                    pending.append(self.flow_id)
                self.event_counts["k_tiles"] += len(fractions)
            else:
                self.flow_id += 1
                steps = [((pipe, owner), durations[pipe]) for pipe in pipes]
                if unit.get("fake"):
                    steps = [(("cscalar", 0), durations["cscalar"])] if cube else [(("vscalar", owner), 0.0)]
                self.flows[self.flow_id] = {**template, "steps": steps}
                pending = [self.flow_id]
            self.call_remaining[group, api_call] += 1
            if api_call:
                held[api_call].extend(pending)
            else:
                for flow in pending:
                    self.enqueue(flow, now)
        self.call_waiting[group] = held

    def pump_controller(self, now):
        while self.pipeline_position < len(self.pipeline):
            owner, task, stage = self.pipeline[self.pipeline_position]
            required = range(2) if self.same else (owner,)
            if stage == 2 and any((peer, task) not in self.vector_ready for peer in required):
                return
            self.launch(owner, task, stage, now)
            self.pipeline_position += 1

    def advance(self, owner, now):
        program = self.programs[owner]
        while self.position[owner] < len(program):
            kind, task, stage = program[self.position[owner]]
            key = self.key(1 - owner if kind == "peer_wait" else owner, task, stage)
            if kind in ("wait", "peer_wait") and key not in self.ready:
                self.blocked[owner] = key
                return
            self.position[owner] += 1
            if kind == "launch":
                self.launch(owner, task, stage, now)
            elif kind == "vector":
                self.launch(owner, task, stage, now)
                return
        self.finished[owner] = now

    def run(self):
        if self.controlled:
            self.pump_controller(self.start)
        for owner in range(len(self.workers)):
            self.advance(owner, self.start)
        while self.events:
            now, _, flow = heapq.heappop(self.events)
            item = self.flows[flow]
            resource = item["steps"][item["position"]][0]
            self.busy.remove(resource)
            if resource[0] == "fixpipe":
                self.l0c_used -= 1
                self.live_c[item["group"]] -= 1
                self.c_bytes -= item["c_bytes"]
            if resource[0] == "cube":
                self.l0ab_used -= 1
                self.live_ab[item["group"]] -= 1
                self.a_bytes -= item["a_bytes"]
                self.b_bytes -= item["b_bytes"]
            if resource[0] == "mte1":
                self.l1_bytes -= item["l1_bytes"]
                self.live_l1[item["group"]] -= 1
            if resource[0] == "vmte3":
                self.ub_live[resource[1]] -= 1
                self.ub_bytes[resource[1]] -= item["ub_bytes"]
            self.pump(resource, now)
            self.pump(("cube", 0), now)
            self.pump(("mte1", 0), now)
            self.pump(("mte2", 0), now)
            if resource[0] == "vmte3":
                self.pump(("vmte2", resource[1]), now)
            item["position"] += 1
            if item["position"] < len(item["steps"]):
                self.enqueue(flow, now)
                continue
            group = item["group"]
            self.flows.pop(flow)
            if item.get("cube_part"):
                item["output"]["remaining_parts"] -= 1
                if not item["output"]["remaining_parts"]:
                    self.enqueue(item["output"]["fix_flow"], now)
                continue
            api_call = item["api_call"]
            self.call_remaining[group, api_call] -= 1
            if not self.call_remaining[group, api_call]:
                for waiting in self.call_waiting[group].pop(api_call + 1, []):
                    self.enqueue(waiting, now)
            self.remaining[group] -= 1
            if self.remaining[group]:
                continue
            self.remaining.pop(group)
            owner, task, stage = group
            if stage in (0, 2):
                key = self.key(owner, task, stage)
                self.ready.add(key)
                for peer in list(self.blocked):
                    if self.blocked.get(peer) == key:
                        self.blocked.pop(peer)
                        self.advance(peer, now)
            else:
                if stage == 1:
                    self.vector_ready.add((owner, task))
                    if self.controlled:
                        self.pump_controller(now)
                self.advance(owner, now)
        if len(self.finished) != len(self.workers) or self.flows or self.blocked or self.l0c_used or self.l0ab_used or self.l1_bytes:
            raise RuntimeError(f"undrained {self.name} program: workers={len(self.finished)}/{len(self.workers)} blocked={self.blocked}")
        return {"predicted_us": max(self.finished.values()), "phase_pipe_active_us": {f"{a}.{b}": v for (a, b), v in self.phase_active.items()},
                "execution_events": dict(self.event_counts), "l0c_capacity_bytes": self.l0c_limit,
                "max_resource_bytes": dict(self.max_resource_bytes)}


def adapter_time(work, costs):
    completions = []
    for job in source.adapter_jobs(work.p, work.s, work.hardware):
        durations = {pipe: sum(job.get(name, 0) * rate for name, rate in costs[pipe]["coefficients"].items())
                     for pipe in ("vscalar", "vmte2", "vector", "vmte3")}
        completions.append(sum(durations.values()))
    return max(completions, default=0.0) + (costs["vscalar"]["coefficients"].get("adapter_workers", 0) if completions else 0)


def predict_work(work, model):
    key = cost_class(work.p, work.s)
    if key not in model["classes"]:
        raise ValueError(f"no measured primitive costs for {key}")
    costs = model["classes"][key]["pipes"]
    for pipe, values in active_features(work).items():
        missing = [name for name, value in values.items() if value > 0 and name not in costs[pipe]["observed_features"]]
        if missing:
            raise ValueError(f"unmeasured primitive costs: {key}/{pipe}/" + ",".join(missing))
    prefix = adapter_time(work, costs)
    startup = max(costs["cscalar"]["coefficients"].get("workers", 0), costs["vscalar"]["coefficients"].get("workers", 0))
    if work.p["layout"] == "TND":
        startup += work.p["B"] * max(costs["cscalar"]["coefficients"].get("batch_entries", 0),
                                     costs["vscalar"]["coefficients"].get("batch_entries", 0))
    cache, results = {}, []
    for pair in work.pairs:
        identity = tuple(tuple(id(job) for job in tasks) for tasks in pair)
        if identity not in cache:
            cache[identity] = PairExecutor(work, pair, costs, prefix + startup, work.s["same_ab"]).run()
        results.append(cache[identity])
    result = max(results, key=lambda value: value["predicted_us"])
    return {**result, "adapter_us": prefix, "startup_us": startup,
            "implementation": implementation(work.p, work.s), "cost_class": key,
            "matmul": work.mm, "schedule": work.s,
            "unidentified_pipe_terms": [pipe for pipe, data in costs.items() if data["rank"] < data["terms"]],
            "target": "kernel task duration; PMU active sums are not standalone stage wall times"}


def predict(inputs, model):
    if model.get("version") != VERSION or model.get("source_sha256") != signature():
        raise ValueError("branch model and execution code differ")
    return predict_work(workload(inputs, model["hardware"]), model)


def fit_costs(samples, hardware):
    try:
        from scipy.optimize import nnls
    except ImportError:
        def nnls(matrix, targets, maxiter=10000):
            solution = source.legacy.pipeline_nnls(matrix.tolist(), targets.tolist())["weights"]
            values = np.asarray(solution, dtype=float)
            return values, float(np.linalg.norm(matrix @ values - targets))
    model = {"version": VERSION, "source_sha256": signature(), "hardware": hardware, "classes": {},
             "latency_used_for_cost_fit": False, "calibration_target": "per-pipe PMU active sums only",
             "coefficient_interpretation": "effective primitive costs; aggregate counters do not uniquely identify every instruction cost"}
    by_class = defaultdict(list)
    for item in samples:
        by_class[item["cost_class"]].append(item)
    for key, rows in by_class.items():
        entry = {"pipes": {}, "calibration_inputs": [row["input_sha256"] for row in rows]}
        for pipe, (_, names) in source.PIPES.items():
            good = [row for row in rows if row["targets"].get(pipe, 0) > 0]
            matrix = np.asarray([[row["features"][pipe].get(name, 0) for name in names] for row in good], dtype=float).reshape((-1, len(names)))
            targets = np.asarray([row["targets"][pipe] for row in good], dtype=float)
            active = np.any(matrix != 0, axis=0)
            rates = np.zeros(len(names))
            if np.any(active):
                normalized = matrix[:, active] / targets[:, None]
                scales = np.maximum(np.max(np.abs(normalized), axis=0), 1e-30)
                solution, _ = nnls(normalized / scales, np.ones(len(good)), maxiter=10000)
                rates[active] = solution / scales
            entry["pipes"][pipe] = {"coefficients": dict(zip(names, map(float, rates))),
                                     "rank": int(np.linalg.matrix_rank(matrix[:, active])) if np.any(active) else 0,
                                     "terms": int(sum(active)), "samples": len(good),
                                     "observed_features": [name for name, seen in zip(names, active) if seen]}
        model["classes"][key] = entry
    return model


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def records(paths):
    rows, hardware = {}, None
    names = ("soc", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes", "ub_bytes")
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") != "ok" or (_number(row.get("checked_elements")) or 0) <= 0:
                    continue
                if (_number(row.get("duration_us")) or 0) <= 0:
                    continue
                if any(_number(row.get(target)) is None for target, _ in source.PIPES.values()):
                    continue
                observed = {name: row[name] if name == "soc" else int(row[name]) for name in names}
                if hardware is not None and observed != hardware:
                    raise ValueError("branch data mix different hardware")
                hardware = observed
                p = source.cost.normalize({name: row[name] for name in source.cost.INPUT_FIELDS})
                identity = source.cost.sha256(p)
                rows[identity] = {**row, "inputs": p, "input_sha256": identity,
                                  "sequence_group": source.cost.sha256({name: p[name] for name in ("S1", "S2")})}
    if hardware is None:
        raise ValueError("no valid independent numerical checks and complete pipe PMU measurements")
    return list(rows.values()), hardware


def _summary(rows):
    good = [r for r in rows if r.get("branch_ape_percent") is not None]
    errors = [r["branch_ape_percent"] for r in good]
    return {"cases": len(rows), "predicted": len(good), "unavailable": len(rows) - len(good),
            "median_ape_percent": statistics.median(errors) if errors else None,
            "max_ape_percent": max(errors, default=None),
            "above": {str(limit): {"count": sum(value > limit for value in errors),
                                   "percent": 100 * sum(value > limit for value in errors) / len(errors) if errors else None}
                      for limit in (10, 20, 30)}}


def build_and_evaluate(paths, evaluation_path, output_dir, model_path=MODEL):
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    historical, hardware = records(paths)
    evaluation, observed = records([evaluation_path])
    if hardware != observed:
        raise ValueError("calibration and evaluation hardware differ")
    raw = {row["input_sha256"]: row for row in historical + evaluation}
    prepared, rejected = [], []
    feature_identity = source.cost.sha256({"rows": list(raw.values()), "primitives": source.source_hash(),
        "api": tiler()[1], "functions": [inspect.getsource(item) for item in
            (Work, implementation, cost_class, workload, batch_work, s1_work, general_work, varlen_work, active_features)]})
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "feature_cache.json"
    cached = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    if cached.get("identity") == feature_identity:
        prepared, rejected = cached["samples"], cached["rejected"]
        print(f"BRANCH FEATURE CACHE HIT samples={len(prepared)}", flush=True)
    for index, row in enumerate(raw.values(), 1):
        if cached.get("identity") == feature_identity:
            break
        if row.get("collection_group") == "same_ab":
            continue
        try:
            work = workload(row["inputs"], hardware)
            if int(row["priority"]) != work.s["priority"]:
                raise ValueError("input-derived family disagrees with the recorded Host priority")
            for stage, mm in enumerate(work.mm, 1):
                for key in ("base_m", "base_n", "base_k"):
                    value = row.get(f"bmm{stage}_{key}")
                    if _number(value) is not None and int(value) != mm[key]:
                        raise ValueError(f"input-derived bmm{stage} {key} differs from Host tiling")
            prepared.append({"input_sha256": row["input_sha256"], "sequence_group": row["sequence_group"],
                "cost_class": cost_class(work.p, work.s), "features": active_features(work),
                "targets": {pipe: float(row[target]) for pipe, (target, _) in source.PIPES.items()}})
            del work
        except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
            rejected.append({"input_sha256": row["input_sha256"], "error": str(error)})
        if index % 50 == 0:
            print(f"BRANCH FEATURES {index}/{len(raw)} prepared={len(prepared)}", flush=True)
    source.cost.save_json(cache_path, {"identity": feature_identity, "samples": prepared, "rejected": rejected})
    folds = 4
    models = {}
    for fold in range(folds):
        training = [row for row in prepared if int(row["sequence_group"][:8], 16) % folds != fold]
        models[fold] = fit_costs(training, hardware)
        models[fold]["heldout_sequence_groups"] = sorted({row["sequence_group"] for row in prepared
                                                         if int(row["sequence_group"][:8], 16) % folds == fold})
    results = []
    for index, row in enumerate(evaluation, 1):
        result = {**row, "branch_status": "unavailable", "branch_evaluation": "sequence_group_heldout_pipe_cost_calibration",
                  "branch_predicted_us": None, "branch_ape_percent": None,
                  "branch_error": "", "total_latency_used_for_fit": False}
        p = row["inputs"]
        try:
            if row.get("collection_group") == "same_ab" and row.get("prediction_before_measurement", "").lower() == "true":
                result.update(branch_predicted_us=float(row["predicted_us"]), branch_evaluation="original_frozen_sameab_prediction",
                              branch_model_sha256=row["model_sha256"])
            else:
                fold = int(row["sequence_group"][:8], 16) % folds
                model = models[fold]
                assert row["sequence_group"] not in {r["sequence_group"] for r in prepared
                                                     if int(r["sequence_group"][:8], 16) % folds != fold}
                work = workload(p, hardware)
                prediction = predict_work(work, model)
                result.update(branch_predicted_us=prediction["predicted_us"], branch_implementation=prediction["implementation"],
                              branch_cost_class=prediction["cost_class"], branch_fold=fold,
                              branch_model_sha256=source.cost.sha256(model),
                              branch_unidentified_pipe_terms="|".join(prediction["unidentified_pipe_terms"]),
                              branch_adapter_us=prediction["adapter_us"], branch_startup_us=prediction["startup_us"],
                              branch_execution_events=json.dumps(prediction["execution_events"], sort_keys=True),
                              branch_resource_bytes=json.dumps(prediction["max_resource_bytes"], sort_keys=True),
                              branch_predicted_phase_pipe_active_us=json.dumps(prediction["phase_pipe_active_us"], sort_keys=True))
                measured = []
                for stage, mm in enumerate(work.mm, 1):
                    for key, value in mm.items():
                        name = f"bmm{stage}_{key}"
                        if name in row and _number(row[name]) is not None and int(row[name]) != value:
                            measured.append(f"{name}:{value}!={row[name]}")
                result["branch_matmul_geometry_differences"] = "|".join(measured)
                for pipe, values in active_features(work).items():
                    data = model["classes"][prediction["cost_class"]]["pipes"][pipe]
                    estimated = sum(value * data["coefficients"].get(name, 0) for name, value in values.items())
                    actual = float(row[source.PIPES[pipe][0]])
                    result[f"branch_{pipe}_active_sum_us"] = estimated
                    result[f"branch_{pipe}_ape_percent"] = 100 * abs(estimated / actual - 1) if actual > 0 else None
                del work
            predicted, actual = result["branch_predicted_us"], float(row["duration_us"])
            if not math.isfinite(predicted) or predicted <= 0:
                raise ValueError("nonpositive branch prediction")
            result.update(branch_status="predicted", branch_ape_percent=100 * abs(predicted / actual - 1))
        except (ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
            result["branch_error"] = str(error)
        result.pop("inputs", None)
        results.append(result)
        if index % 25 == 0:
            print(f"BRANCH EVALUATION {index}/{len(evaluation)}", flush=True)
    final_model = fit_costs(prepared, hardware)
    final_model.update(training_inputs=sorted(row["input_sha256"] for row in prepared),
                       validation="four folds by complete (S1,S2) group; all layouts, dtypes, batches and heads of a sequence pair stay together",
                       cost_fit_excludes="all kernel total latency labels; rates are fitted to independent training PMU counters",
                       sdk_api_sha256=tiler()[1], source_files=FILES,
                       scope="dense FA; uniform-length TND; no sparse/PSE/Rope/backward; SDK geometry differences are reported")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in results for key in row))
    with (output_dir / "branch_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    report = {"groups": {}, "source_sha256": signature(), "feature_failures": rejected,
              "new_npu_measurements": 0, "kernel_latency_used_for_cost_fit": False,
              "evaluation": final_model["validation"], "hardware": hardware}
    for group in sorted({row.get("collection_group") or row["family"] for row in results}):
        selected = [row for row in results if (row.get("collection_group") or row["family"]) == group]
        report["groups"][group] = _summary(selected)
        report["groups"][group]["by_layout"] = {layout: _summary([row for row in selected if row["layout"] == layout])
                                                 for layout in sorted({row["layout"] for row in selected})}
        summary = report["groups"][group]
        print(f"BRANCH RESULT {group}: predicted={summary['predicted']}/{summary['cases']} "
              f"MdAPE={summary['median_ape_percent']}% " + " ".join(
                  f">{limit}%={item['count']}/{summary['predicted']}" for limit, item in summary["above"].items()), flush=True)
    final_model["qualification"] = "experimental source execution model; cross-validation has not met the accuracy target"
    final_model["cross_validation"] = report["groups"]
    source.cost.save_json(model_path, final_model)
    source.cost.save_json(output_dir / "branch_evaluation.json", report)
    print(f"BRANCH_MODEL={model_path}\nBRANCH_PREDICTIONS={output_dir / 'branch_predictions.csv'}", flush=True)
    return report


@lru_cache(maxsize=1)
def _forecast_state(directory):
    path = Path(directory) / "branch_predictions_before_measurement.json"
    return json.loads(path.read_text()) if path.is_file() else {"models": {}, "predictions": {}}


def _load_forecast_model():
    model = json.loads(MODEL.read_text())
    if model.get("source_sha256") != signature():
        raise ValueError("branch model and execution code differ")
    api = tiler()[1]
    if model.get("sdk_api_sha256") == api:
        return model
    cached = ROOT / "out/fa_branch_host/model.json"
    if cached.is_file():
        saved = json.loads(cached.read_text())
        if saved.get("source_sha256") == signature() and saved.get("sdk_api_sha256") == api:
            return saved
    print("BRANCH COSTS: CPU-only SDK adaptation from cached PMU; no NPU build or measurement", flush=True)
    rows, hardware = records([ROOT / "output/fa_input_latency_v1/diagnostics.csv",
                              ROOT / "output/fa_branch_collection/diagnostics.csv"])
    if hardware != model["hardware"]:
        raise ValueError("cached PMU hardware differs from the branch model")
    samples = []
    for row in rows:
        if row.get("collection_group") == "same_ab":
            continue
        try:
            work = workload(row["inputs"], hardware)
        except (ValueError, RuntimeError, subprocess.SubprocessError):
            continue
        samples.append({"input_sha256": row["input_sha256"], "sequence_group": row["sequence_group"],
                        "cost_class": cost_class(work.p, work.s), "features": active_features(work),
                        "targets": {pipe: float(row[target]) for pipe, (target, _) in source.PIPES.items()}})
    converted = fit_costs(samples, hardware)
    converted.update(sdk_api_sha256=api, training_inputs=sorted(row["input_sha256"] for row in samples),
                     validation="SDK adaptation uses cached PMU only; prospective forecasts remain to be validated",
                     source_files=FILES, scope=model["scope"])
    source.cost.save_json(cached, converted)
    return converted


@lru_cache(maxsize=1)
def _forecast_model():
    try:
        return _load_forecast_model()
    except (ValueError, RuntimeError, KeyError, OSError, subprocess.SubprocessError) as error:
        return {"unavailable_reason": str(error)}


def freeze_case(case, hardware, directory):
    state = _forecast_state(str(directory))
    identity = source.cost.sha256(case["inputs"])
    saved = state["predictions"].get(case["name"])
    if saved is not None:
        if saved["input_sha256"] != identity:
            raise ValueError("saved branch forecast belongs to another input")
        return saved, state["models"].get(saved.get("model_sha256"))
    item = {"input_sha256": identity, "status": "unavailable"}
    model = None
    try:
        model = _forecast_model()
        if model.get("unavailable_reason"):
            raise ValueError(model["unavailable_reason"])
        for name, value in model["hardware"].items():
            if str(hardware.get(name)) != str(value):
                raise ValueError(f"branch model hardware differs: {name}")
        model_sha = source.cost.sha256(model)
        item["model_sha256"] = model_sha
        result = predict(case["inputs"], model)
        result["prediction_quality"] = "source_branch_execution_model"
        item.update(status="predicted", result=result)
        state["models"][model_sha] = model
    except (ValueError, RuntimeError, KeyError, OSError, subprocess.SubprocessError) as error:
        item["error"] = str(error)
    state["predictions"][case["name"]] = item
    source.cost.save_json(Path(directory) / "branch_predictions_before_measurement.json", state)
    return item, model
