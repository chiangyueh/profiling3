from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np

import fa_latency_refinement as micro


BASE_FEATURES = ("payload_kb", "copy_calls", "strided_rows")
EXTRA_FEATURES = ("nd2nz_partial_rows", "nd2nz_split_sector_kb", "capacity_pressure_kb")
FEATURES = BASE_FEATURES + EXTRA_FEATURES
PARAMETERS = micro.parameter_names() + tuple(("mte2", name) for name in FEATURES)
VERSION = "varlen_l1_nd2nz_v2"


def signature():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def transfer(rows, width, stride, name, pressure=0):
    full, tail = divmod(width * 2, 256)
    partial = rows if name != "P" and tail else 0
    return {"payload_kb": rows * width * 2 / 1024, "copy_calls": 1,
            "strided_rows": rows if stride > width else 0, "source": name,
            "nd2nz_partial_rows": partial,
            "nd2nz_split_sector_kb": partial * full * (256 - tail) / 1024 if width * 2 % 128 else 0,
            "capacity_pressure_kb": rows * width * 2 / 1024 * pressure,
            "rows": rows, "width_elements": width, "stride_elements": stride}


class Work(micro.Work):
    def job(self, task, offset, client, group):
        unit = super().job(task, offset, client, group)
        if "dma_groups" in unit["cube"][0]:
            return unit
        p = self.p
        footprint = 2 * p["B"] * p["N2"] * p["S2"] * (p["D"] + p["DV"])
        pressure = max(0.0, 1.0 - self.hardware["l2_bytes"] / footprint)
        rows, columns = unit["rows"], unit["columns"]
        chunks = [min(256, columns - pos) for pos in range(0, columns, 256)]
        q = [] if offset else [transfer(rows, p["D"], p["N1"] * p["D"], "Q")]
        first = [[transfer(count, p["D"], p["N2"] * p["D"], "K", pressure)] for count in chunks]
        second = [[transfer(micro.source.cost.align(rows), micro.source.cost.align(count),
                            micro.source.cost.align(count), "P"),
                   transfer(count, p["DV"], p["N2"] * p["DV"], "V", pressure)] for count in chunks]
        unit["cube"][0].update(dma_query=q, dma_groups=first)
        unit["cube"][1].update(dma_query=[], dma_groups=second)
        for job in unit["cube"]:
            copies = job["dma_query"] + [item for group in job["dma_groups"] for item in group]
            job["dma_features"] = {name: sum(item[name] for item in copies) for name in FEATURES}
        return unit


def workload(inputs, hardware, calibration_geometry=None):
    work = Work(inputs, hardware, calibration_geometry)
    if micro.implementation(work.p, work.s) != "varlen_l1":
        raise ValueError("ND2NZ transfer model requires the Varlen L1 implementation")
    work.pairs = micro.varlen_work(work)
    return work


def features(work):
    result = Counter()
    for pair in work.pairs:
        for tasks in pair:
            for unit in tasks:
                for job in unit["cube"]:
                    result.update(job["dma_features"])
    return dict(result)


class Graph(micro.BranchGraph):
    def __init__(self, work):
        super().__init__(work)
        extra = len(PARAMETERS) - len(self.names)
        self.names = PARAMETERS
        self.columns = {name: index for index, name in enumerate(self.names)}
        self.templates = [tuple(row) + (0.0,) * extra for row in self.templates]
        self.template_ids = {(phase, row): index for index, (phase, row) in enumerate(zip(self.phases, self.templates))}
        self.query_data = {}
        self.kv_iteration = self.p_iteration = 0

    def template(self, pipe, values, phase, divisor=1):
        if pipe != "mte2" or len(self.names) != len(PARAMETERS):
            return super().template(pipe, values, phase, divisor)
        row = [0.0] * len(self.names)
        for name in FEATURES:
            row[self.columns[pipe, name]] = float(values.get(name, 0)) / divisor
        key = phase, tuple(row)
        if key not in self.template_ids:
            self.template_ids[key] = len(self.templates)
            self.templates.append(key[1])
            self.phases.append(phase)
        return self.template_ids[key]

    def load(self, item, phase, launch, slot, slots):
        self.events["dma_copy_calls"] += 1
        self.events["dma_payload_bytes"] += round(item["payload_kb"] * 1024)
        self.events["nd2nz_strided_rows"] += item["strided_rows"]
        return self.operation("mte2", 0, item, phase, (launch, slots.get(slot, -1)))

    def cube(self, owner, unit, stage, launch, slots):
        if unit.get("fake"):
            return super().cube(owner, unit, stage, launch, slots)
        phase = "bmm1" if stage == 0 else "bmm2"
        job, mm = unit["cube"][stage // 2], self.work.mm[stage // 2]
        tiles = job["micro_tiles"]
        ab_count = max(1, min(mm["db_l0a"], mm["db_l0b"]))
        c_count = max(1, min(mm["db_l0c"], self.work.hardware.get("l0c_bytes", 131072) // (mm["base_m"] * mm["base_n"] * 4)))
        query_slot = ("dma_q", owner)
        if job["dma_query"]:
            self.query_data[owner] = self.load(job["dma_query"][0], phase, launch, query_slot, slots)
        fixed, accumulators, serial = [], {}, 0
        for api, copies in enumerate(job["dma_groups"]):
            if stage == 0:
                serial = 0
            previous_fix = self.tails.get(("fixpipe", 0), -1)
            issued = self.operation("cscalar", 0, {"commands": 1}, phase, (launch,))
            kv_slot = ("dma_kv", self.kv_iteration % 2)
            self.kv_iteration += 1
            if stage == 0:
                a_ready = self.query_data[owner]
                b_ready = self.load(copies[0], phase, issued, kv_slot, slots)
                selected = [(index, tile) for index, tile in enumerate(tiles) if tile["api"] == api]
                p_slot = None
            else:
                p_slot = ("dma_p", self.p_iteration % 2)
                self.p_iteration += 1
                a_ready = self.load(copies[0], phase, issued, p_slot, slots)
                b_ready = self.load(copies[1], phase, issued, kv_slot, slots)
                selected = list(enumerate(tiles))
            released = []
            for output, tile in selected:
                c_slot = ("c0", output % c_count) if stage == 0 else ("c0", output)
                partial = accumulators.get(output, self.node((slots.get(c_slot, -1), previous_fix))) if stage else self.node((slots.get(c_slot, -1), previous_fix))
                lengths = tile["k"] if stage == 0 else [min(mm["base_k"], copies[1]["rows"] - pos)
                    for pos in range(0, copies[1]["rows"], mm["base_k"])]
                for index, length in enumerate(lengths):
                    l0_slot = ("ab0", serial % ab_count)
                    serial += 1
                    m, n, k = map(micro.source.cost.align, (tile["m"], tile["n"], length))
                    local = {"l0_kb": 2 * (m + n) * k / 1024, "tiles": 1}
                    moved = self.operation("mte1", 0, local, phase, (a_ready, b_ready, slots.get(l0_slot, -1)))
                    released.append(moved)
                    arithmetic = {"flops_m": 2 * m * n * k / 1e6, "tiles": 1,
                                  "output_tiles": int(index == 0 and (stage == 0 or api == 0))}
                    partial = self.operation("cube", 0, arithmetic, phase, (moved, partial))
                    slots[l0_slot] = partial
                    self.operation("cscalar", 0, {"commands": 1 + tile["barrier"]}, phase, (issued,))
                    self.events["k_tiles"] += 1
                    self.events["pipe_m_barriers"] += tile["barrier"]
                if stage == 0:
                    fixed.append(self.store(tile, job, phase, partial, int(output == selected[0][0])))
                    slots[c_slot] = fixed[-1]
                else:
                    accumulators[output] = partial
            end_read = self.node(released)
            if stage == 0:
                self.tails["mte1", 0] = self.node((self.tails.get(("mte1", 0), -1), partial))
                self.events["matmul_end_l0_fences"] += 1
            slots[kv_slot] = end_read
            if p_slot is not None:
                slots[p_slot] = end_read
            else:
                slots[query_slot] = end_read
            self.events["matmul_calls"] += 1
        if stage == 2:
            for output, tile in enumerate(tiles):
                fixed.append(self.store(tile, job, phase, accumulators[output], int(output == 0)))
                slots["c0", output] = fixed[-1]
            self.events["get_tensor_c_calls"] += 1
            self.tails["mte1", 0] = self.node((self.tails.get(("mte1", 0), -1), partial))
            self.events["matmul_end_l0_fences"] += 1
        return self.node(fixed)

    def store(self, tile, job, phase, dependency, first):
        write = tile["m"] * micro.source.cost.align(tile["n"]) * 4 / 1024
        values = {"nz_write_kb": write if job.get("nz_write_kb") else 0,
                  "nd_write_kb": 0 if job.get("nz_write_kb") else write, "stores": 1, "calls": first}
        return self.operation("fixpipe", 0, values, phase, (dependency,))


def branch_graph(work):
    components, seen = [], set()
    for pair in work.pairs:
        identity = tuple(tuple(id(unit) for unit in tasks) for tasks in pair)
        if identity in seen:
            continue
        seen.add(identity)
        component = copy.copy(work)
        component.pairs = [pair]
        components.append(micro.compact_graph(Graph(component).build()))
    if len(components) == 1:
        return components[0]
    a, b, labels, templates, phases, roots = [], [], [], [], [], []
    offset = template_offset = 0
    events = Counter()
    for graph in components:
        a.append(np.where(graph["a"] >= 0, graph["a"] + offset, -1).astype(np.int32))
        b.append(np.where(graph["b"] >= 0, graph["b"] + offset, -1).astype(np.int32))
        labels.append(graph["which"] + template_offset)
        templates.append(graph["templates"])
        phases.extend(graph["phases"])
        roots.append(graph["root"] + offset)
        offset += len(graph["a"])
        template_offset += len(graph["templates"])
        events.update(graph["execution_events"])
    zero = template_offset
    templates.append(np.zeros((1, len(PARAMETERS))))
    phases.append("startup")
    left, right, which = list(np.concatenate(a)), list(np.concatenate(b)), list(np.concatenate(labels))
    while len(roots) > 1:
        first, second = roots.pop(), roots.pop()
        roots.append(len(left))
        left.append(first); right.append(second); which.append(zero)
    return {"a": np.ascontiguousarray(left, dtype=np.int32), "b": np.ascontiguousarray(right, dtype=np.int32),
            "which": np.ascontiguousarray(which, dtype=np.int32), "templates": np.ascontiguousarray(np.vstack(templates)),
            "root": roots[0], "phases": phases, "execution_events": dict(events)}


def fit(samples, hardware):
    from scipy.optimize import linprog
    selected = [row for row in samples if row["cost_class"].startswith("varlen_l1|") and row["targets"].get("mte2", 0) > 0]
    matrix, targets = [], []
    for row in selected:
        observed = row.get("dma_features") or features(workload(row["inputs"], hardware))
        matrix.append([observed[name] for name in FEATURES])
        targets.append(row["targets"]["mte2"])
    x, y = np.array(matrix, dtype=float), np.array(targets, dtype=float)
    scale = np.maximum(np.max(x / y[:, None], axis=0), 1e-30)
    normalized = x / y[:, None] / scale
    n = len(y)
    result = linprog(np.r_[np.zeros(len(FEATURES)), np.ones(n) / n],
        A_ub=np.vstack((np.c_[normalized, -np.eye(n)], np.c_[-normalized, -np.eye(n)])),
        b_ub=np.r_[np.ones(n), -np.ones(n)], bounds=(0, None), method="highs", options={"threads": 1})
    if not result.success:
        raise RuntimeError("source DMA-cost calibration failed: " + result.message)
    coefficients = result.x[:len(FEATURES)] / scale
    return {"version": VERSION, "source_sha256": signature(), "features": list(FEATURES),
            "coefficients": dict(zip(FEATURES, map(float, coefficients))), "feature_rank": int(np.linalg.matrix_rank(normalized)),
            "training_inputs": sorted(row["input_sha256"] for row in selected), "calibration_samples": n,
            "calibration_target": "original measured whole-kernel AIC MTE2 active-sum microseconds; no kernel latency target",
            "new_validation_latency_used_for_fit": False, "shared_fp16_bf16_dma_cost": True,
            "source": "flash_attention_var_len_score.h CopyInMm1AToL1/CopyInMm1BToL1/CopyInMm2AToL1/CopyInMm2BToL1/CopyGmToL1",
            "stride_semantics": "srcDValue selects the next row; skipped address holes are not payload",
            "reuse_semantics": "Q once per query task; K/V 256-row ping-pong buffers; padded contiguous P; no assumed cross-core L1 or L2 hit",
            "cache_miss_cost_independently_calibrated": False}


def fit_extensions(samples, hardware, previous):
    from scipy.optimize import linprog
    selected = [row for row in samples if row["cost_class"].startswith("varlen_l1|") and row["targets"].get("mte2", 0) > 0]
    matrix, targets, baseline = [], [], []
    for row in selected:
        values = row.get("dma_features") or features(workload(row["inputs"], hardware))
        matrix.append([values[name] for name in EXTRA_FEATURES])
        targets.append(row["targets"]["mte2"])
        baseline.append(sum(values[name] * previous["coefficients"][name] for name in BASE_FEATURES))
    x, y, old = np.array(matrix, dtype=float), np.array(targets, dtype=float), np.array(baseline, dtype=float)
    scale = np.maximum(np.max(x / y[:, None], axis=0), 1e-30)
    normalized = x / y[:, None] / scale
    rank = int(np.linalg.matrix_rank(normalized))
    if rank != len(EXTRA_FEATURES):
        raise ValueError("additional DMA costs are not identifiable from these measurements")
    n = len(y)
    target = (y - old) / y
    result = linprog(np.r_[np.zeros(len(EXTRA_FEATURES)), np.ones(n) / n],
        A_ub=np.vstack((np.c_[normalized, -np.eye(n)], np.c_[-normalized, -np.eye(n)])),
        b_ub=np.r_[target, -target], bounds=(0, None), method="highs", options={"threads": 1})
    if not result.success:
        raise RuntimeError("DMA extension calibration failed: " + result.message)
    rates = result.x[:len(EXTRA_FEATURES)] / scale
    coefficients = {name: previous["coefficients"][name] for name in BASE_FEATURES}
    coefficients.update(zip(EXTRA_FEATURES, map(float, rates)))
    return {**previous, "version": VERSION, "source_sha256": signature(), "features": list(FEATURES),
            "coefficients": coefficients, "extension_feature_rank": rank,
            "extension_training_inputs": sorted(row["input_sha256"] for row in selected),
            "extension_calibration_samples": n,
            "extension_calibration_target": "measured whole-kernel AIC MTE2 active-sum; repeated inputs grouped by median; original three DMA coefficients frozen; kernel latency never fitted",
            "extension_feature_support": {name: sorted({micro.source.cost.sha256({key: row['inputs'][key] for key in ('S1', 'S2')})
                for row, values in zip(selected, x) if values[index] > 0}) for index, name in enumerate(EXTRA_FEATURES)},
            "alignment_reference": "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/83RC1alpha002/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_10014.html",
            "alignment_semantics": "ND2NZ partial rows plus excess 256-byte sector capacity of rows not aligned to 128 bytes, weighted by complete sectors; geometric efficiency proxy, not additional payload; P is already NZ and excluded",
            "capacity_semantics": "K/V source payload times max(0, 1-L2_bytes/KV_footprint_bytes); capacity-pressure proxy, not an exact cache-hit simulation",
            "end_fence_source": "Matmul Scheduler::End -> TBufPoolL0::ResetCache waits M_MTE1 for both L0 slots; BMM1 End per N chunk, BMM2 End after GetTensorC",
            "cache_miss_cost_independently_calibrated": False,
            "stage_costs_independently_measured": False,
            "prospective_validation_required": True}


def predict(inputs, model, contract):
    if contract["source_sha256"] != signature():
        raise ValueError("DMA cost model and source differ")
    work = workload(inputs, model["hardware"])
    entry = micro.graph_entry(model, micro.cost_class(work.p, work.s), micro.execution_conditions(work))
    for pipe, values in micro.active_features(work).items():
        if pipe == "mte2":
            continue
        missing = [name for name, value in values.items() if value > 0 and name not in entry["pipes"][pipe]["observed_features"]]
        if missing:
            raise ValueError("unmeasured source primitive: " + pipe + "/" + ",".join(missing))
    rates = list(entry["rates"]) + [contract["coefficients"][name] for name in FEATURES]
    graph = branch_graph(work)
    elapsed, gradient = micro.graph_time(graph, rates, True)
    critical = micro.source.critical_phases(graph, rates)
    pipes, terms = Counter(), {}
    for (pipe, name), value, rate in zip(PARAMETERS, gradient, rates):
        pipes[pipe] += float(value * rate)
        if pipe == "mte2" and name in FEATURES:
            terms[name] = float(value * rate)
    if not math.isclose(sum(pipes.values()), elapsed, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("DMA critical-cost attribution does not sum to predicted latency")
    total = features(work)
    return {"predicted_us": elapsed, "implementation": "varlen_l1", "cost_class": micro.cost_class(work.p, work.s),
            "schedule": work.s, "matmul": work.mm, "critical_path_phase_us": critical,
            "critical_path_pipe_us": dict(pipes), "critical_dma_terms_us": terms,
            "dma_features": total, "dma_active_sum_us": sum(total[name] * contract["coefficients"][name] for name in FEATURES),
            "execution_events": graph["execution_events"], "transport_contract": VERSION,
            "transfer_cost_support": {name: len(contract.get("extension_feature_support", {}).get(name, [])) for name in EXTRA_FEATURES if total[name] > 0},
            "execution_events_scope": "distinct representative worker timelines; dma_features contains whole-device source-copy totals",
            "stage_costs_independently_measured": False,
            "refinement": {"source_conditions": micro.execution_conditions(work),
                           "status": "capacity_pressure_proxy_requires_validation" if total["capacity_pressure_kb"] > 0 else "nd2nz_alignment_cost_requires_validation" if total["nd2nz_split_sector_kb"] > 0 else "source_dma_accounting",
                           "cache_hit_assumed": False},
            "calibration_sdk_api_sha256": model.get("sdk_api_sha256"), "prediction_sdk_api_sha256": micro.tiler()[1],
            "sdk_api_matches_calibration": model.get("sdk_api_sha256") == micro.tiler()[1],
            "accuracy_guaranteed": False}
