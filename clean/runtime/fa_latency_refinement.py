from __future__ import annotations

from . import fa_branch_latency as base

source = base.source
BranchGraph = base.BranchGraph
implementation = base.implementation
cost_class = base.cost_class
varlen_work = base.varlen_work
active_features = base.active_features
parameter_names = base.parameter_names
compact_graph = base.compact_graph


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
            mt, nt = source.cost.ceil_div(m, bm), source.cost.ceil_div(count, bn)
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
        * source.cost.align(t["m"])
        * source.cost.align(t["n"])
        * source.cost.align(k)
        / 1e6
        for t in tiles
        for k in t["k"]
    )
    job["l0_kb"] = sum(
        2
        * (source.cost.align(t["m"]) + source.cost.align(t["n"]))
        * source.cost.align(k)
        / 1024
        for t in tiles
        for k in t["k"]
    )
    manual = p["layout"] == "TND" and s["l1"]
    job["calls"] = len({t["api"] for t in tiles}) if manual else 1
    api_steps = job["calls"]
    if manual and stage == 1:
        width = 192 if s["same_ab"] else 256
        api_steps *= source.cost.ceil_div(columns, width) + 1
    job["commands"] = (
        api_steps + job["tiles"] + sum(t["barrier"] * len(t["k"]) for t in tiles)
    )
    job["api_steps"] = api_steps
    job["m_barriers"] = sum(t["barrier"] * len(t["k"]) for t in tiles)
    return job


def execution_conditions(work):
    conditions = {
        "matmul_api": (
            "batch"
            if work.s["family"] in ("b", "b_drop")
            else (
                "manual_l1"
                if work.p["layout"] == "TND" and work.s["l1"]
                else "direct" if work.s["same_ab"] else "kfc"
            )
        ),
        "batch_broadcast": str(work.p["N1"] != work.p["N2"]),
        "aic_underfilled": str(len(work.pairs) < int(work.hardware["aic_cores"])),
        "query_footprint_fits_l2": str(
            2 * work.p["B"] * work.p["N1"] * work.p["S1"] * work.p["D"]
            <= work.hardware["l2_bytes"]
        ),
        "kv_footprint_fits_l2": str(
            2 * work.p["B"] * work.p["N2"] * work.p["S2"] * (work.p["D"] + work.p["DV"])
            <= work.hardware["l2_bytes"]
        ),
        "sequence_tail": str(bool(work.p["S1"] % work.s["s1_base"])),
        "multiple_s2_windows": str(len(work.offsets()) > 1),
        "single_task_per_aiv": str(
            max(len(tasks) for pair in work.pairs for tasks in pair) == 1
        ),
        "steady_pipeline_reached": str(
            any(len(tasks) > 2 for pair in work.pairs for tasks in pair)
        ),
    }
    for stage in (0, 1):
        jobs = [
            unit["cube"][stage] for unit in work.cache.values() if not unit.get("fake")
        ]
        conditions[f"bmm{stage+1}_pipe_m_barrier"] = str(
            any(job.get("m_barriers", 0) for job in jobs)
        )
        conditions[f"bmm{stage+1}_multiple_k"] = str(
            any(any(len(t["k"]) > 1 for t in job["micro_tiles"]) for job in jobs)
        )
        conditions[f"bmm{stage+1}_partial_tile"] = str(
            any(
                t["m"] != work.mm[stage]["base_m"]
                or t["n"] != work.mm[stage]["base_n"]
                or any(k != work.mm[stage]["base_k"] for k in t["k"])
                for job in jobs
                for t in job["micro_tiles"]
            )
        )
        mm = work.mm[stage]
        kt = max(len(t["k"]) for job in jobs for t in job["micro_tiles"])
        for operand, step in (("a", "step_m"), ("b", "step_n")):
            if work.p["layout"] == "TND" and work.s["l1"]:
                conditions[f"bmm{stage+1}_{operand}_l1_fully_cached"] = (
                    "manual_l1_input"
                )
                continue
            depth, span = mm["depth_" + operand + "1"], kt * max(1, mm[step])
            capacity = depth if depth >= span else max(0, depth - 2)
            if depth == 2 and kt == 1:
                capacity = 0
            conditions[f"bmm{stage+1}_{operand}_l1_fully_cached"] = str(
                capacity >= span
            )
    return conditions


class Work(base.Work):
    def refine_cube(self, stage, rows, columns, heads, job):
        return refine_matmul_job(
            self.p, self.s, self.mm[stage], stage, rows, columns, heads, job
        )


def workload(inputs, hardware):
    return base.workload(inputs, hardware, Work)


def graph_entry(model, key, conditions=None):
    if key in model["classes"]:
        entry = model["classes"][key]
        refinement = entry.get("source_refinement")
        if refinement:
            value = (conditions or {}).get(refinement["condition"])
            if value not in refinement["children"]:
                raise ValueError(
                    "source condition required: " + refinement["condition"]
                )
            return refinement["children"][value]
        return entry
    raise ValueError("unmeasured Varlen L1 cost class: " + key)


def predict_selected(inputs, model):
    p = source.cost.normalize(inputs)
    schedule = source.schedule(p, model["hardware"])
    if p["layout"] == "TND" and not schedule["same_ab"]:
        schedule["l1"] = schedule["s1_base"] <= 128 and max(p["D"], p["DV"]) <= 256
    impl = implementation(p, schedule)
    decision = model["selection"][impl]
    if decision:
        if impl != "varlen_l1" or not model.get("varlen_l1_transfer"):
            raise ValueError("the selected cost model requires Varlen L1 DMA costs")
        from .fa_latency_transfer import predict as predict_transfer

        result = predict_transfer(p, model["refined"], model["varlen_l1_transfer"])
    else:
        result = base.predict(p, model["baseline"])
    result["selected_cost_model"] = "micro_costs" if decision else "baseline"
    return result
