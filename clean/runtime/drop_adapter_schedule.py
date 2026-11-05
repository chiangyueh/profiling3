from . import fa_latency_inputs as cost


def enabled(p):
    return bool(p["has_drop_mask"] and (p["S2"] % 8 or p["layout"] == "TND" and p["B"] > 1))


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


def jobs(work):
    jobs = adapter_jobs(work.p, work.s, work.hardware)
    if not jobs:
        return []
    count = work.p["B"] * work.p["N1"] * work.p["S1"] * work.p["S2"]
    single = (
        cost.ceil_div(
            cost.ceil_div(count, 2048), int(work.hardware["aiv_cores"])
        )
        * 2048
    )
    base = min(
        single, cost.ceil_div(int(work.hardware["ub_bytes"]), 6) // 2048 * 2048
    )
    return [
        {**job, "cast_k": job["unpack_k"], "duplicate_k": base / 1000} for job in jobs
    ]


def build(graph, prefix):
    adapter_ends = []
    adapter_cache = {}
    for job in jobs(graph.work):
        identity = tuple(sorted(job.items()))
        if identity not in adapter_cache:
            graph.tails, graph.buffers = {}, {}
            chunks = max(1, int(job["chunks"]))
            graph.operation(
                "vector",
                0,
                {"duplicate_k": job["duplicate_k"]},
                "drop_adapter",
                (prefix,),
                writes=("adapterOnes",),
            )
            for _ in range(chunks):
                issue = graph.operation(
                    "vscalar", 0, job, "drop_adapter", (prefix,), divisor=chunks
                )
                read = graph.operation(
                    "vmte2",
                    0,
                    job,
                    "drop_adapter",
                    (issue,),
                    writes=("adapterInput",),
                    divisor=chunks,
                )
                selected = graph.operation(
                    "vector",
                    0,
                    {"unpack_k": job["unpack_k"], "chunks": job["chunks"]},
                    "drop_adapter",
                    (read,),
                    reads=("adapterInput",),
                    writes=("adapterSelect",),
                    divisor=chunks,
                )
                compute = graph.operation(
                    "vector",
                    0,
                    {"cast_k": job["cast_k"]},
                    "drop_adapter",
                    (selected,),
                    reads=("adapterSelect",),
                    writes=("adapterOutput",),
                    divisor=chunks,
                )
                end = graph.operation(
                    "vmte3",
                    0,
                    job,
                    "drop_adapter",
                    (compute,),
                    reads=("adapterOutput",),
                    divisor=chunks,
                )
            adapter_cache[identity] = end
        adapter_ends.append(adapter_cache[identity])
    if adapter_ends:
        prefix = graph.node(adapter_ends)
        prefix = graph.node(
            (prefix,),
            graph.template("vscalar", {"adapter_workers": 1}, "drop_adapter"),
        )
    return prefix
