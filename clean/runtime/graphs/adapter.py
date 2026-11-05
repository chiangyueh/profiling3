from ..workloads.adapter import jobs


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
