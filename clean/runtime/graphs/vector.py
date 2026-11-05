from ..schedules import implementation


def build(graph, owner, unit, stage, launch):
    phase = "vec1" if stage == 1 else "vec2"
    job = unit["vector"][owner if graph.work.s["same_ab"] else 0][stage // 2]
    if unit.get("fake") or not job.get("chunks"):
        return launch, launch
    chunks = int(job["chunks"])
    name = implementation(graph.work.p, graph.work.s)
    b_or_s1 = name in ("b", "s1")
    if stage == 1:
        inp, calc, out = "stage1Pong", "stage1Ping", "vec1Output"
        pre_names, post_names = ("scale_k", "nz2nd_k"), ("cast_k", "nd2nz_k")
    elif b_or_s1:
        inp, calc, out = "stage1Ping", "stage1Pong", "commonOutput"
        pre_names, post_names = ("nz2nd_k",), ("cast_k", "nd2nz_k")
    else:
        inp, calc, out = "common", "stage2", "stage2"
        pre_names, post_names = ("nz2nd_k",), ("cast_k", "nd2nz_k")
    pre = {key: job.get(key, 0) for key in pre_names}
    main = {
        key: value
        for key, value in job.items()
        if key not in pre_names + post_names
    }
    post = {key: job.get(key, 0) for key in post_names}
    issued, finished = launch, launch
    for chunk in range(chunks):
        issued = graph.operation(
            "vscalar", owner, job, phase, (issued,), divisor=chunks
        )
        read_deps = [issued, launch]
        if stage == 3 and not b_or_s1:
            read_deps.append(graph.tails.get(("vmte3", owner), -1))
        loaded = graph.operation(
            "vmte2", owner, job, phase, read_deps, writes=(inp,), divisor=chunks
        )
        prefix = graph.operation(
            "vector",
            owner,
            pre,
            phase,
            (loaded,),
            reads=(inp,),
            writes=(calc,),
            divisor=chunks,
        )
        middle = graph.operation(
            "vector",
            owner,
            main,
            phase,
            (prefix,),
            reads=(calc,),
            writes=(calc, "common"),
            divisor=chunks,
        )
        cast = graph.operation(
            "vector",
            owner,
            post,
            phase,
            (middle,),
            reads=(calc,),
            writes=(out,),
            divisor=chunks,
        )
        finished = graph.operation(
            "vmte3", owner, job, phase, (cast,), reads=(out,), divisor=chunks
        )
    if stage == 3:
        graph.tails["vmte2", owner] = graph.node(
            (graph.tails.get(("vmte2", owner), -1), finished)
        )
    graph.events["vector_chunks"] += chunks if not unit.get("fake") else 0
    return issued, finished
