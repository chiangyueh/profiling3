from . import fa_latency_inputs as cost


def finish(p, hardware, family, priority, base, ratio, reuse=False, l1=False, nz=False):
    b, n, q, k = (p[name] for name in ("B", "N1", "S1", "S2"))
    align, ceil = cost.align, cost.ceil_div
    ak, aiv = align(k), int(hardware["aiv_cores"])
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
            "B_BATCH" if s["family"] in ("b", "b_drop")
            else "S1" if s["family"] == "s1" else "GENERAL"
        )
        s["implementation"] += "_NZ" if s["nz"] else "_ND"
        s["implementation"] += "_L1" if s["l1"] else "_REUSE" if s["reuse"] else "_GM"
    return s


def _fake():
    return {
        "cube": [{"commands": 1}, {"commands": 1}],
        "vector": [({}, {})],
        "rows": 0,
        "columns": 0,
        "fake": True,
    }


def stages(count):
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


def program(tasks, peer, schedule):
    return stages(len(tasks))


def controlled_program(tasks, peer, schedule):
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


def sequence_work(work, width):
    workers, s = [], work.s
    for start in range(0, s["total_tasks"], s["split_factor"]):
        tasks = []
        for task in range(start, min(start + s["split_factor"], s["total_tasks"])):
            tasks.extend(work.job(task, offset, 0, start) for offset in work.offsets())
        workers.append(tasks)
    return [workers[first : first + width] for first in range(0, len(workers), width)]
