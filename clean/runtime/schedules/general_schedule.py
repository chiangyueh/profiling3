from .. import fa_latency_inputs as cost
from .schedule_common import _fake, finish, program


def schedule(p, hardware):
    b, n, nk, q, k, d = (p[name] for name in ("B", "N1", "N2", "S1", "S2", "D"))
    align, ceil = cost.align, cost.ceil_div
    aq, ak = align(q), align(k)
    aiv = int(hardware["aiv_cores"])
    base = min(128 if b * n * (n // nk) * ceil(q, min(64, aq)) > aiv else 64, aq)
    ratio = min(
        5 if 1025 <= k <= 1040 else (
            6 if d == 64 and p["layout"] == "BNSD" and 2049 <= k <= 2304 else 8
        ),
        ceil(k, min(128, ak)),
    )
    reuse = d <= 128 and not (
        d == 64 and p["layout"] == "BNSD" and k % 256 and k not in (4032, 42192)
    )
    nz = bool(k % 64 and d != 64)
    return finish(p, hardware, "general", 96, base, ratio, reuse=reuse, nz=nz)


def workload(work):
    s, workers = work.s, []
    for start in range(0, s["total_tasks"], s["split_factor"]):
        end = min(start + s["split_factor"], s["total_tasks"])
        clients = [[], []] if s["reuse"] else [[]]
        for head in range(start // s["s1_outer"], (end - 1) // s["s1_outer"] + 1):
            first, last = max(start, head * s["s1_outer"]), min(
                end, (head + 1) * s["s1_outer"]
            )
            for task in range(first, last):
                client = (task - first) % len(clients)
                clients[client].extend(
                    work.job(task, offset, client, start) for offset in work.offsets()
                )
            if s["reuse"] and (last - first) % 2:
                clients[1].extend(_fake() for _ in work.offsets())
        workers.extend(clients)
    return [workers[first : first + 2] for first in range(0, len(workers), 2)]
