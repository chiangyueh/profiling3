from . import fa_latency_inputs as cost
from .schedule_common import _fake, finish, program


def schedule(p, hardware):
    b, n, q, k, d = (p[name] for name in ("B", "N1", "S1", "S2", "D"))
    align, ceil = cost.align, cost.ceil_div
    aq, ak, ad = align(q), align(k), align(d)
    aiv = int(hardware["aiv_cores"])
    initial = min(128, aq)
    l1 = ad == 64 and 512 <= ak <= 1024 and q <= 3840
    reuse = not l1 and (
        (k >= 512 and b * n > 64 and d <= 128)
        or (b * n >= 4800 and k == 256 and d == 144)
    )
    nz = bool(k % 64 or k == 64 or k >= 704 and k % 128 and b * n * q > 20480)
    maximum = 1 if l1 and ak == 512 or ak < 128 else 4
    if nz and d <= 256:
        maximum = 4
    ratio, total = 1, b * n * ceil(q, initial)
    while (
        ratio < maximum
        and total > (ratio - 1) * aiv // ratio
        and initial * ratio < aq
        and initial * ratio * ak <= 131072
    ):
        ratio += 1
        total = b * n * ceil(ceil(q, initial), ratio)
    while (
        total <= (ratio - 1) * aiv // ratio
        or initial * ratio > aq
        or initial * ratio * ak > 131072
    ):
        ratio -= 1
        if ratio <= 0:
            raise ValueError("official S1 block geometry is not representable")
    return finish(p, hardware, "s1", 97, initial * ratio, ratio, reuse, l1, nz)


def workload(work):
    s, workers = work.s, []
    for start in range(0, s["total_tasks"], s["split_factor"]):
        clients = [[], []] if s["reuse"] else [[]]
        for task in range(start, min(start + s["split_factor"], s["total_tasks"])):
            client = (task - start) % len(clients)
            clients[client].append(work.job(task, 0, client, start))
        if len(clients) == 2 and len(clients[0]) != len(clients[1]):
            clients[1].append(_fake())
        workers.extend(clients)
    return [workers[first : first + 2] for first in range(0, len(workers), 2)]
