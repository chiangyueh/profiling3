from . import fa_latency_inputs as cost
from .schedule_common import finish, program


def is_capable(p, hardware):
    n, q, k, d = (p[name] for name in ("N1", "S1", "S2", "D"))
    aq, ak, ad = map(cost.align, (q, k, d))
    bytes1, bytes2 = n * ((aq + ak) * ad + ak) * 2, n * (aq + ad) * ak * 2
    return (
        ak <= 128
        and bytes1 < int(hardware["l1_bytes"])
        and bytes2 < int(hardware["l1_bytes"])
        and n * aq * ak <= 65536
    )


def schedule(p, hardware):
    aq, ak = cost.align(p["S1"]), cost.align(p["S2"])
    base = min(aq, 256, 8192 // ak // 16 * 16)
    family = "b_drop" if p["has_drop_mask"] else "b"
    return finish(p, hardware, family, 98, base, 0)


def workload(work):
    workers = []
    s = work.s
    for start in range(0, s["total_tasks"], s["split_factor"]):
        workers.append(
            [
                work.job(batch, 0, 0, start)
                for batch in range(
                    start, min(start + s["split_factor"], s["total_tasks"])
                )
            ]
        )
    return [workers[first : first + 2] for first in range(0, len(workers), 2)]
