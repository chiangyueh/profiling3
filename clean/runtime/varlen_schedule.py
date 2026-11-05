from . import fa_latency_inputs as cost
from .schedule_common import controlled_program, finish, sequence_work, stages


def schedule(p, hardware):
    b, q, k = (p[name] for name in ("B", "S1", "S2"))
    aq, ak = cost.align(q), cost.align(k)
    best = 128 if b * q <= 10000 or k >= 1024 else 256 if b >= 5 or k >= 512 else 512
    base, ratio = min(best, aq), min(8, cost.ceil_div(k, min(128, ak)))
    return finish(p, hardware, "varlen", 94, base, ratio)


def workload(work):
    return sequence_work(work, 1 if work.s["same_ab"] else 2)


def program(tasks, peer, schedule):
    if schedule["l1"]:
        return controlled_program(tasks, peer, schedule)
    return stages(len(tasks))
