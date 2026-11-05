from collections import Counter

from .. import inputs, schedules
from ..features import PIPES
from . import adapter, transfer
from .adapter import jobs as adapter_jobs
from .tasks import Work


def build(values, hardware, api_policy=None):
    p = inputs.normalize(values)
    schedule = schedules.select(p, hardware)
    implementation = schedules.implementation(p, schedule)
    if implementation == "varlen_l1":
        p = inputs.api_kernel_inputs(p, api_policy)
        schedule = schedules.select(p, hardware)
    work = Work(p, hardware, schedule)
    if implementation == "varlen_l1" and schedules.implementation(work.p, work.s) != "varlen_l1":
        raise ValueError("ND2NZ transfer model requires the Varlen L1 implementation")
    work.pairs = schedules.module(work.s).workload(work)
    if work.manual_transfer:
        transfer.add_vector_reads(work)
    return work, implementation


def active_features(work):
    totals = {pipe: Counter() for pipe in PIPES}
    same = work.s["same_ab"]
    for pair in work.pairs:
        for tasks in pair:
            for task in tasks:
                for job in task["cube"]:
                    for pipe in ("cube", "mte1", "mte2", "fixpipe", "cscalar"):
                        totals[pipe].update(
                            {name: job.get(name, 0) for name in PIPES[pipe][1]}
                        )
                for owner in task["vector"]:
                    for job in owner:
                        for pipe in ("vector", "vmte2", "vmte3", "vscalar"):
                            totals[pipe].update(
                                {name: job.get(name, 0) for name in PIPES[pipe][1]}
                            )
    aiv = sum(2 if same else len(pair) for pair in work.pairs)
    aic = len(work.pairs)
    if work.s["drop_adapter"]:
        aiv, aic = work.hardware["aiv_cores"], work.hardware["aic_cores"]
    totals["cscalar"]["workers"] = aic
    totals["vscalar"]["workers"] = aiv
    for pipe, workers in (("cscalar", aic), ("vscalar", aiv)):
        totals[pipe]["batch_entries"] = (
            work.p["B"] * workers if work.p["layout"] == "TND" else 0
        )
    for job in adapter_jobs(work):
        for pipe in ("vector", "vmte2", "vmte3", "vscalar"):
            totals[pipe].update({name: job.get(name, 0) for name in PIPES[pipe][1]})
    totals["vscalar"]["adapter_workers"] = aiv if work.s["drop_adapter"] else 0
    result = {pipe: dict(values) for pipe, values in totals.items()}
    if work.manual_transfer:
        result["mte2"] = transfer.features(work)
    return result
