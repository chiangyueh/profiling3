from .. import inputs as cost


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
