from collections import Counter

from .. import inputs as cost
from ..features import TRANSFER_FEATURES as FEATURES


def transfer(rows, width, stride, name, pressure=0):
    full, tail = divmod(width * 2, 256)
    partial = rows if name != "P" and tail else 0
    return {
        "payload_kb": rows * width * 2 / 1024,
        "copy_calls": 1,
        "strided_rows": rows if stride > width else 0,
        "source": name,
        "nd2nz_partial_rows": partial,
        "nd2nz_split_sector_kb": (
            partial * full * (256 - tail) / 1024 if width * 2 % 128 else 0
        ),
        "capacity_pressure_kb": rows * width * 2 / 1024 * pressure,
        "rows": rows,
        "width_elements": width,
        "stride_elements": stride,
    }


def add_copies(work, unit, offset):
    if "dma_groups" in unit["cube"][0]:
        return unit
    p = work.p
    footprint = 2 * p["B"] * p["N2"] * p["S2"] * (p["D"] + p["DV"])
    pressure = max(0.0, 1.0 - work.hardware["l2_bytes"] / footprint)
    rows, columns = unit["rows"], unit["columns"]
    chunks = [min(256, columns - pos) for pos in range(0, columns, 256)]
    q = [] if offset else [transfer(rows, p["D"], p["N1"] * p["D"], "Q")]
    first = [
        [transfer(count, p["D"], p["N2"] * p["D"], "K", pressure)]
        for count in chunks
    ]
    second = [
        [
            transfer(
                cost.align(rows),
                cost.align(count),
                cost.align(count),
                "P",
            ),
            transfer(count, p["DV"], p["N2"] * p["DV"], "V", pressure),
        ]
        for count in chunks
    ]
    unit["cube"][0].update(dma_query=q, dma_groups=first)
    unit["cube"][1].update(dma_query=[], dma_groups=second)
    for job in unit["cube"]:
        copies = job["dma_query"] + [
            item for group in job["dma_groups"] for item in group
        ]
        job["dma_features"] = {
            name: sum(item[name] for item in copies) for name in FEATURES
        }
    return unit


def add_vector_reads(work):
    for unit in work.cache.values():
        for _, stage2 in unit["vector"]:
            if not stage2["accumulator_read_bytes"]:
                size = 4 * unit["rows"] * cost.align(work.p["DV"])
                stage2["read_kb"] += size / 1024
                stage2["loads"] += stage2["chunks"]
                stage2["accumulator_read_bytes"] = size


def features(work):
    result = Counter()
    for pair in work.pairs:
        for tasks in pair:
            for unit in tasks:
                for job in unit["cube"]:
                    result.update(job["dma_features"])
    return dict(result)
