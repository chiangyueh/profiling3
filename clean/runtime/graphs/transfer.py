from .. import inputs as cost
from . import matmul


def load(graph, item, phase, launch, slot, slots):
    graph.events["dma_copy_calls"] += 1
    graph.events["dma_payload_bytes"] += round(item["payload_kb"] * 1024)
    graph.events["nd2nz_strided_rows"] += item["strided_rows"]
    return graph.operation("mte2", 0, item, phase, (launch, slots.get(slot, -1)))


def cube(graph, owner, unit, stage, launch, slots):
    if unit.get("fake"):
        return matmul.cube(graph, owner, unit, stage, launch, slots)
    phase = "bmm1" if stage == 0 else "bmm2"
    job, mm = unit["cube"][stage // 2], graph.work.mm[stage // 2]
    tiles = job["micro_tiles"]
    ab_count = max(1, min(mm["db_l0a"], mm["db_l0b"]))
    c_count = max(
        1,
        min(
            mm["db_l0c"],
            graph.work.hardware.get("l0c_bytes", 131072)
            // (mm["base_m"] * mm["base_n"] * 4),
        ),
    )
    query_slot = ("dma_q", owner)
    if job["dma_query"]:
        graph.query_data[owner] = load(graph,
            job["dma_query"][0], phase, launch, query_slot, slots
        )
    fixed, accumulators, serial = [], {}, 0
    for api, copies in enumerate(job["dma_groups"]):
        if stage == 0:
            serial = 0
        previous_fix = graph.tails.get(("fixpipe", 0), -1)
        issued = graph.operation("cscalar", 0, {"commands": 1}, phase, (launch,))
        kv_slot = ("dma_kv", graph.kv_iteration % 2)
        graph.kv_iteration += 1
        if stage == 0:
            a_ready = graph.query_data[owner]
            b_ready = load(graph, copies[0], phase, issued, kv_slot, slots)
            selected = [
                (index, tile)
                for index, tile in enumerate(tiles)
                if tile["api"] == api
            ]
            p_slot = None
        else:
            p_slot = ("dma_p", graph.p_iteration % 2)
            graph.p_iteration += 1
            a_ready = load(graph, copies[0], phase, issued, p_slot, slots)
            b_ready = load(graph, copies[1], phase, issued, kv_slot, slots)
            selected = list(enumerate(tiles))
        released = []
        for output, tile in selected:
            c_slot = ("c0", output % c_count) if stage == 0 else ("c0", output)
            partial = (
                accumulators.get(
                    output, graph.node((slots.get(c_slot, -1), previous_fix))
                )
                if stage
                else graph.node((slots.get(c_slot, -1), previous_fix))
            )
            lengths = (
                tile["k"]
                if stage == 0
                else [
                    min(mm["base_k"], copies[1]["rows"] - pos)
                    for pos in range(0, copies[1]["rows"], mm["base_k"])
                ]
            )
            for index, length in enumerate(lengths):
                l0_slot = ("ab0", serial % ab_count)
                serial += 1
                m, n, k = map(
                    cost.align, (tile["m"], tile["n"], length)
                )
                local = {"l0_kb": 2 * (m + n) * k / 1024, "tiles": 1}
                moved = graph.operation(
                    "mte1",
                    0,
                    local,
                    phase,
                    (a_ready, b_ready, slots.get(l0_slot, -1)),
                )
                released.append(moved)
                arithmetic = {
                    "flops_m": 2 * m * n * k / 1e6,
                    "tiles": 1,
                    "output_tiles": int(index == 0 and (stage == 0 or api == 0)),
                }
                partial = graph.operation(
                    "cube", 0, arithmetic, phase, (moved, partial)
                )
                slots[l0_slot] = partial
                graph.operation(
                    "cscalar",
                    0,
                    {"commands": 1 + tile["barrier"]},
                    phase,
                    (issued,),
                )
                graph.events["k_tiles"] += 1
                graph.events["pipe_m_barriers"] += tile["barrier"]
            if stage == 0:
                fixed.append(
                    store(graph,
                        tile, job, phase, partial, int(output == selected[0][0])
                    )
                )
                slots[c_slot] = fixed[-1]
            else:
                accumulators[output] = partial
        end_read = graph.node(released)
        if stage == 0:
            graph.tails["mte1", 0] = graph.node(
                (graph.tails.get(("mte1", 0), -1), partial)
            )
            graph.events["matmul_end_l0_fences"] += 1
        slots[kv_slot] = end_read
        if p_slot is not None:
            slots[p_slot] = end_read
        else:
            slots[query_slot] = end_read
        graph.events["matmul_calls"] += 1
    if stage == 2:
        for output, tile in enumerate(tiles):
            fixed.append(
                store(graph, tile, job, phase, accumulators[output], int(output == 0))
            )
            slots["c0", output] = fixed[-1]
        graph.events["get_tensor_c_calls"] += 1
        graph.tails["mte1", 0] = graph.node(
            (graph.tails.get(("mte1", 0), -1), partial)
        )
        graph.events["matmul_end_l0_fences"] += 1
    return graph.node(fixed)


def store(graph, tile, job, phase, dependency, first):
    write = tile["m"] * cost.align(tile["n"]) * 4 / 1024
    values = {
        "nz_write_kb": write if job.get("nz_write_kb") else 0,
        "nd_write_kb": 0 if job.get("nz_write_kb") else write,
        "stores": 1,
        "calls": first,
    }
    return graph.operation("fixpipe", 0, values, phase, (dependency,))
