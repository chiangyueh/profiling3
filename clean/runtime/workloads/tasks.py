from .. import inputs as input_utils, schedules
from ..native.matmul import geometry, mm_requests
from . import matmul, transfer, vector


class Work:
    def __init__(self, inputs, hardware, schedule):
        self.p = inputs
        self.s = dict(schedule)
        self.hardware = hardware
        self.mm = [dict(record) for record in geometry(tuple(mm_requests(self.p, self.s, hardware)))]
        schedules.configure_matmul(self.p, self.s, self.mm)
        self.manual_transfer = schedules.implementation(self.p, self.s) == "varlen_l1"
        self.previous = {}
        self.cache = {}
        self.pairs = []

    def job(self, task, offset, client, group):
        p, s = self.p, self.s
        batch_mode = s["family"] in ("b", "b_drop")
        rows = (
            p["S1"]
            if batch_mode
            else min(s["s1_base"], p["S1"] - task % s["s1_outer"] * s["s1_base"])
        )
        head = 0 if batch_mode else task // s["s1_outer"] % p["N1"]
        batch = task if batch_mode else task // (s["s1_outer"] * p["N1"])
        heads, kv_heads = (p["N1"], p["N2"]) if batch_mode else (1, 1)
        window = (
            s["s2_base"] * s["n_ratio"]
            if s["same_ab"] or s["family"] in ("general", "varlen")
            else p["S2"]
        )
        columns = min(window, p["S2"] - offset)
        cubes, sharing = [], []
        key = batch, head // (p["N1"] // p["N2"]), offset
        cached_sharing = tuple(
            (
                bool(
                    s["reuse"]
                    and client == 1
                    and self.previous.get((group, bmm_index, "pair", offset)) == key
                ),
                bool(
                    s["family"] == "s1"
                    and s["l1"]
                    and bmm_index == 1
                    and self.previous.get((group, client, "value")) == key[:2]
                ),
                p["layout"] == "TND" and s["l1"] and bmm_index == 0 and offset > 0,
            )
            for bmm_index in (0, 1)
        )
        identity = (
            rows,
            columns,
            cached_sharing,
            offset == 0,
            offset + columns == p["S2"],
        )
        if identity in self.cache:
            for bmm_index in (0, 1):
                self.previous[group, bmm_index, "pair", offset] = key
            self.previous[group, client, "value"] = key[:2]
            return self.cache[identity]
        for bmm_index in (0, 1):
            shared = bool(
                s["reuse"]
                and client == 1
                and self.previous.get((group, bmm_index, "pair", offset)) == key
            )
            carried = bool(
                s["family"] == "s1"
                and s["l1"]
                and bmm_index == 1
                and self.previous.get((group, client, "value")) == key[:2]
            )
            q_carried = p["layout"] == "TND" and s["l1"] and bmm_index == 0 and offset > 0
            job = matmul.cube_job(
                p,
                s,
                self.mm[bmm_index],
                bmm_index,
                rows,
                columns,
                heads,
                kv_heads,
                shared,
                carried,
                q_carried,
            )
            if s["same_ab"] and s["policy"] == "UNSPLITK":
                mm = self.mm[bmm_index]
                mt = input_utils.ceil_div(rows, mm["base_m"])
                nt = input_utils.ceil_div(
                    columns if bmm_index == 0 else p["DV"], mm["base_n"]
                )
                kt = input_utils.ceil_div(
                    p["D"] if bmm_index == 0 else columns, mm["base_k"]
                )
                job["loads"] = input_utils.ceil_div(mt * kt, 2) + input_utils.ceil_div(
                    nt * kt, 2
                )
            if p["layout"] == "TND" and s["l1"] and not s["same_ab"]:
                parts = [min(256, columns - first) for first in range(0, columns, 256)]
                job["loads"] = (
                    (0 if q_carried else 1) + len(parts)
                    if bmm_index == 0
                    else 2 * len(parts)
                )
                job["calls"] = len(parts)
                job["commands"] = len(parts) + job["tiles"]
                if bmm_index == 1:
                    job["k_slices"] = [
                        input_utils.align(part) / sum(map(input_utils.align, parts))
                        for part in parts
                    ]
            if p["layout"] == "TND" and s["l1"] and bmm_index == 0:
                job["output_group_tiles"] = input_utils.ceil_div(
                    rows, self.mm[bmm_index]["base_m"]
                ) * input_utils.ceil_div(256, self.mm[bmm_index]["base_n"])
            if (
                not batch_mode
                and not (p["layout"] == "TND" and s["l1"])
                and s["policy"] == "NORMAL"
            ):
                trace = matmul.matmul_cache_trace(
                    p, self.mm[bmm_index], bmm_index, rows, columns, shared, carried, q_carried
                )
                job["cache_trace"] = trace
                for name in ("read_kb", "loads", "address_gap_kb"):
                    job[name] = sum(part[name] for output in trace for part in output)
            cubes.append(matmul.refine_matmul_job(p, s, self.mm[bmm_index], bmm_index, rows, columns, heads, job))
            sharing.append((shared, carried, q_carried))
            self.previous[group, bmm_index, "pair", offset] = key
        self.previous[group, client, "value"] = key[:2]
        vectors = []
        for owned in (
            (input_utils.ceil_div(rows, 2), rows // 2) if s["same_ab"] else (rows,)
        ):
            if not owned:
                vectors.append(({}, {}))
                continue
            v1, v2 = vector.vector_job(
                p, s, owned, columns, offset == 0, offset + columns == p["S2"]
            )
            v1["scale_k"] = v1["cast_k"]
            if batch_mode:
                v1, v2 = (
                    {key: value * heads for key, value in v.items()} for v in (v1, v2)
                )
            vectors.append((v1, v2))
        identity = (
            rows,
            columns,
            tuple(sharing),
            offset == 0,
            offset + columns == p["S2"],
        )
        if identity not in self.cache:
            self.cache[identity] = {
                "cube": cubes,
                "vector": vectors,
                "rows": rows,
                "columns": columns,
                "fake": False,
                "heads": heads,
            }
        unit = self.cache[identity]
        if self.manual_transfer:
            transfer.add_copies(self, unit, offset)
        return unit

    def offsets(self):
        s, p = self.s, self.p
        window = (
            s["s2_base"] * s["n_ratio"]
            if s["same_ab"] or s["family"] in ("general", "varlen")
            else p["S2"]
        )
        return range(0, p["S2"], window)
