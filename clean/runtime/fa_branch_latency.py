from __future__ import annotations
import hashlib
import os
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from . import fa_latency_model as source
from .drop_adapter_schedule import jobs as adapter_jobs
from .varlen_schedule import workload as varlen_work

ROOT = Path(__file__).resolve().parents[1]
PIPES = dict(source.PIPES)
PIPES["vector"] = (PIPES["vector"][0], PIPES["vector"][1] + ("scale_k", "duplicate_k"))


@lru_cache(maxsize=1)
def tiler():
    sdk = source.cann_root()
    cpp = ROOT / "native" / "fa_branch_matmul_tiling.cpp"
    library = next(
        sdk / "lib64" / name
        for name in ("libtiling_api.so", "libtiling_api.a")
        if (sdk / "lib64" / name).is_file()
    )
    identity = source.cost.sha256(
        {
            "source": cpp.read_text(),
            "api": hashlib.sha256(library.read_bytes()).hexdigest(),
        }
    )
    folder = ROOT / "out/fa_branch_host"
    folder.mkdir(parents=True, exist_ok=True)
    binary, marker = folder / "matmul_tiling", folder / "signature"
    if not binary.is_file() or not marker.is_file() or marker.read_text() != identity:
        command = [
            "g++",
            "-O0",
            "-std=c++17",
            str(cpp),
            "-I",
            str(sdk / "include"),
            "-I",
            str(sdk / "include/ascendc/host_api"),
            "-L",
            str(sdk / "lib64"),
            "-Wl,-rpath," + str(sdk / "lib64"),
            "-ltiling_api",
            "-lplatform",
            "-lregister",
            "-lascendalog",
            "-lgraph_base",
            "-ldl",
            "-lc_sec",
            "-o",
            str(binary) + ".tmp",
        ]
        built = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if built.returncode:
            raise RuntimeError(
                "CPU Matmul trace helper failed: " + built.stdout[-2000:]
            )
        Path(str(binary) + ".tmp").replace(binary)
        marker.write_text(identity)
    return str(binary), identity


@lru_cache(maxsize=8192)
def geometry(requests):
    binary, _ = tiler()
    result = subprocess.run(
        [str(binary)],
        input="\n".join(" ".join(map(str, row)) for row in requests) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    names = (
        "base_m",
        "base_n",
        "base_k",
        "depth_a1",
        "depth_b1",
        "db_l0c",
        "step_m",
        "step_n",
        "db_l0a",
        "db_l0b",
        "step_ka",
        "step_kb",
        "iterate_order",
        "trans_length",
    )
    lines = result.stdout.strip().splitlines()
    if (
        result.returncode
        or len(lines) != 2
        or any(len(line.split()) != len(names) for line in lines)
    ):
        raise ValueError(
            "CPU Matmul API rejected branch geometry: " + result.stderr[-500:]
        )
    return [dict(zip(names, map(int, line.split()))) for line in lines]


def implementation(p, s):
    if p["layout"] == "TND":
        return (
            "varlen_same_ab"
            if s["same_ab"]
            else "varlen_l1" if s["l1"] else "varlen_gm"
        )
    return "b" if s["family"] in ("b", "b_drop") else s["family"]


def cost_class(p, s):
    family = implementation(p, s)
    path = "paired_reuse" if s["reuse"] else "l1_carry" if s["l1"] else "gm"
    return "|".join(
        (
            family,
            p["dtype"],
            p["layout"],
            path,
            "NZ" if s["nz"] else "ND",
            "STATIC" if s.get("static_matmul") else s["policy"],
        )
    )


def matmul_cache_trace(
    p, mm, stage, rows, columns, shared=False, carried=False, q_carried=False
):
    ceil, align = source.cost.ceil_div, source.cost.align
    m, n, k = (
        rows,
        columns if stage == 0 else p["DV"],
        p["D"] if stage == 0 else columns,
    )
    bm, bn, bk = (mm[name] for name in ("base_m", "base_n", "base_k"))
    mt, nt, kt = ceil(m, bm), ceil(n, bn), ceil(k, bk)
    sm, sn = max(1, mm["step_m"]), max(1, mm["step_n"])
    order = mm["iterate_order"]

    def capacity(depth, major):
        if depth > 2:
            return depth if depth >= kt * major else depth - 2
        return 0 if depth < kt * major or kt == 1 and depth == 2 else depth

    ac, bc = capacity(mm["depth_a1"], sm), capacity(mm["depth_b1"], sn)
    astream = max(1, mm["depth_a1"] - ac)
    bstream = max(1, mm["depth_b1"] - bc)
    ap, bp = 0, 0
    trace = []
    outer = range(0, mt, sm) if order == 1 else range(0, nt, sn)
    for major in outer:
        if order == 1:
            ap = 0
        else:
            bp = 0
        minor = range(0, nt, sn) if order == 1 else range(0, mt, sm)
        for secondary in minor:
            first = (
                range(secondary, min(secondary + sn, nt))
                if order == 1
                else range(secondary, min(secondary + sm, mt))
            )
            for i in first:
                if order == 1:
                    bp = 0
                else:
                    ap = 0
                inner = (
                    range(major, min(major + sm, mt))
                    if order == 1
                    else range(major, min(major + sn, nt))
                )
                for j in inner:
                    mi, ni = (j, i) if order == 1 else (i, j)
                    parts = []
                    for ki in range(kt):
                        ai = (mi * kt + ki) % (sm * kt) if order == 1 else ki
                        bi = ki if order == 1 else (ni * kt + ki) % (sn * kt)
                        amiss = not (ai < ac and ai < ap) and not q_carried
                        bmiss = not (bi < bc and bi < bp) and not (shared or carried)
                        if amiss and ai < ac:
                            ap += 1
                        if bmiss and bi < bc:
                            bp += 1
                        ak = align(min(bk, k - ki * bk))
                        ar, br = min(bm, m - mi * bm), min(bn, n - ni * bn)
                        ar_bytes = 2 * ar * ak if amiss else 0
                        br_bytes = 2 * br * ak if bmiss else 0
                        factor = p["B"] if p["layout"] == "SBH" else 1
                        astride = p["D"] * (
                            p["N1"] * factor if p["layout"] != "BNSD" else 1
                        )
                        bwidth = p["D"] if stage == 0 else p["DV"]
                        bstride = bwidth * (
                            p["N2"] * factor if p["layout"] != "BNSD" else 1
                        )
                        gap = (
                            max(0, ar - 1) * max(0, astride - p["D"]) * 2
                            if amiss and stage == 0
                            else 0
                        )
                        gap += (
                            max(0, br - 1) * max(0, bstride - bwidth) * 2
                            if bmiss
                            else 0
                        )
                        parts.append(
                            {
                                "read_kb": (ar_bytes + br_bytes) / 1024,
                                "loads": int(amiss) + int(bmiss),
                                "address_gap_kb": gap / 1024,
                                "a_slot": ai if ai < ac else ac + ki % astream,
                                "b_slot": bi if bi < bc else bc + ki % bstream,
                                "a_miss": amiss,
                                "b_miss": bmiss,
                                "a_cached": ai < ac,
                                "b_cached": bi < bc,
                            }
                        )
                    trace.append(parts)
    return trace


class Work:
    def __init__(self, inputs, hardware):
        self.p = source.cost.normalize(inputs)
        self.s = source.schedule(self.p, hardware)
        if self.p["layout"] == "TND" and not self.s["same_ab"]:
            self.s["l1"] = (
                self.s["s1_base"] <= 128 and max(self.p["D"], self.p["DV"]) <= 256
            )
        self.hardware = hardware
        self.mm = [
            dict(record)
            for record in geometry(tuple(source.mm_requests(self.p, self.s, hardware)))
        ]
        a, b = self.mm
        static = (
            self.p["dtype"] == "BF16"
            and not self.p["has_drop_mask"]
            and self.p["layout"] in ("BSND", "SBH")
            and self.s["family"] in ("b", "s1")
        )
        static = static and (a["base_m"], a["base_n"], a["base_k"]) == (
            b["base_m"],
            b["base_k"],
            b["base_n"],
        )
        templates = (
            (16, 16, 80),
            (32, 32, 80),
            (48, 48, 80),
            (64, 64, 80),
            (16, 16, 96),
            (32, 32, 96),
        )
        static = static and (
            (a["base_m"], a["base_n"], a["base_k"]) in templates
            if self.s["family"] == "b"
            else a["base_m"] == a["base_n"] == 128 and a["base_k"] in (80, 96, 128)
        )
        self.s["static_matmul"] = bool(
            static and (self.p["layout"] == "BSND" or a["base_k"] == 96)
        )
        self.previous = {}
        self.cache = {}
        self.pairs = []

    def refine_cube(self, stage, rows, columns, heads, job):
        return job

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
                    and self.previous.get((group, stage, "pair", offset)) == key
                ),
                bool(
                    s["family"] == "s1"
                    and s["l1"]
                    and stage == 1
                    and self.previous.get((group, client, "value")) == key[:2]
                ),
                p["layout"] == "TND" and s["l1"] and stage == 0 and offset > 0,
            )
            for stage in (0, 1)
        )
        identity = (
            rows,
            columns,
            cached_sharing,
            offset == 0,
            offset + columns == p["S2"],
        )
        if identity in self.cache:
            for stage in (0, 1):
                self.previous[group, stage, "pair", offset] = key
            self.previous[group, client, "value"] = key[:2]
            return self.cache[identity]
        for stage in (0, 1):
            shared = bool(
                s["reuse"]
                and client == 1
                and self.previous.get((group, stage, "pair", offset)) == key
            )
            carried = bool(
                s["family"] == "s1"
                and s["l1"]
                and stage == 1
                and self.previous.get((group, client, "value")) == key[:2]
            )
            q_carried = p["layout"] == "TND" and s["l1"] and stage == 0 and offset > 0
            job = source.cube_job(
                p,
                s,
                self.mm[stage],
                stage * 2,
                rows,
                columns,
                heads,
                kv_heads,
                shared,
                carried,
                q_carried,
            )
            if s["same_ab"] and s["policy"] == "UNSPLITK":
                mm = self.mm[stage]
                mt = source.cost.ceil_div(rows, mm["base_m"])
                nt = source.cost.ceil_div(
                    columns if stage == 0 else p["DV"], mm["base_n"]
                )
                kt = source.cost.ceil_div(
                    p["D"] if stage == 0 else columns, mm["base_k"]
                )
                job["loads"] = source.cost.ceil_div(mt * kt, 2) + source.cost.ceil_div(
                    nt * kt, 2
                )
            if p["layout"] == "TND" and s["l1"] and not s["same_ab"]:
                parts = [min(256, columns - first) for first in range(0, columns, 256)]
                job["loads"] = (
                    (0 if q_carried else 1) + len(parts)
                    if stage == 0
                    else 2 * len(parts)
                )
                job["calls"] = len(parts)
                job["commands"] = len(parts) + job["tiles"]
                if stage == 1:
                    job["k_slices"] = [
                        source.cost.align(part) / sum(map(source.cost.align, parts))
                        for part in parts
                    ]
            if p["layout"] == "TND" and s["l1"] and stage == 0:
                job["output_group_tiles"] = source.cost.ceil_div(
                    rows, self.mm[stage]["base_m"]
                ) * source.cost.ceil_div(256, self.mm[stage]["base_n"])
            if (
                not batch_mode
                and not (p["layout"] == "TND" and s["l1"])
                and s["policy"] == "NORMAL"
            ):
                trace = matmul_cache_trace(
                    p, self.mm[stage], stage, rows, columns, shared, carried, q_carried
                )
                job["cache_trace"] = trace
                for name in ("read_kb", "loads", "address_gap_kb"):
                    job[name] = sum(part[name] for output in trace for part in output)
            cubes.append(self.refine_cube(stage, rows, columns, heads, job))
            sharing.append((shared, carried, q_carried))
            self.previous[group, stage, "pair", offset] = key
        self.previous[group, client, "value"] = key[:2]
        vectors = []
        for owned in (
            (source.cost.ceil_div(rows, 2), rows // 2) if s["same_ab"] else (rows,)
        ):
            if not owned:
                vectors.append(({}, {}))
                continue
            v1, v2 = source.vector_job(
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
        return self.cache[identity]

    def offsets(self):
        s, p = self.s, self.p
        window = (
            s["s2_base"] * s["n_ratio"]
            if s["same_ab"] or s["family"] in ("general", "varlen")
            else p["S2"]
        )
        return range(0, p["S2"], window)


def workload(inputs, hardware, work_type=Work):
    work = work_type(inputs, hardware)
    work.pairs = source.schedule_module(work.s).workload(work)
    return work


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
    return {pipe: dict(values) for pipe, values in totals.items()}


def parameter_names():
    return tuple((pipe, name) for pipe, (_, names) in PIPES.items() for name in names)


class BranchGraph:
    def __init__(self, work):
        self.work = work
        self.names = parameter_names()
        self.columns = {key: i for i, key in enumerate(self.names)}
        self.templates, self.template_ids, self.phases = [], {}, []
        self.left, self.right, self.labels = [], [], []
        self.tails, self.buffers = {}, {}
        self.zero = self.template("cscalar", {}, "startup")
        self.events = Counter()

    def template(self, pipe, values, phase, divisor=1):
        row = [0.0] * len(self.names)
        for name in PIPES[pipe][1]:
            row[self.columns[pipe, name]] = float(values.get(name, 0)) / divisor
        key = phase, tuple(row)
        if key not in self.template_ids:
            self.template_ids[key] = len(self.templates)
            self.templates.append(key[1])
            self.phases.append(phase)
        return self.template_ids[key]

    def node(self, deps=(), label=None):
        deps = list(dict.fromkeys(d for d in deps if d >= 0))
        while len(deps) > 2:
            deps.append(self.node(deps[:2]))
            deps = deps[2:]
        at = len(self.left)
        self.left.append(deps[0] if deps else -1)
        self.right.append(deps[1] if len(deps) > 1 else -1)
        self.labels.append(self.zero if label is None else label)
        return at

    def resolve(self, at, deps):
        self.left[at] = self.node(deps)

    def operation(
        self, pipe, owner, values, phase, deps=(), reads=(), writes=(), divisor=1
    ):
        key = pipe, owner
        deps = list(deps) + [self.tails.get(key, -1)]
        names = list(reads) + list(writes)
        for buf in names:
            writer, readers = self.buffers.get((owner, buf), (-1, []))
            deps.append(writer)
            if buf in writes:
                deps.extend(readers)
        result = self.node(deps, self.template(pipe, values, phase, divisor))
        self.tails[key] = result
        for buf in reads:
            writer, readers = self.buffers.get((owner, buf), (-1, []))
            self.buffers[owner, buf] = writer, readers + [result]
        for buf in writes:
            self.buffers[owner, buf] = result, []
        return result

    def vector(self, owner, unit, stage, launch):
        phase = "vec1" if stage == 1 else "vec2"
        job = unit["vector"][owner if self.work.s["same_ab"] else 0][stage // 2]
        if unit.get("fake") or not job.get("chunks"):
            return launch, launch
        chunks = int(job["chunks"])
        name = implementation(self.work.p, self.work.s)
        b_or_s1 = name in ("b", "s1")
        if stage == 1:
            inp, calc, out = "stage1Pong", "stage1Ping", "vec1Output"
            pre_names, post_names = ("scale_k", "nz2nd_k"), ("cast_k", "nd2nz_k")
        elif b_or_s1:
            inp, calc, out = "stage1Ping", "stage1Pong", "commonOutput"
            pre_names, post_names = ("nz2nd_k",), ("cast_k", "nd2nz_k")
        else:
            inp, calc, out = "common", "stage2", "stage2"
            pre_names, post_names = ("nz2nd_k",), ("cast_k", "nd2nz_k")
        pre = {key: job.get(key, 0) for key in pre_names}
        main = {
            key: value
            for key, value in job.items()
            if key not in pre_names + post_names
        }
        post = {key: job.get(key, 0) for key in post_names}
        issued, finished = launch, launch
        for chunk in range(chunks):
            issued = self.operation(
                "vscalar", owner, job, phase, (issued,), divisor=chunks
            )
            read_deps = [issued, launch]
            if stage == 3 and not b_or_s1:
                read_deps.append(self.tails.get(("vmte3", owner), -1))
            loaded = self.operation(
                "vmte2", owner, job, phase, read_deps, writes=(inp,), divisor=chunks
            )
            prefix = self.operation(
                "vector",
                owner,
                pre,
                phase,
                (loaded,),
                reads=(inp,),
                writes=(calc,),
                divisor=chunks,
            )
            middle = self.operation(
                "vector",
                owner,
                main,
                phase,
                (prefix,),
                reads=(calc,),
                writes=(calc, "common"),
                divisor=chunks,
            )
            cast = self.operation(
                "vector",
                owner,
                post,
                phase,
                (middle,),
                reads=(calc,),
                writes=(out,),
                divisor=chunks,
            )
            finished = self.operation(
                "vmte3", owner, job, phase, (cast,), reads=(out,), divisor=chunks
            )
        if stage == 3:
            self.tails["vmte2", owner] = self.node(
                (self.tails.get(("vmte2", owner), -1), finished)
            )
        self.events["vector_chunks"] += chunks if not unit.get("fake") else 0
        return issued, finished


    def pair(self, pair, prefix):
        self.tails, self.buffers = {}, {}
        same = self.work.s["same_ab"]
        workers = [pair[0], pair[0]] if same else pair
        controlled = same or implementation(self.work.p, self.work.s) == "varlen_l1"
        launches, done, vec_done = {}, {}, {}
        for owner, tasks in enumerate(workers):
            for task in range(len(tasks)):
                for stage in (0, 2):
                    key = (0 if same else owner), task, stage
                    if key not in done:
                        launches[key], done[key] = self.node(), self.node()
                for stage in (1, 3):
                    vec_done[owner, task, stage] = self.node()
        init = {
            "workers": 1,
            "batch_entries": self.work.p["B"] if self.work.p["layout"] == "TND" else 0,
        }
        startup = self.operation("cscalar", 0, init, "startup", (prefix,))
        self.tails["cscalar", 0] = startup
        for owner in range(len(workers)):
            self.operation("vscalar", owner, init, "startup", (prefix,))
        client_messages = [[] for _ in workers]
        for owner, tasks in enumerate(workers):
            previous = self.tails["vscalar", owner]
            program = source.schedule_module(self.work.s).program(
                tasks, workers[1 - owner] if len(workers) == 2 else [], self.work.s
            )
            for kind, task, stage in program:
                key = (
                    (0 if same else 1 - owner if kind == "peer_wait" else owner),
                    task,
                    stage,
                )
                if kind in ("wait", "peer_wait"):
                    previous = self.node((previous, done[key]))
                elif kind == "launch":
                    self.resolve(launches[key], (previous,))
                    client_messages[owner].append(key)
                else:
                    issued, end = self.vector(owner, tasks[task], stage, previous)
                    self.resolve(vec_done[owner, task, stage], (end,))
                    previous = issued
            self.tails["vscalar", owner] = previous
        if controlled:
            owners = (0,) if same else range(len(workers))
            messages = []
            for step in range(max(map(len, workers)) + 1):
                messages.extend(
                    (owner, step, 0) for owner in owners if step < len(workers[owner])
                )
                messages.extend(
                    (owner, step - 1, 2)
                    for owner in owners
                    if 0 <= step - 1 < len(workers[owner])
                )
        else:
            messages = [
                items[index]
                for index in range(max(map(len, client_messages)))
                for items in client_messages
                if index < len(items)
            ]
        slots = {}
        for owner, task, stage in messages:
            key = owner, task, stage
            dependencies = [startup]
            if not controlled:
                dependencies.append(launches[key])
            if stage == 2:
                dependencies.extend(
                    vec_done[peer, task, 1] for peer in (range(2) if same else (owner,))
                )
            launch = self.node(dependencies)
            end = self.cube(owner, workers[owner][task], stage, launch, slots)
            self.resolve(done[key], (end,))
        return self.node(list(self.tails.values()))

    def build(self):
        prefix = self.node()
        prefix = source.drop_adapter_schedule.build(self, prefix)
        roots, seen = [], set()
        for pair in self.work.pairs:
            identity = tuple(tuple(id(job) for job in tasks) for tasks in pair)
            if identity not in seen:
                seen.add(identity)
                roots.append(self.pair(pair, prefix))
        root = self.node(roots)
        a, b, which = (
            np.asarray(x, dtype=np.int32) for x in (self.left, self.right, self.labels)
        )
        order = np.empty(len(a), dtype=np.int32)
        if source.graph_engine().fa_latency_order(
            len(a), a.ctypes.data, b.ctypes.data, order.ctypes.data
        ) != len(a):
            raise RuntimeError("branch source dependency graph contains a cycle")
        inverse = np.empty(len(a), dtype=np.int32)
        inverse[order] = np.arange(len(a), dtype=np.int32)
        a = np.where(a[order] >= 0, inverse[np.maximum(a[order], 0)], -1).astype(
            np.int32
        )
        b = np.where(b[order] >= 0, inverse[np.maximum(b[order], 0)], -1).astype(
            np.int32
        )
        return {
            "a": np.ascontiguousarray(a),
            "b": np.ascontiguousarray(b),
            "which": np.ascontiguousarray(which[order]),
            "root": int(inverse[root]),
            "templates": np.asarray(self.templates, dtype=np.float64),
            "phases": self.phases,
            "execution_events": dict(self.events),
        }


@lru_cache(maxsize=1)
def compact_engine():
    import ctypes

    cpp = ROOT / "native" / "fa_branch_graph.cpp"
    folder = ROOT / "out/fa_branch_host"
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / "graph_compact.so"
    marker = folder / "graph_compact.signature"
    key = hashlib.sha256(cpp.read_bytes()).hexdigest() + os.uname().machine
    if not library.is_file() or not marker.is_file() or marker.read_text() != key:
        result = subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(cpp),
                "-o",
                str(library) + ".tmp",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError(
                "CPU graph compactor build failed: " + result.stdout[-1000:]
            )
        Path(str(library) + ".tmp").replace(library)
        marker.write_text(key)
    engine = ctypes.CDLL(str(library))
    engine.fa_compact_graph.argtypes = [ctypes.c_int] * 2 + [ctypes.c_void_p] * 8
    engine.fa_compact_graph.restype = ctypes.c_int
    return engine


def compact_graph(graph):
    import ctypes

    n = len(graph["a"])
    zero = np.ascontiguousarray(np.all(graph["templates"] == 0, axis=1), dtype=np.uint8)
    a, b, labels = (np.empty(n, dtype=np.int32) for _ in range(3))
    root = ctypes.c_int()
    count = compact_engine().fa_compact_graph(
        n,
        graph["root"],
        graph["a"].ctypes.data,
        graph["b"].ctypes.data,
        graph["which"].ctypes.data,
        zero.ctypes.data,
        a.ctypes.data,
        b.ctypes.data,
        labels.ctypes.data,
        ctypes.byref(root),
    )
    if count <= 0 or root.value < 0:
        raise ValueError("dependency graph compaction failed")
    return {
        **graph,
        "a": a[:count].copy(),
        "b": b[:count].copy(),
        "which": labels[:count].copy(),
        "root": root.value,
    }
