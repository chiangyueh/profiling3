from collections import Counter

import numpy as np

from .. import schedules
from ..features import PARAMETERS, PIPES, TRANSFER_FEATURES, TRANSFER_PARAMETERS
from ..schedules import implementation
from ..native.graph import graph_engine
from . import adapter, matmul, transfer, vector


class Graph:
    def __init__(self, work):
        self.work = work
        self.names = TRANSFER_PARAMETERS if work.manual_transfer else PARAMETERS
        self.columns = {key: i for i, key in enumerate(self.names)}
        self.templates, self.template_ids, self.phases = [], {}, []
        self.left, self.right, self.labels = [], [], []
        self.tails, self.buffers = {}, {}
        self.zero = self.template("cscalar", {}, "startup")
        self.events = Counter()
        self.query_data = {}
        self.kv_iteration = self.p_iteration = 0

    def template(self, pipe, values, phase, divisor=1):
        row = [0.0] * len(self.names)
        names = TRANSFER_FEATURES if pipe == "mte2" and self.work.manual_transfer else PIPES[pipe][1]
        for name in names:
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

    def cube(self, owner, unit, stage, launch, slots):
        builder = transfer if self.work.manual_transfer else matmul
        return builder.cube(self, owner, unit, stage, launch, slots)

    def vector(self, owner, unit, stage, launch):
        return vector.build(self, owner, unit, stage, launch)


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
            program = schedules.module(self.work.s).program(
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
        prefix = adapter.build(self, prefix)
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
        if graph_engine().fa_latency_order(
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
