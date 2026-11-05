from __future__ import annotations

import copy
from collections import Counter
import numpy as np

from . import fa_branch_latency as base

source = base.source
implementation = base.implementation
cost_class = base.cost_class
varlen_work = base.varlen_work
active_features = base.active_features
parameter_names = base.parameter_names
compact_graph = base.compact_graph


def matmul_microtiles(p, s, mm, stage, rows, columns, heads):
    m, n, k = (
        rows,
        columns if stage == 0 else p["DV"],
        p["D"] if stage == 0 else columns,
    )
    bm, bn, bk = (mm[name] for name in ("base_m", "base_n", "base_k"))
    manual = p["layout"] == "TND" and s["l1"]
    nsplit = (
        256
        if manual and stage == 0
        else 128 if manual and s["same_ab"] and stage == 1 else n
    )
    ksplit = (192 if s["same_ab"] else 256) if manual and stage == 1 else k
    kparts = [
        min(bk, min(ksplit, k - outer) - inner)
        for outer in range(0, k, ksplit)
        for inner in range(0, min(ksplit, k - outer), bk)
    ]
    result = []
    api = 0
    for head in range(heads):
        for start in range(0, n, nsplit):
            count = min(nsplit, n - start)
            mt, nt = source.cost.ceil_div(m, bm), source.cost.ceil_div(count, bn)
            sm, sn = max(1, mm["step_m"]), max(1, mm["step_n"])
            order = mm["iterate_order"]
            for major in range(0, mt if order == 1 else nt, sm if order == 1 else sn):
                for secondary in range(
                    0, nt if order == 1 else mt, sn if order == 1 else sm
                ):
                    for first in range(
                        secondary,
                        min(
                            secondary + (sn if order == 1 else sm),
                            nt if order == 1 else mt,
                        ),
                    ):
                        for second in range(
                            major,
                            min(
                                major + (sm if order == 1 else sn),
                                mt if order == 1 else nt,
                            ),
                        ):
                            mi, ni = (second, first) if order == 1 else (first, second)
                            rm, rn = min(bm, m - mi * bm), min(bn, count - ni * bn)
                            mad_m, mad_n = (
                                (bm, bn)
                                if s.get("static_matmul")
                                else (16 if rm == 1 else rm, rn)
                            )
                            barrier = int((mad_m // 16) * (mad_n // 16) < 10)
                            result.append(
                                {
                                    "m": rm,
                                    "n": rn,
                                    "k": kparts,
                                    "barrier": barrier,
                                    "api": api if manual else 0,
                                    "head": head,
                                }
                            )
            api += 1
    return result


def refine_matmul_job(p, s, mm, stage, rows, columns, heads, job):
    tiles = matmul_microtiles(p, s, mm, stage, rows, columns, heads)
    job["micro_tiles"] = tiles
    job["tiles"] = sum(len(t["k"]) for t in tiles)
    job["output_tiles"] = job["stores"] = len(tiles)
    job["flops_m"] = sum(
        2
        * source.cost.align(t["m"])
        * source.cost.align(t["n"])
        * source.cost.align(k)
        / 1e6
        for t in tiles
        for k in t["k"]
    )
    job["l0_kb"] = sum(
        2
        * (source.cost.align(t["m"]) + source.cost.align(t["n"]))
        * source.cost.align(k)
        / 1024
        for t in tiles
        for k in t["k"]
    )
    manual = p["layout"] == "TND" and s["l1"]
    job["calls"] = len({t["api"] for t in tiles}) if manual else 1
    api_steps = job["calls"]
    if manual and stage == 1:
        width = 192 if s["same_ab"] else 256
        api_steps *= source.cost.ceil_div(columns, width) + 1
    job["commands"] = (
        api_steps + job["tiles"] + sum(t["barrier"] * len(t["k"]) for t in tiles)
    )
    job["api_steps"] = api_steps
    job["m_barriers"] = sum(t["barrier"] * len(t["k"]) for t in tiles)
    return job


class Work(base.Work):
    def refine_cube(self, stage, rows, columns, heads, job):
        return refine_matmul_job(
            self.p, self.s, self.mm[stage], stage, rows, columns, heads, job
        )


def workload(inputs, hardware):
    return base.workload(inputs, hardware, Work)


class BranchGraph(base.BranchGraph):
    def cube(self, owner, unit, stage, launch, slot_state):
        phase = 'bmm1' if stage == 0 else 'bmm2'
        job, mm = unit['cube'][stage // 2], self.work.mm[stage // 2]
        if unit.get('fake'):
            self.events['fake_matmul_messages'] += 1
            return self.operation('cscalar', 0, job, phase, (launch,))
        tiles = job['micro_tiles']
        a_count = max(1, mm['depth_a1'] // max(1, mm['step_m']))
        b_count = max(1, mm['depth_b1'] // max(1, mm['step_n']))
        ab_count = max(1, min(mm['db_l0a'], mm['db_l0b']))
        c_count = max(1, min(mm['db_l0c'], self.work.hardware.get('l0c_bytes', 131072) // (mm['base_m'] * mm['base_n'] * 4)))
        last_api = self.node((launch, self.tails.get(('mte1', 0), -1)))
        prior_fix = self.tails.get(('fixpipe', 0), -1)
        operand_data, fixed, previous_api = {}, [], None
        batch_mode = self.work.s['family'] in ('b', 'b_drop')
        bulk = None
        if batch_mode:
            issued = self.operation('cscalar', 0, {'commands': job['api_steps']}, phase, (launch,))
            bulk = self.operation('mte2', 0, job, phase, (issued, last_api))
        for output, tile in enumerate(tiles):
            api_start = tile['api'] != previous_api
            if api_start:
                if fixed:
                    last_api = fixed[-1]
                if not batch_mode:
                    self.operation('cscalar', 0, {'commands': job['api_steps'] / job['calls']}, phase, (launch, last_api))
                previous_api = tile['api']
            partial = launch
            for k, length in enumerate(tile['k']):
                issued = self.operation('cscalar', 0, {'commands': 1}, phase, (launch,))
                a_slot = (stage, 'a1', (output * len(tile['k']) + k) % a_count)
                b_slot = (stage, 'b1', (output * len(tile['k']) + k) % b_count)
                l0_slot = ('ab0', (output * len(tile['k']) + k) % ab_count)
                c_slot = ('c0', output % c_count)
                transfer = job.get('cache_trace')
                values = transfer[output][k] if transfer else job
                if transfer:
                    a_slot = (stage, 'a1', values['a_slot'])
                    b_slot = (stage, 'b1', values['b_slot'])
                dependencies = [issued, last_api]
                for operand, slot in (('a', a_slot), ('b', b_slot)):
                    if not transfer or values[operand + '_miss']:
                        dependencies.append(slot_state.get(slot, -1))
                    else:
                        dependencies.append(operand_data.get(slot, last_api))
                if bulk is not None:
                    loaded = self.node(dependencies + [bulk])
                elif not transfer or values['loads']:
                    fraction = source.cost.align(length) / sum(source.cost.align(v) for v in tile['k'])
                    loaded = self.operation('mte2', 0, values, phase, dependencies,
                                            divisor=1 if transfer else len(tiles) / fraction)
                else:
                    loaded = self.node(dependencies)
                for operand, slot in (('a', a_slot), ('b', b_slot)):
                    if not transfer or values[operand + '_miss']:
                        operand_data[slot] = loaded
                m, n, kval = map(source.cost.align, (tile['m'], tile['n'], length))
                local = {'l0_kb': 2 * (m + n) * kval / 1024, 'tiles': 1}
                moved = self.operation('mte1', 0, local, phase, (loaded, slot_state.get(l0_slot, -1)))
                slot_state[a_slot] = slot_state[b_slot] = moved
                arithmetic = {'flops_m': 2 * m * n * kval / 1e6, 'tiles': 1, 'output_tiles': int(k == 0)}
                computed = self.operation('cube', 0, arithmetic, phase,
                                          (moved, partial, prior_fix if output == 0 and k == 0 else -1,
                                           slot_state.get(c_slot, -1) if k == 0 else -1))
                slot_state[l0_slot] = computed
                partial = computed
                if tile['barrier']:
                    self.operation('cscalar', 0, {'commands': 1}, phase, (launch,))
                    self.events['pipe_m_barriers'] += 1
            if batch_mode:
                self.tails['mte1',0] = self.node((self.tails.get(('mte1',0),-1),partial))
                self.events['batch_mte1_fences'] += 1
            write = tile['m'] * source.cost.align(tile['n']) * 4 / 1024
            output_job = {'nz_write_kb': write if job.get('nz_write_kb') else 0,
                          'nd_write_kb': write if not job.get('nz_write_kb') else 0,
                          'stores': 1, 'calls': int(api_start)}
            stored = self.operation('fixpipe', 0, output_job, phase, (partial,))
            slot_state[c_slot] = stored
            fixed.append(stored)
        self.events['matmul_calls'] += int(job['api_steps'])
        self.events['k_tiles'] += sum(len(tile['k']) for tile in tiles)
        self.events['batch_bulk_copies'] += int(batch_mode)
        return fixed[-1]


def branch_graph(work):
    components, seen = [], set()
    for pair in work.pairs:
        identity = tuple(tuple(id(job) for job in tasks) for tasks in pair)
        if identity in seen:
            continue
        seen.add(identity)
        component = copy.copy(work)
        component.pairs = [pair]
        components.append(compact_graph(BranchGraph(component).build()))
    if len(components) == 1:
        return components[0]
    a, b, labels, templates, phases, roots = [], [], [], [], [], []
    offset, template_offset, events = 0, 0, Counter()
    for graph in components:
        a.append(np.where(graph['a'] >= 0, graph['a'] + offset, -1).astype(np.int32))
        b.append(np.where(graph['b'] >= 0, graph['b'] + offset, -1).astype(np.int32))
        labels.append(graph['which'] + template_offset)
        templates.append(graph['templates'])
        phases.extend(graph['phases'])
        roots.append(graph['root'] + offset)
        offset += len(graph['a'])
        template_offset += len(graph['templates'])
        events.update(graph['execution_events'])
    root_a, root_b = [], []
    root = roots[0]
    for value in roots[1:]:
        root_a.append(root)
        root_b.append(value)
        root = offset + len(root_a) - 1
    return {'a': np.concatenate(a + [np.array(root_a, dtype=np.int32)]),
            'b': np.concatenate(b + [np.array(root_b, dtype=np.int32)]),
            'which': np.concatenate(labels + [np.zeros(len(root_a), dtype=np.int32)]),
            'templates': np.vstack(templates), 'phases': phases, 'root': root,
            'execution_events': dict(events)}
