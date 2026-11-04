from __future__ import annotations
import copy
import json
import os
from collections import Counter
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from . import fa_branch_latency as base
source = base.source
ROOT = Path(__file__).resolve().parents[1]
VERSION = "fa_branch_execution_v4"
PIPES = base.PIPES
FILES = base.FILES
_ARRAY_CACHE = {}
_ARRAY_CACHE_BYTES = 0
tiler = base.tiler
geometry = base.geometry
implementation = base.implementation
cost_class = base.cost_class
matmul_cache_trace = base.matmul_cache_trace
_fake = base._fake
batch_work = base.batch_work
s1_work = base.s1_work
general_work = base.general_work
varlen_work = base.varlen_work
adapter_jobs = base.adapter_jobs
active_features = base.active_features
_stages = base._stages
varlen_l1_program = base.varlen_l1_program
parameter_names = base.parameter_names
compact_engine = base.compact_engine
compact_graph = base.compact_graph
phase_features = base.phase_features


def matmul_microtiles(p, s, mm, stage, rows, columns, heads):
    m, n, k = rows, columns if stage == 0 else p['DV'], p['D'] if stage == 0 else columns
    bm, bn, bk = (mm[name] for name in ('base_m', 'base_n', 'base_k'))
    manual = p['layout'] == 'TND' and s['l1']
    nsplit = 256 if manual and stage == 0 else 128 if manual and s['same_ab'] and stage == 1 else n
    ksplit = (192 if s['same_ab'] else 256) if manual and stage == 1 else k
    kparts = [min(bk, min(ksplit, k - outer) - inner)
              for outer in range(0, k, ksplit) for inner in range(0, min(ksplit, k - outer), bk)]
    result = []
    api = 0
    for head in range(heads):
        for start in range(0, n, nsplit):
            count = min(nsplit, n - start)
            mt, nt = source.cost.ceil_div(m, bm), source.cost.ceil_div(count, bn)
            sm, sn = max(1, mm['step_m']), max(1, mm['step_n'])
            order = mm['iterate_order']
            for major in range(0, mt if order == 1 else nt, sm if order == 1 else sn):
                for secondary in range(0, nt if order == 1 else mt, sn if order == 1 else sm):
                    for first in range(secondary, min(secondary + (sn if order == 1 else sm), nt if order == 1 else mt)):
                        for second in range(major, min(major + (sm if order == 1 else sn), mt if order == 1 else nt)):
                            mi, ni = (second, first) if order == 1 else (first, second)
                            rm, rn = min(bm, m - mi * bm), min(bn, count - ni * bn)
                            mad_m, mad_n = (bm, bn) if s.get('static_matmul') else (16 if rm == 1 else rm, rn)
                            barrier = int((mad_m // 16) * (mad_n // 16) < 10)
                            result.append({'m': rm, 'n': rn, 'k': kparts, 'barrier': barrier,
                                           'api': api if manual else 0, 'head': head})
            api += 1
    return result


def refine_matmul_job(p, s, mm, stage, rows, columns, heads, job):
    tiles = matmul_microtiles(p, s, mm, stage, rows, columns, heads)
    job['micro_tiles'] = tiles
    job['tiles'] = sum(len(t['k']) for t in tiles)
    job['output_tiles'] = job['stores'] = len(tiles)
    job['flops_m'] = sum(2 * source.cost.align(t['m']) * source.cost.align(t['n']) * source.cost.align(k) / 1e6
                         for t in tiles for k in t['k'])
    job['l0_kb'] = sum(2 * (source.cost.align(t['m']) + source.cost.align(t['n'])) * source.cost.align(k) / 1024
                      for t in tiles for k in t['k'])
    manual = p['layout'] == 'TND' and s['l1']
    job['calls'] = len({t['api'] for t in tiles}) if manual else 1
    api_steps = job['calls']
    if manual and stage == 1:
        width = 192 if s['same_ab'] else 256
        api_steps *= source.cost.ceil_div(columns, width) + 1
    job['commands'] = api_steps + job['tiles'] + sum(t['barrier'] * len(t['k']) for t in tiles)
    job['api_steps'] = api_steps
    job['m_barriers'] = sum(t['barrier'] * len(t['k']) for t in tiles)
    return job


def execution_conditions(work):
    conditions = {'matmul_api': 'batch' if work.s['family'] in ('b', 'b_drop') else
                  'manual_l1' if work.p['layout'] == 'TND' and work.s['l1'] else
                  'direct' if work.s['same_ab'] else 'kfc',
                  'batch_broadcast': str(work.p['N1'] != work.p['N2']),
                  'aic_underfilled':str(len(work.pairs)<int(work.hardware['aic_cores'])),
                  'query_footprint_fits_l2':str(2*work.p['B']*work.p['N1']*work.p['S1']*work.p['D']<=work.hardware['l2_bytes']),
                  'kv_footprint_fits_l2':str(2*work.p['B']*work.p['N2']*work.p['S2']*(work.p['D']+work.p['DV'])<=work.hardware['l2_bytes']),
                  'sequence_tail': str(bool(work.p['S1'] % work.s['s1_base'])),
                  'multiple_s2_windows': str(len(work.offsets()) > 1),
                  'single_task_per_aiv': str(max(len(tasks) for pair in work.pairs for tasks in pair) == 1),
                  'steady_pipeline_reached': str(any(len(tasks) > 2 for pair in work.pairs for tasks in pair))}
    for stage in (0, 1):
        jobs = [unit['cube'][stage] for unit in work.cache.values() if not unit.get('fake')]
        conditions[f'bmm{stage+1}_pipe_m_barrier'] = str(any(job.get('m_barriers', 0) for job in jobs))
        conditions[f'bmm{stage+1}_multiple_k'] = str(any(any(len(t['k']) > 1 for t in job['micro_tiles']) for job in jobs))
        conditions[f'bmm{stage+1}_partial_tile'] = str(any(
            t['m'] != work.mm[stage]['base_m'] or t['n'] != work.mm[stage]['base_n']
            or any(k != work.mm[stage]['base_k'] for k in t['k']) for job in jobs for t in job['micro_tiles']))
        mm = work.mm[stage]
        kt = max(len(t['k']) for job in jobs for t in job['micro_tiles'])
        for operand, step in (('a', 'step_m'), ('b', 'step_n')):
            if work.p['layout'] == 'TND' and work.s['l1']:
                conditions[f'bmm{stage+1}_{operand}_l1_fully_cached'] = 'manual_l1_input'
                continue
            depth, span = mm['depth_'+operand+'1'], kt * max(1, mm[step])
            capacity = depth if depth >= span else max(0, depth - 2)
            if depth == 2 and kt == 1:
                capacity = 0
            conditions[f'bmm{stage+1}_{operand}_l1_fully_cached'] = str(capacity >= span)
    return conditions


class Work(base.Work):
    def job(self, task, offset, client, group):
        p, s = self.p, self.s
        batch_mode = s["family"] in ("b", "b_drop")
        rows = p["S1"] if batch_mode else min(s["s1_base"], p["S1"] - task % s["s1_outer"] * s["s1_base"])
        head = 0 if batch_mode else task // s["s1_outer"] % p["N1"]
        batch = task if batch_mode else task // (s["s1_outer"] * p["N1"])
        heads, kv_heads = (p["N1"], p["N2"]) if batch_mode else (1, 1)
        window = s["s2_base"] * s["n_ratio"] if s["same_ab"] or s["family"] in ("general", "varlen") else p["S2"]
        columns = min(window, p["S2"] - offset)
        cubes, sharing = [], []
        key = batch, head // (p["N1"] // p["N2"]), offset
        cached_sharing = tuple((
            bool(s['reuse'] and client == 1 and self.previous.get((group, stage, 'pair', offset)) == key),
            bool(s['family'] == 's1' and s['l1'] and stage == 1 and self.previous.get((group, client, 'value')) == key[:2]),
            p['layout'] == 'TND' and s['l1'] and stage == 0 and offset > 0) for stage in (0, 1))
        identity = rows, columns, cached_sharing, offset == 0, offset + columns == p['S2']
        if identity in self.cache:
            for stage in (0, 1):
                self.previous[group, stage, 'pair', offset] = key
            self.previous[group, client, 'value'] = key[:2]
            return self.cache[identity]
        for stage in (0, 1):
            shared = bool(s["reuse"] and client == 1 and self.previous.get((group, stage, "pair", offset)) == key)
            carried = bool(s["family"] == "s1" and s["l1"] and stage == 1
                           and self.previous.get((group, client, "value")) == key[:2])
            q_carried = p["layout"] == "TND" and s["l1"] and stage == 0 and offset > 0
            job = source.cube_job(p, s, self.mm[stage], stage * 2, rows, columns, heads, kv_heads,
                                  shared, carried, q_carried)
            if s["same_ab"] and s["policy"] == "UNSPLITK":
                mm = self.mm[stage]
                mt = source.cost.ceil_div(rows, mm["base_m"])
                nt = source.cost.ceil_div(columns if stage == 0 else p["DV"], mm["base_n"])
                kt = source.cost.ceil_div(p["D"] if stage == 0 else columns, mm["base_k"])
                job["loads"] = source.cost.ceil_div(mt * kt, 2) + source.cost.ceil_div(nt * kt, 2)
            if p["layout"] == "TND" and s["l1"] and not s["same_ab"]:
                parts = [min(256, columns - first) for first in range(0, columns, 256)]
                job["loads"] = (0 if q_carried else 1) + len(parts) if stage == 0 else 2 * len(parts)
                job["calls"] = len(parts)
                job["commands"] = len(parts) + job["tiles"]
                if stage == 1:
                    job["k_slices"] = [source.cost.align(part) / sum(map(source.cost.align, parts)) for part in parts]
            if p["layout"] == "TND" and s["l1"] and stage == 0:
                job["output_group_tiles"] = source.cost.ceil_div(rows, self.mm[stage]["base_m"]) * source.cost.ceil_div(256, self.mm[stage]["base_n"])
            if not batch_mode and not (p["layout"] == "TND" and s["l1"]) and s["policy"] == "NORMAL":
                trace = matmul_cache_trace(p, self.mm[stage], stage, rows, columns, shared, carried, q_carried)
                job["cache_trace"] = trace
                for name in ("read_kb", "loads", "address_gap_kb"):
                    job[name] = sum(part[name] for output in trace for part in output)
            job = refine_matmul_job(p, s, self.mm[stage], stage, rows, columns, heads, job)
            cubes.append(job)
            sharing.append((shared, carried, q_carried))
            self.previous[group, stage, "pair", offset] = key
        self.previous[group, client, "value"] = key[:2]
        vectors = []
        for owned in ((source.cost.ceil_div(rows, 2), rows // 2) if s["same_ab"] else (rows,)):
            if not owned:
                vectors.append(({}, {}))
                continue
            v1, v2 = source.vector_job(p, s, owned, columns, offset == 0, offset + columns == p["S2"])
            v1["scale_k"] = v1["cast_k"]
            if batch_mode:
                v1, v2 = ({key: value * heads for key, value in v.items()} for v in (v1, v2))
            vectors.append((v1, v2))
        identity = rows, columns, tuple(sharing), offset == 0, offset + columns == p["S2"]
        if identity not in self.cache:
            self.cache[identity] = {"cube": cubes, "vector": vectors, "rows": rows, "columns": columns,
                                    "fake": False, "heads": heads}
        return self.cache[identity]


def workload(inputs, hardware, calibration_geometry=None):
    work = Work(inputs, hardware, calibration_geometry)
    name = implementation(work.p, work.s)
    work.pairs = (batch_work(work) if name == "b" else s1_work(work) if name == "s1"
                  else general_work(work) if name == "general" else varlen_work(work))
    if name == "same_ab":
        work.pairs = [[tasks] for pair in work.pairs for tasks in pair]
    return work


def batch_program(tasks):
    return _stages(len(tasks))


def s1_program(tasks):
    return _stages(len(tasks))


def general_program(tasks):
    return _stages(len(tasks))


def varlen_gm_program(tasks):
    return _stages(len(tasks))


def release_graph_pages(graph):
    import mmap
    for name in ('a','b','which','templates'):
        if isinstance(graph[name], CompressedGraphArray):
            graph[name]._pin = None
        mapping=getattr(graph[name],'_mmap',None)
        if mapping is not None and hasattr(mapping,'madvise'):
            mapping.madvise(mmap.MADV_DONTNEED)


def graph_time(graph,rates,gradient=False):
    result=source.graph_time(graph,rates,gradient)
    release_graph_pages(graph)
    return result


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


class CompressedGraphArray:
    def __init__(self, path, description):
        self.path, self.description, self._pin = str(path), description, None
        self.shape = tuple(description['shape'])
        self.dtype = np.dtype(description['dtype'])

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        self._pin = compressed_array(self.path, self.description)
        return np.asarray(self._pin, dtype=dtype)

    @property
    def ctypes(self):
        self._pin = compressed_array(self.path, self.description)
        return self._pin.ctypes

    def __matmul__(self, other):
        return np.asarray(self) @ other

    def __ne__(self, other):
        return np.asarray(self) != other


def compressed_array(path, description):
    import zlib
    global _ARRAY_CACHE_BYTES
    identity = path, description['offset']
    if identity in _ARRAY_CACHE:
        value = _ARRAY_CACHE.pop(identity)
        _ARRAY_CACHE[identity] = value
        return value
    with open(path, 'rb') as stream:
        stream.seek(description['offset'])
        packed = stream.read(description['compressed_bytes'])
    value = np.frombuffer(zlib.decompress(packed), dtype=description['dtype']).reshape(tuple(description['shape']))
    if value.nbytes != description['raw_bytes']:
        raise ValueError('compressed CPU graph cache is incomplete')
    limit = 320 * 1024 * 1024
    while _ARRAY_CACHE and _ARRAY_CACHE_BYTES + value.nbytes > limit:
        first = next(iter(_ARRAY_CACHE))
        _ARRAY_CACHE_BYTES -= _ARRAY_CACHE.pop(first).nbytes
    _ARRAY_CACHE[identity] = value
    _ARRAY_CACHE_BYTES += value.nbytes
    return value


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


def primitive_class(pipe, key):
    impl,dtype,layout,path,fmt,policy=key.split('|')
    vector_function={'varlen_gm':'general','varlen_l1':'general','varlen_same_ab':'same_ab'}.get(impl,impl)
    api = 'batch' if impl == 'b' else 'manual_l1' if impl == 'varlen_l1' or impl == 'varlen_same_ab' and path == 'l1_carry' else 'direct' if impl in ('same_ab', 'varlen_same_ab') else 'kfc'
    if pipe in ('cube','mte1'):
        return (pipe,dtype,policy) if policy == 'STATIC' else (pipe,dtype,policy,api)
    if pipe=='fixpipe':
        return (pipe,policy) if policy == 'STATIC' else (pipe,policy,api)
    if pipe=='vector':
        return pipe,vector_function,dtype
    if pipe=='cscalar':
        api='batch' if impl=='b' else 'direct_same_ab' if impl in ('same_ab','varlen_same_ab') else 'manual_l1' if impl=='varlen_l1' else 'normal'
        return pipe,api,policy,fmt
    if pipe=='vscalar':
        return pipe,vector_function,'direct' if impl in ('same_ab','varlen_same_ab','varlen_l1') else 'kfc',policy
    api='batch' if impl=='b' else 'normal' if impl in ('general','s1','varlen_gm') else impl
    return pipe,api,layout,policy


def primitive_pools(pipe, key):
    exact = primitive_class(pipe, key)
    if pipe in ('cube', 'mte1') and len(exact) == 4 or pipe == 'fixpipe' and len(exact) == 3:
        return exact, exact[:-1]
    return (exact,)


def graph_entry(model, key, conditions=None):
    if key in model['classes']:
        entry=model['classes'][key]
        refinement=entry.get('source_refinement')
        if refinement and conditions is not None:
            child=refinement['children'].get(conditions.get(refinement['condition']))
            if child is not None:return child
        return entry
    pipes={}
    for pipe,(_,fields) in PIPES.items():
        parent = None
        candidates = primitive_pools(pipe, key)
        for candidate in candidates:
            value = model['primitive_priors'].get(json.dumps(candidate))
            if value is not None and (value['calibration_input_count'] >= len(value['observed_features']) or candidate == candidates[-1]):
                parent = value
                break
        if parent is None:
            raise ValueError('unmeasured source operation class: '+str(primitive_class(pipe,key)))
        pipes[pipe]={'coefficients':dict(zip(parent['names'],parent['rates'])),'observed_features':parent['observed_features'],
                     'rank':parent['rank'],'terms':parent['terms'],'samples':parent['calibration_input_count']}
    return {'pipes':pipes,'rates':[pipes[pipe]['coefficients'].get(name,0) for pipe,name in parameter_names()],
            'calibration_input_count':0,'cost_scope':'source-matched primitives, no direct execution-class latency samples'}


def predict_work(work, model, graph=None):
    key=cost_class(work.p,work.s);entry=graph_entry(model,key,execution_conditions(work))
    for pipe,values in active_features(work).items():
        missing=[n for n,val in values.items() if val>0 and n not in entry['pipes'][pipe]['observed_features']]
        if missing:raise ValueError('unmeasured primitive: '+key+'/'+pipe+'/'+','.join(missing))
    graph=branch_graph(work) if graph is None else graph
    predicted,gradient=graph_time(graph,entry['rates'],True)
    critical=source.critical_phases(graph,entry['rates'])
    demands={f'{phase}.{pipe}':sum(value*entry['pipes'][pipe]['coefficients'].get(n,0) for n,value in values.items())
             for phase,pipes in phase_features(work).items() for pipe,values in pipes.items()}
    assessment = refinement_status(work,model)
    return {'predicted_us':predicted,'execution_events_scope':'distinct representative worker timelines; identical workers share a graph for maximum completion',
            'whole_device_mmad_count':active_features(work)['cube']['tiles'],'out_of_calibration_range':list(assessment['micro_context_extrapolation']) + [item['condition']+'='+item['value'] for item in assessment['unseen_source_conditions']], 'critical_path_phase_us':critical,'critical_path_pipe_us':critical_pipe_times(gradient,entry['rates']),'phase_pipe_active_us':demands,
            'phase_pipe_active_scope':'whole-device source-operation service sums; not isolated stage wall times; Scalar excludes modeled polling',
            'adapter_us':critical.get('drop_adapter',0),'startup_us':critical.get('startup',0),
            'implementation':implementation(work.p,work.s),'cost_class':key,'schedule':work.s,'matmul':work.mm,
            'execution_events':graph['execution_events'],'max_resource_bytes':{},
            'calibration_sdk_api_sha256':model.get('sdk_api_sha256'),'prediction_sdk_api_sha256':tiler()[1],
            'sdk_api_matches_calibration':model.get('sdk_api_sha256')==tiler()[1],
            'unidentified_pipe_terms':[p for p,x in entry['pipes'].items() if x['rank']<x['terms']],
            'direct_class_calibration_samples':entry['calibration_input_count'],
            'refinement':assessment,
            'target':'kernel Task Duration(us); source-directed FIFO and buffer hazards; training-only calibrated operation costs'}


def predict(inputs, model):
    if model.get('version')!=VERSION:
        raise ValueError('branch model and execution code differ')
    return predict_work(workload(inputs,model['hardware']),model)


def refinement_status(work, model):
    key = cost_class(work.p, work.s)
    conditions = execution_conditions(work)
    entry = graph_entry(model,key,conditions) if key in model.get('classes', {}) else None
    support = (entry or {}).get('source_condition_support', {})
    unseen = [{'condition': name, 'value': value} for name, value in conditions.items()
              if value not in support.get(name, {})]
    matched = [row for row in model.get('refinement_candidates', [])
               if row['cost_class'] == key and conditions.get(row['condition']) == row['value']]
    warnings = [row for row in matched if row.get('systematic_bias_evidence') and row['median_ape_percent'] > 10]
    suggestive = [row for row in matched if row['median_ape_percent'] > 10
                  and row['stable_outside_repeat_interval_count'] > 0 and not row.get('systematic_bias_evidence')]
    context = micro_context(active_features(work))
    ranges = (entry or {}).get('micro_context_ranges', {})
    extrapolated = {name: {'input': value, 'calibration_range': ranges.get(name)} for name, value in context.items()
                    if name not in ranges or value < ranges[name][0] - max(1e-12, abs(ranges[name][0])*1e-10)
                    or value > ranges[name][1] + max(1e-12, abs(ranges[name][1])*1e-10)}
    def compact(rows):
        keys = ('condition','value','sequence_groups','median_ape_percent','median_signed_error_percent','sign_test_fdr_q','suggested_action')
        return [{name: row[name] for name in keys} for row in rows]
    attribution = {pipe:{'rank':stats['rank'],'terms':stats['terms']} for pipe,stats in (entry or {}).get('local_identifiability',{}).items() if stats['rank']<stats['terms']}
    return {'source_conditions': conditions, 'unseen_source_conditions': unseen,
            'costs_not_individually_identified':attribution,
            'cost_attribution_scope':'micro cost library; unresolved terms need independent information before further separation; validated combined costs still predict kernel latency',
            'micro_context_extrapolation': extrapolated,
            'known_systematic_bias': compact(warnings), 'suggestive_refinement_candidates': compact(suggestive),
            'status': 'unmeasured_source_condition' if unseen else 'source_condition_requires_refinement_study' if warnings else 'micro_cost_context_extrapolation' if extrapolated else 'more_independent_shapes_needed_for_candidate' if suggestive else 'no_confirmed_refinement_trigger',
            'accuracy_guaranteed': False,
            'split_acceptance': 'source branch or resource dependency explains it; stable group residuals exceed repeat variation; new costs are identifiable; nested sequence-group validation improves without family regression',
            'runtime_inputs': 'operator inputs, hardware and frozen calibration metadata only; no measured latency needed'}


def micro_context(features):
    def ratio(pipe, numerator, denominator):
        values = features[pipe]
        return float(values.get(numerator, 0)) / max(float(values.get(denominator, 0)), 1e-30)
    return {'flops_m_per_mmad': ratio('cube', 'flops_m', 'tiles'),
            'outputs_per_mmad': ratio('cube', 'output_tiles', 'tiles'),
            'l0_kb_per_mmad': ratio('mte1', 'l0_kb', 'tiles'),
            'gm_read_kb_per_load': ratio('mte2', 'read_kb', 'loads'),
            'gm_gap_per_read': ratio('mte2', 'address_gap_kb', 'read_kb'),
            'nd_kb_per_store': ratio('fixpipe', 'nd_write_kb', 'stores'),
            'nz_kb_per_store': ratio('fixpipe', 'nz_write_kb', 'stores'),
            'softmax_k_per_chunk': ratio('vector', 'softmax_k', 'chunks'),
            'scalar_commands_per_aic': ratio('cscalar', 'commands', 'workers'),
            'vector_chunks_per_aiv': ratio('vscalar', 'chunks', 'workers'),
            'active_aic': float(features['cscalar'].get('workers', 0))}


def critical_pipe_times(gradient, rates):
    result = dict.fromkeys(PIPES, 0.0)
    for (pipe, _), derivative, rate in zip(parameter_names(), gradient, rates):
        result[pipe] += float(derivative * rate)
    return result


def predict_selected(inputs, model):
    p = source.cost.normalize(inputs)
    schedule = source.schedule(p, model['hardware'])
    if p['layout'] == 'TND' and not schedule['same_ab']:
        schedule['l1'] = schedule['s1_base'] <= 128 and max(p['D'],p['DV']) <= 256
    impl = implementation(p,schedule)
    decision = model['selection'][impl]
    if decision['approved']:
        if impl == 'varlen_l1' and model.get('varlen_l1_transfer'):
            from .fa_latency_transfer import predict as predict_transfer
            result = predict_transfer(p,model['refined'],model['varlen_l1_transfer'])
        else:
            result = predict(p,model['refined'])
    else:
        result = base.predict(p,model['baseline'])
        work = workload(p,model['hardware'])
        result['refinement'] = refinement_status(work,model['refined'])
    result['selected_cost_model'] = 'micro_costs' if decision['approved'] else 'baseline'
    result['model_selection_evidence'] = decision
    result['accuracy_guaranteed'] = False
    return result
