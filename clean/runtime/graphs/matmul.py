from .. import inputs as cost


def cube(graph, owner, unit, stage, launch, slot_state):
    phase = 'bmm1' if stage == 0 else 'bmm2'
    job, mm = unit['cube'][stage // 2], graph.work.mm[stage // 2]
    if unit.get('fake'):
        graph.events['fake_matmul_messages'] += 1
        return graph.operation('cscalar', 0, job, phase, (launch,))
    tiles = job['micro_tiles']
    a_count = max(1, mm['depth_a1'] // max(1, mm['step_m']))
    b_count = max(1, mm['depth_b1'] // max(1, mm['step_n']))
    ab_count = max(1, min(mm['db_l0a'], mm['db_l0b']))
    c_count = max(1, min(mm['db_l0c'], graph.work.hardware.get('l0c_bytes', 131072) // (mm['base_m'] * mm['base_n'] * 4)))
    last_api = graph.node((launch, graph.tails.get(('mte1', 0), -1)))
    prior_fix = graph.tails.get(('fixpipe', 0), -1)
    operand_data, fixed, previous_api = {}, [], None
    batch_mode = graph.work.s['family'] in ('b', 'b_drop')
    bulk = None
    if batch_mode:
        issued = graph.operation('cscalar', 0, {'commands': job['api_steps']}, phase, (launch,))
        bulk = graph.operation('mte2', 0, job, phase, (issued, last_api))
    for output, tile in enumerate(tiles):
        api_start = tile['api'] != previous_api
        if api_start:
            if fixed:
                last_api = fixed[-1]
            if not batch_mode:
                graph.operation('cscalar', 0, {'commands': job['api_steps'] / job['calls']}, phase, (launch, last_api))
            previous_api = tile['api']
        partial = launch
        for k, length in enumerate(tile['k']):
            issued = graph.operation('cscalar', 0, {'commands': 1}, phase, (launch,))
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
                loaded = graph.node(dependencies + [bulk])
            elif not transfer or values['loads']:
                fraction = cost.align(length) / sum(cost.align(v) for v in tile['k'])
                loaded = graph.operation('mte2', 0, values, phase, dependencies,
                                        divisor=1 if transfer else len(tiles) / fraction)
            else:
                loaded = graph.node(dependencies)
            for operand, slot in (('a', a_slot), ('b', b_slot)):
                if not transfer or values[operand + '_miss']:
                    operand_data[slot] = loaded
            m, n, kval = map(cost.align, (tile['m'], tile['n'], length))
            local = {'l0_kb': 2 * (m + n) * kval / 1024, 'tiles': 1}
            moved = graph.operation('mte1', 0, local, phase, (loaded, slot_state.get(l0_slot, -1)))
            slot_state[a_slot] = slot_state[b_slot] = moved
            arithmetic = {'flops_m': 2 * m * n * kval / 1e6, 'tiles': 1, 'output_tiles': int(k == 0)}
            computed = graph.operation('cube', 0, arithmetic, phase,
                                      (moved, partial, prior_fix if output == 0 and k == 0 else -1,
                                       slot_state.get(c_slot, -1) if k == 0 else -1))
            slot_state[l0_slot] = computed
            partial = computed
            if tile['barrier']:
                graph.operation('cscalar', 0, {'commands': 1}, phase, (launch,))
                graph.events['pipe_m_barriers'] += 1
        if batch_mode:
            graph.tails['mte1',0] = graph.node((graph.tails.get(('mte1',0),-1),partial))
            graph.events['batch_mte1_fences'] += 1
        write = tile['m'] * cost.align(tile['n']) * 4 / 1024
        output_job = {'nz_write_kb': write if job.get('nz_write_kb') else 0,
                      'nd_write_kb': write if not job.get('nz_write_kb') else 0,
                      'stores': 1, 'calls': int(api_start)}
        stored = graph.operation('fixpipe', 0, output_job, phase, (partial,))
        slot_state[c_slot] = stored
        fixed.append(stored)
    graph.events['matmul_calls'] += int(job['api_steps'])
    graph.events['k_tiles'] += sum(len(tile['k']) for tile in tiles)
    graph.events['batch_bulk_copies'] += int(batch_mode)
    return fixed[-1]
