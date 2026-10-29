from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import collect_fa_stages as stages

ROOT = Path(__file__).resolve().parent
CASE, SHAPE, _ = stages.CASES[-1]
INPUTS = dict(zip(stages.cost.INPUT_FIELDS, SHAPE))
WORK = ROOT / 'out/fa_stage_profile/instructions'
OUTPUT = ROOT / 'output/fa_stage_profile/instructions.csv'
GIB = 1024 ** 3
FIELDS = ('record_type', 'core', 'pipe', 'pc', 'instruction', 'call_count', 'cycles', 'reported_time_us',
          'start_raw', 'duration_raw', 'detail', 'extra_json')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def available_memory():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    limits = [int(values['MemAvailable'].split()[0]) * 1024]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if not controllers:
            base, maximum, current = Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            base, maximum, current = Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        folder = base / relative.lstrip('/')
        while folder == base or base in folder.parents:
            if (folder / maximum).is_file() and (folder / current).is_file():
                value = (folder / maximum).read_text().strip()
                if value != 'max':
                    limits.append(max(0, int(value) - int((folder / current).read_text())))
            if folder == base:
                break
            folder = folder.parent
    return min(limits)


def references(path):
    if not path.is_file():
        raise FileNotFoundError(f'existing NPU measurements are missing: {path}')
    with path.open(newline='') as source:
        rows = [row for row in csv.DictReader(source) if row.get('case') == CASE and row.get('status') == 'ok']
    rows = [row for row in rows if all(str(row.get(name)) == str(value) for name, value in INPUTS.items())]
    if not rows:
        raise RuntimeError(f'no successful measurements of {CASE} in {path}')
    selected = next((row for row in rows if row['phase'] == 'control'), rows[0])
    required = ('core_num', 'total_tasks', 'split_factor', 's1_base', 's2_base', 'n_ratio', 'aic_cores', 'aiv_cores')
    geometry = {name: int(selected[name]) for name in required}
    if int(selected['tiling_key']) != stages.KEY or selected.get('soc') != 'Ascend910B3':
        raise RuntimeError('this instruction collection requires the existing Ascend910B3 SameAB FP16 measurements')
    return rows, geometry


def cached_kernel():
    failures = []
    for kind, base in (('instrumented_control', ROOT / 'out/fa_stage_profile/cache'),
                       ('native', Path(os.environ.get('FA_CACHE_ROOT', str(ROOT / 'out/fa_cache'))).resolve())):
        try:
            binary, vendor = stages.cache_location(base)
            folder = vendor / 'op_impl/ai_core/tbe/kernel' / os.environ.get('FA_SOC_UNIT', 'ascend910b') / 'flash_attention_score'
            objects = []
            for path in folder.glob('*.json'):
                data = json.loads(path.read_text())
                if any(int(item.get('tilingKey', -1)) == stages.KEY for item in data.get('kernelList', [])):
                    objects.append(path.with_name(data['binFileName'] + data.get('binFileSuffix', '.o')))
            if len(objects) != 1 or not objects[0].is_file():
                raise RuntimeError('the cached kernel binary is ambiguous or missing')
            return kind, binary, vendor, objects[0]
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            failures.append(str(error))
    raise RuntimeError('no existing kernel can be reused; no build was started: ' + ' | '.join(failures))


def expected_matrix_count(rows, geometry):
    count = 0
    selected = [row for row in rows if row['phase'] in ('bmm1', 'bmm2')]
    if {(row['phase'], row['key_window']) for row in selected} != {
            ('bmm1', 'first'), ('bmm1', 'last'), ('bmm2', 'first'), ('bmm2', 'last')} or len(selected) != 4:
        raise RuntimeError('the four existing BMM window measurements are required for trace completeness checking')
    for row in selected:
        metrics = json.loads(row.get('counter_diagnostics_json') or '{}').get('ArithmeticUtilization/aic', {})
        instructions = metrics.get('aic_cube_total_instr_number', {}).get('core_sum')
        if instructions is None or str(row.get('gating_verified')).lower() != 'true':
            raise RuntimeError('verified per-stage Cube instruction counts are missing from the existing measurement CSV')
        count += float(instructions) / float(row['aic_active_cores']) * geometry['split_factor']
    if count <= 0 or not count.is_integer():
        raise RuntimeError('the existing instruction counts do not identify a complete per-core execution')
    return int(count)


def private_simulator(sdk, geometry):
    source = sdk / 'tools/simulator/Ascend910B3/lib'
    for name in ('libruntime_camodel.so', 'libnpu_drv_camodel.so', 'libmodel_top.so'):
        if not (source / name).is_file():
            raise FileNotFoundError(f'official simulator component is missing: {source / name}')
    target = WORK / 'model/tools/simulator/Ascend910B3/lib'
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        destination = target / path.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if path.suffix == '.json' or path.name in ('libmodel_top.so', 'libruntime_camodel.so', 'libnpu_drv_camodel.so'):
            shutil.copy2(path, destination)
        elif path.is_file():
            destination.symlink_to(path.resolve())
    config = target / 'config_stars.json'
    if config.is_file():
        data = json.loads(config.read_text())
        if 'model_top' not in data:
            raise RuntimeError(f'unknown simulator topology configuration: {config}')
        data['model_top'].update(num_aic=geometry['aic_cores'], num_aiv=geometry['aiv_cores'])
        if 'pem' in data:
            data['pem'].update(parsim=0, parsim_thd_limit=1)
        temporary = config.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(config)
    (target / 'libruntime.so').unlink(missing_ok=True)
    (target / 'libruntime.so').symlink_to(target / 'libruntime_camodel.so')
    return source, target


def monitor(argv, env, timeout):
    log = WORK / 'profile.log'
    start = time.monotonic()
    with log.open('w') as output:
        process = subprocess.Popen(argv, cwd=WORK, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            last_notice = 0
            while process.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    raise TimeoutError(f'simulator exceeded {timeout}s; partial files retained in {WORK}')
                snapshot = subprocess.run(['ps', '-eo', 'pid=,ppid=,pgid=,rss='], capture_output=True, text=True, timeout=10)
                if snapshot.returncode:
                    raise RuntimeError('cannot monitor simulator memory usage')
                processes = [tuple(map(int, line.split())) for line in snapshot.stdout.splitlines() if len(line.split()) == 4]
                descendants = {process.pid}
                while True:
                    expanded = descendants | {pid for pid, parent, group, _ in processes if parent in descendants or group == process.pid}
                    if expanded == descendants:
                        break
                    descendants = expanded
                rss = sum(rss * 1024 for pid, _, _, rss in processes if pid in descendants)
                if rss > 20 * GIB or available_memory() < 2 * GIB:
                    raise MemoryError('simulator memory limit reached; process group stopped and partial files retained')
                if elapsed - last_notice >= 30:
                    print(f'SIMULATING: elapsed={elapsed:.0f}s RSS={rss / GIB:.2f}GiB; log={log}', flush=True)
                    last_notice = elapsed
                if elapsed > 10 and int(elapsed) % 5 == 0:
                    used = sum(path.stat().st_size for path in WORK.rglob('*') if path.is_file())
                    if used > 2 * GIB:
                        raise OSError('simulator trace exceeded 2GiB; partial files retained')
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f'official simulator exited {process.returncode}; log={log}')
        except BaseException:
            for pid in locals().get('descendants', set()) - {process.pid}:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise


def trace_events(path):
    decoder = json.JSONDecoder()
    with path.open() as source:
        buffer = ''
        while '"traceEvents"' not in buffer:
            chunk = source.read(65536)
            if not chunk:
                raise ValueError(f'no traceEvents array: {path}')
            buffer = (buffer + chunk)[-131072:]
        buffer = buffer[buffer.index('"traceEvents"') + len('"traceEvents"'):]
        while '[' not in buffer:
            chunk = source.read(65536)
            if not chunk:
                raise ValueError(f'incomplete trace header: {path}')
            buffer += chunk
        buffer = buffer[buffer.index('[') + 1:]
        while True:
            buffer = buffer.lstrip(' \r\n\t,')
            if buffer.startswith(']'):
                return
            try:
                event, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = source.read(65536)
                if not chunk:
                    raise ValueError(f'incomplete instruction trace: {path}')
                buffer += chunk
                continue
            yield event
            buffer = buffer[end:]


def core_name(path):
    match = re.search(r'core0[._](?:cubecore|veccore)\d+', str(path))
    return match[0] if match else None


def validate_result(geometry):
    route, effective = stages.read_trace(WORK)
    hardware = json.loads((WORK / 'hardware.json').read_text())
    fields = {'core_num': 'FA_CORE_NUM', 'total_tasks': 'FA_TOTAL_TASKS', 'split_factor': 'FA_SPLIT_FACTOR',
              's1_base': 'FA_S1_BASE', 's2_base': 'FA_S2_BASE', 'n_ratio': 'FA_N_RATIO'}
    if route != (95, stages.KEY) or effective is None or effective.get('FA_BMM_PRESET') != 0 or effective.get('FA_STAGE_PROFILE_ID', 0) != 0:
        raise RuntimeError('simulator Host route differs from the selected existing kernel')
    if any(effective.get(field) != geometry[name] for name, field in fields.items()) or any(
            hardware.get(name) != geometry[name] for name in ('aic_cores', 'aiv_cores')):
        raise RuntimeError(f'simulator topology or schedule differs from NPU reference: hardware={hardware}, effective={effective}')
    return {'numerical_check': stages.check_output(INPUTS, WORK / 'output/output.bin'),
            'hardware': hardware, 'effective_tiling': effective}


def export_report(metadata, reference_rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.tmp')
    counters = collections.Counter()
    trace_cores, summary_cores = set(), set()
    attempt = WORK / 'run_manifest.json'
    verified = attempt.is_file() and all(metadata.get(name) == value for name, value in json.loads(attempt.read_text()).items())
    profile = WORK / 'profile' if verified else WORK / 'no_matching_trace'
    metadata['trace_matches_current_run'] = verified
    predictions = json.loads((ROOT / 'fa_instruction_predictions.json').read_text())
    if predictions['inputs'] != INPUTS or predictions['case'] != CASE:
        raise ValueError('frozen instruction-study predictions do not match the selected inputs')
    metadata['frozen_forecast_model_sha256'] = predictions['model_sha256']
    metadata['forecast_evaluation_is_new_blind_test'] = predictions['is_new_blind_test']
    with temporary.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerow({'record_type': 'metadata', 'extra_json': json.dumps(metadata, separators=(',', ':'))})
        for prediction in predictions['rows']:
            writer.writerow({'record_type': 'model_prediction', 'pipe': prediction['pipe'],
                             'extra_json': json.dumps(prediction, separators=(',', ':'))})
        for row in reference_rows:
            selected = {key: value for key, value in row.items() if key in ('case', 'phase', 'key_window', 'query_task',
                'status', 'physical_npu_id', 'soc', 'tiling_key', 'protocol_sha256', 'compiled_kernel_source_sha256',
                'gating_verified', 'stage_selector', 'task_duration_us', 'task_duration_samples_us', 'freq_min_mhz', 'freq_max_mhz') or
                key.endswith(('_sum_us', '_sum_samples_us', '_max_us', '_max_samples_us', '_active_cores')) or key in INPUTS}
            writer.writerow({'record_type': 'npu_reference', 'extra_json': json.dumps(selected, separators=(',', ':'))})
        tables = collections.defaultdict(list)
        for path in profile.rglob('*.csv'):
            core = core_name(path)
            if core is None or re.search(r'_instr_exe(?:_prof)?(?:_\d+)?\.csv$', path.name) is None:
                continue
            tables[core].append(path)
        for core, candidates in sorted(tables.items()):
            preferred = [path for path in candidates if '_prof' not in path.name]
            candidates = preferred or candidates
            if len({digest(path) for path in candidates}) != 1:
                raise ValueError(f'multiple different instruction tables for {core}; refusing to merge launches')
            path = sorted(candidates)[0]
            with path.open(newline='', encoding='utf-8-sig') as source:
                for row in csv.DictReader(source):
                    normalized = {re.sub(r'[^a-z0-9]', '', key.lower()): value for key, value in row.items() if key}
                    pc = normalized.get('addr', normalized.get('pcaddress', normalized.get('pcaddr', normalized.get('pc', ''))))
                    name = normalized.get('instr', normalized.get('instrname', normalized.get('instruction', normalized.get('name', ''))))
                    if not pc or not name:
                        raise ValueError(f'instruction table has no PC or instruction name: {path}')
                    if (stages.finite(normalized.get('callcount')) or 0) <= 0 or stages.finite(normalized.get('cycles', normalized.get('cycle'))) is None:
                        raise ValueError(f'instruction table has no numeric call counts or cycles: {path}')
                    writer.writerow({'record_type': 'instruction', 'core': core, 'pc': pc, 'instruction': name,
                        'pipe': normalized.get('pipe', ''), 'call_count': normalized.get('callcount', ''),
                        'cycles': normalized.get('cycles', normalized.get('cycle', '')),
                        'reported_time_us': normalized.get('runningtimeus', ''),
                        'detail': normalized.get('detail', normalized.get('instrdetail', normalized.get('params', ''))),
                        'extra_json': json.dumps({key: value for key, value in normalized.items() if key not in
                            ('addr', 'pcaddress', 'pcaddr', 'pc', 'instr', 'instrname', 'instruction', 'name', 'pipe',
                             'callcount', 'cycles', 'cycle', 'runningtimeus', 'detail', 'instrdetail', 'params')}, separators=(',', ':'))})
                    counters['instruction'] += 1
                    summary_cores.add(core)
        traces = [path for path in profile.rglob('trace.json') if core_name(path)]
        if len({core_name(path) for path in traces}) != len(traces):
            raise ValueError('multiple traces for one core; refusing to merge different launches')
        for path in sorted(traces):
            core = core_name(path)
            with path.open() as source:
                header = source.read(65536)
            unit = re.search(r'"displayTimeUnit"\s*:\s*"([^"]+)"', header)
            writer.writerow({'record_type': 'trace_header', 'core': core, 'extra_json': json.dumps({
                'displayTimeUnit': unit[1] if unit else None, 'time_conversion_applied': False,
                'trace_bytes': path.stat().st_size}, separators=(',', ':'))})
            intervals = collections.defaultdict(list)
            for event in trace_events(path):
                if event.get('ph') != 'X':
                    continue
                start, duration = stages.finite(event.get('ts')), stages.finite(event.get('dur'))
                if start is None or duration is None or duration < 0:
                    raise ValueError(f'invalid instruction timestamp: {path}')
                name, pipe, args = str(event.get('name', '')), str(event.get('tid', '')), event.get('args') or {}
                if not isinstance(args, dict):
                    raise ValueError(f'invalid instruction arguments: {path}')
                pc = args.get('addr', args.get('pc', args.get('PC', '')))
                intervals[pipe].append((start, start + duration))
                counters['dynamic_events'] += 1
                keep = re.search(r'wait|set_?flag|barrier|sync|mad|fixpipe|fixp', name, re.I) or re.search(r'CUBE|MATRIX|FIX|MTE', pipe, re.I)
                if keep:
                    writer.writerow({'record_type': 'instruction_event', 'core': core, 'pipe': pipe, 'pc': pc,
                        'instruction': name, 'start_raw': start, 'duration_raw': duration,
                        'detail': args.get('detail', ''), 'extra_json': json.dumps(args, separators=(',', ':'))})
                    counters['exported_dynamic_events'] += 1
                if re.search(r'(^|_)m?mad', name, re.I):
                    counters['matrix_events'] += 1
                    if pc:
                        counters['matrix_events_with_pc'] += 1
                    if args.get('detail') or set(args) - {'addr', 'pc', 'PC', 'code'}:
                        counters['matrix_events_with_detail'] += 1
                if re.search(r'wait|barrier|set_?flag|sync', name, re.I):
                    counters['sync_events'] += 1
            if not intervals:
                raise ValueError(f'empty instruction trace: {path}')
            for pipe, spans in intervals.items():
                merged = []
                for start, end in sorted(spans):
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])
                for start, end in merged:
                    writer.writerow({'record_type': 'busy_interval', 'core': core, 'pipe': pipe,
                                     'start_raw': start, 'duration_raw': end - start})
                    counters['busy_intervals'] += 1
            trace_cores.add(core)
        expected = {'core0.cubecore0', 'core0.veccore0', 'core0.veccore1'}
        canonical = lambda values: {value.replace('core0_', 'core0.') for value in values}
        counters['complete'] = int(expected <= canonical(trace_cores) and expected <= canonical(summary_cores)
                                   and counters['matrix_events'] == metadata.get('expected_matrix_events')
                                   and counters['matrix_events_with_pc'] == counters['matrix_events']
                                   and counters['matrix_events_with_detail'] == counters['matrix_events']
                                   and counters['sync_events'] > 0)
        writer.writerow({'record_type': 'collection_summary', 'extra_json': json.dumps(dict(counters), separators=(',', ':'))})
    temporary.replace(OUTPUT)
    return counters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, default=ROOT / 'output/fa_stage_profile/stages.csv')
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata = {'status': 'incomplete', 'case': CASE, 'inputs': INPUTS, 'tiling_key': stages.KEY,
        'collection_kind': 'official_instruction_simulator', 'is_npu_measurement': False,
        'trace_timestamp_unit': 'raw_simulator_units_not_assumed_microseconds',
        'dynamic_export': 'all engine busy intervals and compute/transfer/synchronization events; other scalar/vector events retained as per-PC summaries',
        'timing_validation_scope': 'simulator model cycles and instruction flow; not NPU ground truth or a validated global-memory contention model',
        'phase_comparison_scope': 'simulation uses the unmarked control; existing gated NPU phases contain instrumentation barriers, so segment comparisons are diagnostic',
        'code_mapping_requires_debug_info': True, 'cpu_cores': 1, 'timeout_seconds': args.timeout}
    reference_rows = []
    try:
        reference_rows, geometry = references(args.reference.resolve())
        metadata['expected_matrix_events'] = expected_matrix_count(reference_rows, geometry)
        kind, binary, vendor, obj = cached_kernel()
        sdk = Path(os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')).resolve()
        simulator = sdk / 'tools/simulator/Ascend910B3/lib'
        profiler = sdk / 'bin/msprof'
        if not profiler.is_file() or not (simulator / 'libruntime_camodel.so').is_file():
            raise FileNotFoundError(f'CANN msprof or Ascend910B3 simulator is missing under {sdk}')
        signature = {'schema': 1, 'inputs': INPUTS, 'geometry': geometry, 'cache_kind': kind,
            'kernel_sha256': digest(obj), 'launcher_sha256': digest(binary), 'profiler_sha256': digest(profiler),
            'simulator_sha256': digest(simulator / 'libruntime_camodel.so'),
            'simulator_config_sha256': {path.name: digest(path) for path in simulator.glob('*.json')},
            'model_top_sha256': digest(simulator / 'libmodel_top.so'), 'pem_model_sha256': digest(simulator / 'libpem_davinci.so')}
        marker = WORK / 'complete.json'
        metadata.update(signature, sdk=str(sdk), kernel_binary=str(obj), reference_file=str(args.reference.resolve()),
                        reference_binary_identity='source/build context only; old NPU CSV has no binary SHA256')
        certificate = json.loads(marker.read_text()) if marker.is_file() else {}
        if certificate.get('signature') == signature:
            metadata.update(certificate['metadata'])
            metadata['cache_hit'] = True
            metadata['status'] = 'complete'
            counts = export_report(metadata, reference_rows)
            if not counts['complete']:
                raise RuntimeError('cached trace is incomplete; original files retained')
            print(f'INSTRUCTION CACHE HIT: no compilation, simulation or NPU run\nCSV={OUTPUT}', flush=True)
            return
        attempt = WORK / 'run_manifest.json'
        if attempt.is_file() and json.loads(attempt.read_text()) == signature and (WORK / 'output/output.bin').is_file():
            try:
                metadata.update(validate_result(geometry), status='complete', recovered_existing_trace=True)
                counts = export_report(metadata, reference_rows)
                if counts['complete']:
                    marker.write_text(json.dumps({'signature': signature, 'metadata': metadata}, sort_keys=True) + '\n')
                    (WORK / 'output/output.bin').unlink(missing_ok=True)
                    print(f'RECOVERED: existing complete instruction trace; no simulation or NPU run\nCSV={OUTPUT}', flush=True)
                    return
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                metadata['previous_partial_trace_error'] = str(error)
            metadata['status'] = 'incomplete'
        memory = available_memory()
        free = shutil.disk_usage(WORK).free
        print(f'SHAPE={SHAPE}; tiling_key={stages.KEY}; cached_binary={kind}', flush=True)
        print(f'CPU SIMULATOR: logical device 0; physical NPU launches=0; CPU cores=1; available_memory={memory / GIB:.2f}GiB', flush=True)
        if memory < 24 * GIB:
            raise MemoryError(f'24GiB available memory is required before simulation; found {memory / GIB:.2f}GiB; nothing was launched')
        if free < 3 * GIB:
            raise OSError(f'3GiB free disk space is required; found {free / GIB:.2f}GiB; nothing was launched')
        source, model = private_simulator(sdk, geometry)
        env = stages.runtime_env(vendor, dict(os.environ))
        for name in list(env):
            if name.startswith('FA_') or name in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_DEVICE_ID', 'DEVICE_ID'):
                env.pop(name)
        env.update({'FA_' + name: str(value) for name, value in zip(stages.cost.INPUT_FIELDS[:7], SHAPE[:7])})
        env.update(FA_NPU_ID='0', FA_DTYPE_KIND='1', FA_DTYPE_BYTES='2', FA_LAYOUT='0', FA_HAS_DROP='0',
            FA_INPUT_PATTERN='1', FA_WARMUP='0', FA_REPEAT='1', FA_PROFILE='0', FA_STAGE_PROFILE_ID='0',
            FA_CUSTOM_OPP_ROOT=str(vendor), ATTENTION_CUSTOM_OPP_ROOT=str(vendor),
            FA_ROUTE_TRACE=str(WORK / 'route.tsv'), FA_EFFECTIVE_TILING_TRACE=str(WORK / 'effective.tsv'),
            FA_HARDWARE_TRACE=str(WORK / 'hardware.json'),
            LD_LIBRARY_PATH=str(model) + ':' + env.get('LD_LIBRARY_PATH', ''),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        help_result = subprocess.run([str(profiler), 'op', 'simulator', '--help'], env=env, capture_output=True, text=True, timeout=30)
        help_text = help_result.stdout + help_result.stderr
        if help_result.returncode or not all(option in help_text for option in ('--core-id', '--launch-count', '--output')):
            raise RuntimeError('installed msprof does not expose the required official simulator options')
        if (WORK / 'profile').exists():
            shutil.rmtree(WORK / 'profile')
        for path in (WORK / 'route.tsv', WORK / 'effective.tsv', WORK / 'hardware.json', WORK / 'output/output.bin'):
            path.unlink(missing_ok=True)
        for name in ('numerical_check', 'hardware', 'effective_tiling', 'recovered_existing_trace', 'cache_hit'):
            metadata.pop(name, None)
        marker.unlink(missing_ok=True)
        (WORK / 'run_manifest.json').write_text(json.dumps(signature, sort_keys=True) + '\n')
        argv = [str(profiler), 'op', 'simulator', '--aic-metrics=PipeUtilization', '--kernel-name=FlashAttentionScore',
                '--launch-count=1', '--core-id=0', '--output=' + str(WORK / 'profile'), str(binary)]
        metadata.update(command=argv, simulator_source=str(source), simulator_private_config=str(model),
            simulator_config={path.name: json.loads(path.read_text()) for path in model.glob('*.json')},
            expected_schedule=geometry, simulated_topology={'aic_cores': geometry['aic_cores'], 'aiv_cores': geometry['aiv_cores']})
        print(f'SIMULATOR START: one kernel launch; parse Cube 0 and its two Vector subcores; log={WORK / "profile.log"}', flush=True)
        monitor(argv, env, args.timeout)
        metadata.update(validate_result(geometry), status='complete')
        counts = export_report(metadata, reference_rows)
        if not counts['complete']:
            raise RuntimeError(f'instruction collection is incomplete: {dict(counts)}; raw files retained')
        marker.write_text(json.dumps({'signature': signature, 'metadata': metadata}, sort_keys=True) + '\n')
        (WORK / 'output/output.bin').unlink(missing_ok=True)
        print(f'COMPLETE: instructions={counts["instruction"]}, dynamic_events={counts["dynamic_events"]}; CSV={OUTPUT}; bytes={OUTPUT.stat().st_size}', flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=str(error))
        log = WORK / 'profile.log'
        if log.is_file():
            with log.open('rb') as source:
                source.seek(max(0, log.stat().st_size - 8192))
                metadata['error_log_tail'] = '\n'.join(source.read().decode(errors='replace').splitlines()[-10:])
        try:
            export_report(metadata, reference_rows)
        except Exception as export_error:
            stages.save_csv(OUTPUT, [{'record_type': 'metadata', 'extra_json': json.dumps({**metadata, 'export_error': str(export_error)})}])
        print(f'FAILED: {error}\nCSV={OUTPUT}\nLOG={WORK / "profile.log"}', flush=True)
        if metadata.get('error_log_tail'):
            print(metadata['error_log_tail'], flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
