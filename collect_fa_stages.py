from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import fa_latency_inputs as cost

ROOT = Path(__file__).resolve().parent
KEY = 1153197616
LIMIT_SECONDS = 900
REPEATS = 3
CASES = (
    ('fa_branch_same_ab_nz_9265ddfd9e63', (1, 8, 1, 256, 641, 96, 96, 'FP16', 'BNSD', 0), 'first'),
    ('fa_branch_same_ab_nz_12d4cd573d21', (1, 8, 1, 256, 1282, 96, 96, 'FP16', 'BNSD', 0), 'first'),
    ('fa_branch_same_ab_nz_5ab5d8e5a801', (2, 32, 1, 256, 1282, 96, 96, 'FP16', 'BNSD', 0), 'last'),
)
PHASES = ('bmm1', 'vec1', 'bmm2', 'vec2')
PIPE_COLUMNS = {
    'aic': ('cube', 'mte1', 'mte2', 'mte3', 'fixpipe', 'scalar'),
    'aiv': ('vec', 'mte2', 'mte3', 'scalar'),
}
MEMORY_COLUMNS = {'gm_to_l1_kb': 'GM_to_L1_datas(KB)', 'l0c_to_gm_kb': 'L0C_to_GM_datas(KB)',
                  'gm_to_ub_kb': 'GM_to_UB_datas(KB)', 'ub_to_gm_kb': 'UB_to_GM_datas(KB)'}


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def profile_table(path):
    with path.open(newline='', encoding='utf-8-sig') as source:
        rows = [[cell.strip() for cell in row] for row in csv.reader(source) if any(cell.strip() for cell in row)]
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError(f'invalid profiler CSV: {path}')
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def read_trace(folder):
    route = None
    effective = None
    for line in (folder / 'route.tsv').read_text().splitlines():
        fields = line.split('\t')
        if len(fields) == 4 and fields[0] == 'FlashAttentionScore' and fields[2] == '0':
            route = int(fields[1]), int(fields[3])
    for line in (folder / 'effective.tsv').read_text().splitlines():
        fields = line.split('\t')
        if fields[0] == 'FINAL':
            effective = {field.partition('=')[0]: int(field.partition('=')[2]) for field in fields[2:] if '=' in field}
    return route, effective


def check_output(inputs, path):
    p = cost.normalize(inputs)
    if p['dtype'] != 'FP16' or p['layout'] != 'BNSD' or p['has_drop_mask']:
        raise ValueError('this phase measurement uses the three existing FP16 BNSD cases without dropout')
    elements = p['B'] * p['N1'] * p['S1'] * p['DV']
    if not path.is_file() or path.stat().st_size != elements * 4:
        raise RuntimeError('attention output is missing or has an incorrect size')
    output = np.memmap(path, dtype=np.float32, mode='r')
    def tensor(batch, head, rows, heads, sequence, width, salt):
        indices = ((batch * heads + head) * sequence + rows[:, None]) * width + np.arange(width)
        hashed = indices.astype(np.uint32) ^ np.uint32(salt)
        hashed ^= hashed >> 16
        hashed *= np.uint32(0x7feb352d)
        hashed ^= hashed >> 15
        hashed *= np.uint32(0x846ca68b)
        hashed ^= hashed >> 16
        values = ((hashed & 2047).astype(np.int32) - 1024).astype(np.float32) / np.float32(1024.0)
        return values.astype(np.float16).astype(np.float64)
    checked, maximum = 0, 0.0
    for batch in sorted({0, p['B'] - 1}):
        for head in sorted({0, p['N1'] - 1}):
            kv = head // (p['N1'] // p['N2'])
            positions = np.arange(p['S2'])
            keys = tensor(batch, kv, positions, p['N2'], p['S2'], p['D'], 0x87654321)
            values = tensor(batch, kv, positions, p['N2'], p['S2'], p['DV'], 0x31415926)
            for query in sorted({0, p['S1'] // 2, p['S1'] - 1}):
                q = tensor(batch, head, np.array([query]), p['N1'], p['S1'], p['D'], 0x12345678)[0]
                scores = keys @ q / math.sqrt(p['D'])
                weights = np.exp(scores - np.max(scores))
                golden = (weights / np.sum(weights)) @ values
                offset = ((batch * p['N1'] + head) * p['S1'] + query) * p['DV']
                actual = np.asarray(output[offset:offset + p['DV']], dtype=np.float64)
                if not np.all(np.isfinite(actual)) or not np.allclose(actual, golden, rtol=0.01, atol=1e-4):
                    raise RuntimeError(f'independent numerical check failed: b={batch}, head={head}, query={query}')
                maximum = max(maximum, float(np.max(np.abs(actual - golden))))
                checked += p['DV']
    return {'checked_elements': checked, 'max_abs_error': maximum}


def command(argv, env, cwd, log, deadline, maximum):
    remaining = min(maximum, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError('15-minute total budget exhausted; no further command was started')
    with Path(log).open('w') as output:
        process = subprocess.Popen(argv, env=env, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise
    if code:
        lines = Path(log).read_text(errors='replace').splitlines()
        raise RuntimeError(f'command exited {code}; log={log}; ' + ' | '.join(lines[-5:]))


def prepare_runner(ops_root, folder):
    base = ROOT / 'patches/fa_candidate_hook.patch'
    stage = ROOT / 'patches/fa_stage_profile.patch'
    files = sorted({line[6:] for patch in (base, stage) for line in patch.read_text().splitlines() if line.startswith('+++ b/')})
    dirty = subprocess.run(['git', '-C', str(ops_root), 'diff', '--quiet', '--', *files])
    if dirty.returncode:
        raise RuntimeError('official source files for this measurement have local changes; they were left untouched')
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='source_', dir=folder) as temporary:
        checkout = Path(temporary)
        for name in files:
            target = checkout / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ops_root / name).read_bytes())
        for args in (['init', '-q'], ['add', '.'], ['apply', str(base)], ['apply', str(stage)]):
            result = subprocess.run(['git', '-C', temporary, *args], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('stage instrumentation does not match official v8.5.0: ' + result.stderr[-2000:])
        patch = subprocess.run(['git', '-C', temporary, 'diff', '--binary', '--no-ext-diff'], check=True, capture_output=True, text=True).stdout
        kernel_sources = {name.split('/op_kernel/', 1)[1]: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
                          for name in files if '/op_kernel/' in name}
    (folder / 'patches').mkdir(exist_ok=True)
    (folder / 'patches/fa_candidate_hook.patch').write_text(patch)
    shutil.copy2(ROOT / 'patches/ops_transformer_incremental_build.patch', folder / 'patches')
    for name in ('build.sh', 'run.sh', 'fa_launcher.cpp'):
        shutil.copy2(ROOT / name, folder / name)
    return hashlib.sha256(patch.encode()).hexdigest(), kernel_sources


def cache_location(base):
    version = os.environ.get('OPS_TRANSFORMER_VERSION', 'v8.5.0')
    soc = os.environ.get('FA_SOC_UNIT', 'ascend910b')
    folder = base / version / soc / str(KEY) / 'dtype_1'
    vendors = list((folder / 'opp/vendors').glob('*_transformer'))
    if len(vendors) != 1 or not (folder / 'fa_npu').is_file():
        raise FileNotFoundError(f'cached FP16 FA binary is missing: {folder}; the baseline will not be rebuilt')
    vendor = vendors[0]
    found = False
    for path in vendor.glob(f'op_impl/ai_core/tbe/kernel/{soc}/flash_attention_score/*.json'):
        data = json.loads(path.read_text())
        keys = {int(item['tilingKey']) for item in data.get('kernelList', []) if 'tilingKey' in item}
        inputs = {item.get('name'): item for item in data.get('supportInfo', {}).get('inputs', [])}
        binary = path.with_name(data.get('binFileName', '') + data.get('binFileSuffix', '.o'))
        if KEY in keys and inputs.get('query', {}).get('dtype') == 'float16' and binary.is_file():
            found = True
            break
    if not found or not (vendor / 'bin/set_env.bash').is_file():
        raise RuntimeError(f'the cached package does not contain FP16 tiling_key={KEY}: {vendor}')
    return folder / 'fa_npu', vendor


def runtime_env(vendor, env):
    sdk = Path(env.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest'))
    script = 'set -e; if [ -f "$1" ]; then source "$1"; fi; source "$2"; printf "\\0FA_ENV_BEGIN\\0"; env -0'
    result = subprocess.run(['bash', '-c', script, 'bash', str(sdk / 'bin/setenv.bash'), str(vendor / 'bin/set_env.bash')],
                            env=env, capture_output=True, timeout=20)
    marker = b'\0FA_ENV_BEGIN\0'
    if result.returncode or marker not in result.stdout:
        raise RuntimeError(f'could not source runtime environment: {vendor / "bin/set_env.bash"}')
    exported = dict(item.decode().split('=', 1) for item in result.stdout.split(marker, 1)[1].split(b'\0') if b'=' in item)
    exported.update(FA_CUSTOM_OPP_ROOT=str(vendor), ATTENTION_CUSTOM_OPP_ROOT=str(vendor))
    return exported


def save_csv(path, rows):
    fields = list(dict.fromkeys(name for row in rows for name in row))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def raw_counter_diagnostics(groups):
    selected_columns = re.compile(r'fops|flops|instr|mac|fp16|fp32|wait|stall|conflict|cflt|total_cycles', re.I)
    result = {}
    for kind in ('ArithmeticUtilization', 'PipeUtilization'):
        for engine, prefix in (('aic', 'cube'), ('aiv', 'vector')):
            values = {}
            for (table_kind, _), records in groups.items():
                if table_kind != kind:
                    continue
                cores = [row for row in records if row.get('sub_block_id', '').startswith(prefix)]
                for name in sorted({name for row in cores for name in row if selected_columns.search(name)}):
                    if name.startswith(('aic_', 'aiv_')) and not name.startswith(engine + '_'):
                        continue
                    numbers = [finite(row.get(name)) for row in cores]
                    numbers = [number for number in numbers if number is not None]
                    if numbers:
                        values.setdefault(name, []).append((sum(numbers), statistics.mean(numbers), max(numbers)))
            diagnostics = {}
            for name, replays in values.items():
                if not any(any(value != 0 for value in replay) for replay in replays):
                    continue
                item = {'core_mean': statistics.median(replay[1] for replay in replays),
                        'core_max': statistics.median(replay[2] for replay in replays),
                        'replay_core_max_min': min(replay[2] for replay in replays),
                        'replay_core_max_max': max(replay[2] for replay in replays), 'replays': len(replays)}
                if not re.search(r'ratio|rate|\(\%\)', name, re.I):
                    item['core_sum'] = statistics.median(replay[0] for replay in replays)
                diagnostics[name] = item
            if diagnostics:
                result[kind + '/' + engine] = diagnostics
    return result


def parse_metrics(folder, phase, allow_partial=False):
    target_engine = None if phase not in PHASES else 'aic' if phase in ('bmm1', 'bmm2') else 'aiv'
    groups = {}
    for path in sorted(folder.rglob('*.csv')):
        match = re.fullmatch(r'(?:\d+_)?(OpBasicInfo|PipeUtilization|Memory|ArithmeticUtilization)_(\d{17})\.csv', path.name)
        if match is None:
            continue
        records = profile_table(path)
        identity = match.groups()
        if identity in groups and groups[identity] != records:
            raise RuntimeError('conflicting profiler tables for one launch')
        groups[identity] = records
    basic = [row for (kind, _), records in groups.items() if kind == 'OpBasicInfo' for row in records
             if re.match(rf'^FlashAttentionScore_[A-Za-z0-9]+_{KEY}(?:_|$)', row.get('Op Name', ''))]
    durations = [float(row['Task Duration(us)']) for row in basic if finite(row.get('Task Duration(us)'))]
    if len(durations) != REPEATS and not allow_partial:
        raise RuntimeError(f'expected {REPEATS} kernel launches, got {len(durations)}')
    result = {'profile_launches': len(durations), 'task_duration_us': statistics.median(durations) if durations else '',
              'task_duration_min_us': min(durations) if durations else '', 'task_duration_max_us': max(durations) if durations else '',
              'task_duration_spread_ratio': max(durations) / min(durations) if durations and min(durations) > 0 else '',
              'task_duration_samples_us': json.dumps(durations),
              'basic_stamps': json.dumps(sorted(stamp for kind, stamp in groups if kind == 'OpBasicInfo')),
              'pipe_stamps': json.dumps(sorted(stamp for kind, stamp in groups if kind == 'PipeUtilization')),
              'memory_stamps': json.dumps(sorted(stamp for kind, stamp in groups if kind == 'Memory')),
              'profile_pids': json.dumps(sorted({row.get('Pid', '') for row in basic})),
              'counter_diagnostics_json': json.dumps(raw_counter_diagnostics(groups), separators=(',', ':')),
              'raw_metric_table_counts': json.dumps({kind: sum(item == kind for item, _ in groups) for kind in ('OpBasicInfo', 'PipeUtilization', 'Memory', 'ArithmeticUtilization')})}
    freq = [float(row['Current Freq']) for row in basic if finite(row.get('Current Freq'))]
    result.update(freq_min_mhz=min(freq) if freq else '', freq_max_mhz=max(freq) if freq else '')
    for engine, pipes in PIPE_COLUMNS.items():
        core = 'cube' if engine == 'aic' else 'vector'
        launches = []
        for (kind, _), records in groups.items():
            if kind != 'PipeUtilization':
                continue
            selected = [row for row in records if row.get('sub_block_id', '').startswith(core)]
            if not selected and target_engine in (None, engine) and not allow_partial:
                raise RuntimeError(f'no {engine} core rows in PipeUtilization')
            values = {pipe: [finite(row.get(f'{engine}_{pipe}_time(us)')) for row in selected] for pipe in pipes}
            required = ('cube', 'mte1', 'mte2', 'fixpipe') if engine == 'aic' else ('vec', 'mte2', 'mte3')
            if not allow_partial and target_engine in (None, engine) and any(not any(value is not None for value in values[pipe]) for pipe in required):
                raise RuntimeError(f'missing {engine} per-pipe active times')
            observed = {pipe: [value for value in values[pipe] if value is not None] for pipe in pipes}
            core_times = [finite(row.get(engine + '_time(us)')) for row in selected]
            core_times = [value for value in core_times if value is not None]
            launches.append({**{f'{engine}_{pipe}_sum_us': sum(observed[pipe]) if observed[pipe] else None for pipe in pipes},
                             **{f'{engine}_{pipe}_max_us': max(observed[pipe]) if observed[pipe] else None for pipe in pipes},
                             f'{engine}_own_table_time_max_us': max(core_times) if core_times else None,
                             f'{engine}_active_cores': sum(any((values[pipe][index] or 0) > 1e-6 for pipe in required) for index in range(len(selected)))})
        if len(launches) != REPEATS and not allow_partial:
            raise RuntimeError(f'expected {REPEATS} PipeUtilization tables, got {len(launches)}')
        names = list(launches[0]) if launches else [name for pipe in pipes for name in (f'{engine}_{pipe}_sum_us', f'{engine}_{pipe}_max_us')] + [f'{engine}_active_cores', f'{engine}_own_table_time_max_us']
        for name in names:
            values = [row[name] for row in launches if row[name] is not None]
            result[name] = statistics.median(values) if values else ''
            result[name.replace('_us', '_samples_us') if name.endswith('_us') else name + '_samples'] = json.dumps(values)
        for pipe in pipes:
            values = [row[f'{engine}_{pipe}_sum_us'] for row in launches if row[f'{engine}_{pipe}_sum_us'] is not None]
            result[f'{engine}_{pipe}_spread_ratio'] = max(values) / min(values) if values and min(values) > 0 else ''
    memory = []
    for (kind, _), records in groups.items():
        if kind != 'Memory':
            continue
        row = {}
        for name, column in MEMORY_COLUMNS.items():
            values = [finite(item.get(column)) for item in records]
            values = [value for value in values if value is not None]
            owner = 'aic' if name in ('gm_to_l1_kb', 'l0c_to_gm_kb') else 'aiv'
            if not values and target_engine in (None, owner) and not allow_partial:
                raise RuntimeError(f'missing memory counter: {column}')
            row[name] = sum(values) if values else None
            prefix = 'cube' if owner == 'aic' else 'vector'
            owned = [finite(item.get(column)) for item in records if item.get('sub_block_id', '').startswith(prefix)]
            owned = [value for value in owned if value is not None]
            row[name + '_owner_only'] = sum(owned) if owned else None
        memory.append(row)
    if len(memory) != REPEATS and not allow_partial:
        raise RuntimeError(f'expected {REPEATS} Memory tables, got {len(memory)}')
    for name in (*MEMORY_COLUMNS, *(name + '_owner_only' for name in MEMORY_COLUMNS)):
        values = [row[name] for row in memory if row[name] is not None]
        result[name] = statistics.median(values) if values else ''
        result[name + '_samples'] = json.dumps(values)
    if phase in PHASES:
        target = 'aic' if phase in ('bmm1', 'bmm2') else 'aiv'
        other = 'aiv' if target == 'aic' else 'aic'
        busy = {engine: sum(finite(result[f'{engine}_{pipe}_sum_us']) or 0 for pipe in pipes if pipe != 'scalar') for engine, pipes in PIPE_COLUMNS.items()}
        result['gating_target_engine'] = target
        result['gating_target_busy_us'] = busy[target]
        result['gating_other_busy_us'] = busy[other]
        result['gating_other_target_ratio'] = busy[other] / busy[target] if busy[target] > 0 else ''
        result['gating_verified'] = busy[target] > 0 and busy[other] <= max(0.1, busy[target] * 0.1)
        result['gating_error'] = 'selected_stage_empty' if busy[target] <= 0 else 'unselected_engine_active' if not result['gating_verified'] else ''
        if not any(kind == 'PipeUtilization' for kind, _ in groups):
            result['gating_error'] = 'no_cached_pipe_tables'
        result['unselected_engine_counters_reported'] = any(result[f'{other}_{pipe}_sum_us'] != '' for pipe in PIPE_COLUMNS[other])
    return result


def export_cached(output, work, npu_id):
    if not output.is_file():
        raise FileNotFoundError(f'existing stage collection is missing: {output}; no NPU operation was attempted')
    with output.open(newline='') as source:
        rows = list(csv.DictReader(source))
    if any(int(row['physical_npu_id']) != npu_id for row in rows):
        raise RuntimeError('the saved collection uses another NPU ID; export with its recorded --id')
    recovered, schema = 0, {}
    for row in rows:
        row.setdefault('acquisition_status', row['status'])
        row.setdefault('acquisition_error', row.get('error', ''))
        row.update(export_action='cached_tables_only', calibration_ready=False,
                   numerical_check_recorded=(finite(row.get('checked_elements')) or 0) > 0,
                   diagnostic_error='')
        folder = work / 'cases' / row['case'] / (row['phase'] + '_' + row['key_window'])
        if work.resolve() not in folder.resolve().parents:
            raise RuntimeError('invalid saved case path')
        tables = folder / 'tables'
        files = list(tables.glob('*.csv')) if tables.is_dir() else []
        row['cached_table_files'] = len(files)
        if not files:
            row['diagnostic_error'] = 'cached metric tables are missing; original status and values retained'
            continue
        try:
            row.update(parse_metrics(tables, row['phase'], allow_partial=True))
            issues = [row['gating_error']] if row.get('gating_error') else []
            try:
                route, effective = read_trace(folder)
                row['cached_route_matches'] = route == (95, KEY)
                row['cached_stage_selector'] = effective.get('FA_STAGE_PROFILE_ID', '') if effective else ''
                row['cached_selector_matches'] = row['phase'] in ('baseline', 'protocol_reference') or row['cached_stage_selector'] == int(row['stage_selector'])
                if not row['cached_route_matches']:
                    issues.append('cached route differs from the selected route')
                if not row['cached_selector_matches']:
                    issues.append('cached stage selector differs from the requested selector')
            except (OSError, ValueError, KeyError) as error:
                issues.append('cached trace unavailable: ' + str(error))
            row['diagnostic_error'] = '; '.join(issues)
            for path in files:
                match = re.fullmatch(r'(?:\d+_)?(OpBasicInfo|PipeUtilization|Memory|ArithmeticUtilization)_\d{17}\.csv', path.name)
                if match and match[1] not in schema:
                    records = profile_table(path)
                    if records:
                        schema[match[1]] = list(records[0])
            recovered += 1
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            row['diagnostic_error'] = str(error)
    if rows:
        rows[0]['raw_metric_column_names'] = json.dumps(schema, separators=(',', ':'))
    save_csv(output, rows)
    print(f'CACHED EXPORT: {recovered}/{len(rows)} rows have readable raw tables; NPU launches=0; compilations=0', flush=True)
    for row in rows:
        if row['phase'] in PHASES:
            print(f'{row["case"]} {row["phase"]} {row["key_window"]}: target={row.get("gating_target_busy_us", "")} other={row.get("gating_other_busy_us", "")} diagnosis={row["diagnostic_error"]}', flush=True)
    print(f'CSV={output}; diagnostic values are not accepted stage measurements', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=4)
    parser.add_argument('--export-cached', action='store_true')
    args = parser.parse_args()
    if args.id < 0:
        parser.error('--id must be nonnegative')
    os.chdir(ROOT)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    work = ROOT / 'out/fa_stage_profile'
    output = ROOT / 'output/fa_stage_profile/stages.csv'
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (work / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.export_cached:
        export_cached(output, work, args.id)
        return
    deadline = time.monotonic() + LIMIT_SECONDS
    env = dict(os.environ)
    env.update(FA_NPU_ID=str(args.id), ASCEND_RT_VISIBLE_DEVICES=str(args.id), FA_DTYPE_KIND='1', FA_BUILD_JOBS='1', FA_TILING_KEY=str(KEY))
    env.pop('FA_DISCOVER_ONLY', None)
    normal_cache = Path(env.get('FA_CACHE_ROOT', str(ROOT / 'out/fa_cache'))).resolve()
    normal_binary, normal_vendor = cache_location(normal_cache)
    normal_env = runtime_env(normal_vendor, env)
    profiler = shutil.which('msprof', path=normal_env.get('PATH'))
    if profiler is None:
        raise FileNotFoundError('msprof is missing from the configured CANN environment')
    help_result = subprocess.run([profiler, 'op', '--help'], env=normal_env, capture_output=True, text=True, timeout=20)
    if '--aic-metrics' not in help_result.stdout + help_result.stderr:
        raise RuntimeError('msprof operator-profiling options are unavailable; no compilation or measurement was started')
    with tempfile.TemporaryDirectory(prefix='capability_', dir=work) as temporary:
        probe = subprocess.run([profiler, 'op', '--aic-metrics=KernelScale,PipeUtilization,Memory,ArithmeticUtilization',
                                str(Path(temporary) / 'missing_application')], env=normal_env, cwd=temporary,
                               capture_output=True, text=True, timeout=20)
    probe_text = (probe.stdout + probe.stderr).lower()
    if 'invalid metric' in probe_text or re.search(r'argument --aic-metrics[^\n]*invalid', probe_text):
        raise RuntimeError('installed msprof rejects KernelScale; no instrumentation was compiled')
    if 'device profiling is not supported' in probe_text or "can't find valid libdcmi" in probe_text:
        raise RuntimeError('NPU profiling environment is unavailable; no instrumentation was compiled')
    runner = work / 'runner'
    ops = Path(env.get('OPS_TRANSFORMER_ROOT', str(ROOT.parent / 'ops-transformer-official-8.5.0'))).resolve()
    patch_sha, kernel_sources = prepare_runner(ops, runner)
    kernel_source_sha = hashlib.sha256(json.dumps(kernel_sources, sort_keys=True).encode()).hexdigest()
    manifest = normal_binary.parent / 'build_manifest.txt'
    signature = hashlib.sha256(json.dumps({'patch': patch_sha, 'cases': CASES, 'repeats': REPEATS,
        'warmup': 2, 'metrics': 'KernelScale,PipeUtilization,Memory,ArithmeticUtilization', 'input_pattern': 1, 'unmarked_instrumented_control': True,
        'baseline_manifest': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None}, sort_keys=True).encode()).hexdigest()
    instrumented_signature = hashlib.sha256((signature + kernel_source_sha).encode()).hexdigest()
    reference_signature = hashlib.sha256((signature + ':native:default_metrics:profiler_warmup10:launcher_warmup5:launch10:rounds3').encode()).hexdigest()
    plan = []
    for name, shape, query_position in CASES:
        inputs = dict(zip(cost.INPUT_FIELDS, shape))
        positions = ('single',) if inputs['S2'] <= 1024 else ('first', 'last')
        for phase, position in [('baseline', 'all'), ('control', 'all')] + [(phase, position) for position in positions for phase in PHASES]:
            selector = 0 if phase in ('baseline', 'control') else PHASES.index(phase) + 1 + (256 if position == 'last' else 0) + (512 if query_position == 'last' else 0)
            plan.append({'case': name, **inputs, 'tiling_key': KEY, 'physical_npu_id': args.id, 'phase': phase, 'key_window': position,
                         'query_task': query_position if phase not in ('baseline', 'control') else 'all', 'stage_selector': selector, 'instrumented': phase != 'baseline',
                         'measurement_scope': 'unmodified_kernel' if phase == 'baseline' else 'instrumented_kernel_without_markers' if phase == 'control' else 'isolated_phase_engine_counters_one_query_task_per_core',
                         'phase_wall_time_measured': False, 'barriers_around_selected_phase': phase not in ('baseline', 'control'),
                         'protocol_sha256': signature if phase == 'baseline' else instrumented_signature,
                         'expected_kernel_source_sha256': '' if phase == 'baseline' else kernel_source_sha,
                         'status': 'pending', 'error': ''})
    for round_id in range(1, 4):
        plan.append({**plan[0], 'phase': 'protocol_reference', 'key_window': 'round_' + str(round_id),
                     'instrumented': False, 'measurement_scope': 'unmodified_kernel_original_collection_protocol',
                     'protocol_sha256': reference_signature, 'profiler_warmup': 10, 'launcher_warmup': 5,
                     'requested_profile_launches': 10, 'profile_metrics': 'Default'})
    previous = list(csv.DictReader(output.open())) if output.is_file() else []
    saved = {(row['case'], row['phase'], row['key_window']): row for row in previous
             if row.get('status') == 'ok' and row.get('protocol_sha256') ==
             {'baseline': signature, 'protocol_reference': reference_signature}.get(row['phase'], instrumented_signature)
             and row.get('physical_npu_id') == str(args.id)}
    for row in plan:
        old = saved.get((row['case'], row['phase'], row['key_window']))
        if old is not None:
            row.update(old)
    save_csv(output, plan)
    if all(row['status'] == 'ok' for row in plan):
        print(f'COMPLETE: all {len(plan)} measurements already cached; CSV={output}', flush=True)
        return
    print(f'SHAPES: 3; tiling_key={KEY}; stage measurements=20; cached-baseline measurements=3; unmarked controls=3; NPU={args.id}', flush=True)
    print('BUILD: at most one instrumented tiling key, one CPU; baseline binaries are reused; total execution limit=900s', flush=True)
    print('REFERENCE: one existing shape, original cached binary, original profiler settings, three rounds; successful stage rows are reused', flush=True)
    stage_env = dict(env, FA_CACHE_ROOT=str(work / 'cache'), OPS_TRANSFORMER_ROOT=str(ops), FA_KERNEL_SOURCE_SHA256=kernel_source_sha)
    stage_binary = stage_vendor = None
    for index, row in enumerate(plan, 1):
        if row['status'] == 'ok':
            print(f'CACHE HIT {index}/{len(plan)} {row["case"]} {row["phase"]} {row["key_window"]}', flush=True)
            continue
        if time.monotonic() >= deadline:
            row.update(status='not_measured', error='15-minute total budget exhausted')
            save_csv(output, plan)
            continue
        native = row['phase'] in ('baseline', 'protocol_reference')
        reference = row['phase'] == 'protocol_reference'
        if not native and stage_binary is None:
            try:
                print(f'BUILD CHECK: instrumented tiling_key={KEY}; one CPU; limit=600s; log={work / "build.log"}', flush=True)
                command(['bash', str(runner / 'build.sh')], stage_env, runner, work / 'build.log', deadline, 600)
                for line in (work / 'build.log').read_text(errors='replace').splitlines():
                    if line.startswith(('BUILD CACHE HIT:', 'BUILD READY:')):
                        print(line, flush=True)
                stage_binary, stage_vendor = cache_location(work / 'cache')
                if 'kernel_source=' + kernel_source_sha not in (stage_binary.parent / 'build_manifest.txt').read_text().splitlines():
                    raise RuntimeError('instrumented cache has no matching kernel source fingerprint; no NPU measurement started')
                provenance = stage_binary.parent / 'stage_source_manifest.json'
                if not provenance.is_file() or json.loads(provenance.read_text()) != kernel_sources:
                    copied = ops / 'build/binary' / env.get('FA_SOC_UNIT', 'ascend910b') / 'src/flash_attention_score/op_kernel'
                    for name, expected in kernel_sources.items():
                        path = copied / name
                        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                            raise RuntimeError(f'compiled kernel source differs from the instrumented source: {path}; no NPU measurement started')
                    provenance.write_text(json.dumps(kernel_sources, sort_keys=True))
                for pending in plan:
                    if pending['phase'] not in ('baseline', 'protocol_reference'):
                        pending.update(compiled_kernel_source_sha256=kernel_source_sha, kernel_source_verified=True)
                print(f'KERNEL SOURCE VERIFIED: {kernel_source_sha}', flush=True)
                stage_env = runtime_env(stage_vendor, stage_env)
            except BaseException as error:
                for pending in plan:
                    if pending['status'] != 'ok':
                        pending.update(status='not_measured', error='instrumented build unavailable: ' + str(error))
                save_csv(output, plan)
                raise
        folder = work / 'cases' / row['case'] / (row['phase'] + '_' + row['key_window'])
        folder.mkdir(parents=True, exist_ok=True)
        inputs = {name: row[name] for name in cost.INPUT_FIELDS}
        configured = dict(normal_env if native else stage_env)
        configured.update({'FA_' + name: str(inputs[name]) for name in cost.INPUT_FIELDS[:7]})
        configured.update(FA_DTYPE_KIND='1', FA_DTYPE_BYTES='2', FA_LAYOUT='0', FA_HAS_DROP='0')
        configured.update(FA_CUSTOM_OPP_ROOT=str(normal_vendor if native else stage_vendor),
                          ATTENTION_CUSTOM_OPP_ROOT=str(normal_vendor if native else stage_vendor),
                          FA_STAGE_PROFILE_ID=str(row['stage_selector']), FA_WARMUP='5' if reference else '0', FA_REPEAT='10' if reference else str(REPEATS),
                          FA_INPUT_PATTERN='1', FA_PROFILE='0', FA_EFFECTIVE_TILING_TRACE=str(folder / 'effective.tsv'),
                          FA_ROUTE_TRACE=str(folder / 'route.tsv'), FA_HARDWARE_TRACE=str(folder / 'hardware.json'))
        configured['LD_LIBRARY_PATH'] = (normal_env if native else stage_env).get('LD_LIBRARY_PATH', '')
        configured['ASCEND_CUSTOM_OPP_PATH'] = (normal_env if native else stage_env).get('ASCEND_CUSTOM_OPP_PATH', '')
        for name in ('FA_S1_BASE', 'FA_S2_BASE', 'FA_N_RATIO', 'FA_BMM_PRESET', 'FA_LATENCY_TRACE', 'FA_DISCOVER_ONLY'):
            configured.pop(name, None)
        (folder / 'output').mkdir(exist_ok=True)
        for path in (folder / 'route.tsv', folder / 'effective.tsv', folder / 'hardware.json', folder / 'output/output.bin'):
            path.unlink(missing_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix='profiler_', dir=folder))
        metrics = 'PipeUtilization,Memory,ArithmeticUtilization'
        if row['phase'] in PHASES:
            metrics = 'KernelScale,' + metrics
        binary = normal_binary if native else stage_binary
        argv = [profiler, 'op', '--aic-metrics=' + metrics, '--kernel-name=FlashAttentionScore', '--warm-up=2',
                '--launch-count=' + str(REPEATS), '--output=' + str(scratch), str(binary)]
        if reference:
            argv = [profiler, 'op', '--kernel-name=FlashAttentionScore', '--warm-up=10', '--launch-count=10',
                    '--launch-skip-before-match=5', '--output=' + str(scratch), str(binary)]
        start = time.monotonic()
        print(f'MEASURE {index}/{len(plan)} {row["case"]} {row["phase"]} {row["key_window"]} query={row["query_task"]}', flush=True)
        try:
            command(argv, configured, folder, folder / 'profile.log', deadline, 120 if reference else 45)
            route, effective = read_trace(folder)
            priority, key = route if route is not None else (None, None)
            if priority != 95 or key != KEY or effective is None or effective['FA_BMM_PRESET'] != 0:
                raise RuntimeError('Host route or Matmul preset differs from the selected existing case')
            if not native and effective.get('FA_STAGE_PROFILE_ID') != int(row['stage_selector']):
                raise RuntimeError('instrumented Host did not record the selected phase')
            row.update(parse_metrics(scratch, row['phase'], allow_partial=reference))
            if reference and (row['profile_launches'] != 10 or any(json.loads(row['raw_metric_table_counts'])[kind] != 10
                                                                  for kind in ('PipeUtilization', 'Memory', 'ArithmeticUtilization'))):
                raise RuntimeError('original-protocol reference did not produce ten complete profiler samples')
            correctness = check_output(inputs, folder / 'output/output.bin')
            hardware = json.loads((folder / 'hardware.json').read_text())
            row.update(checked_elements=correctness['checked_elements'], max_abs_error=correctness['max_abs_error'],
                       **{name: hardware[name] for name in ('soc', 'aic_cores', 'aiv_cores')},
                       core_num=effective['FA_CORE_NUM'], total_tasks=effective['FA_TOTAL_TASKS'], split_factor=effective['FA_SPLIT_FACTOR'],
                       s1_base=effective['FA_S1_BASE'], s2_base=effective['FA_S2_BASE'], n_ratio=effective['FA_N_RATIO'],
                       collected_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status='ok', error='')
            if not row.get('gating_verified', True):
                row.update(status='range_unverified', error=row.get('gating_error', 'stage range is not verified'))
                print(f'UNVERIFIED: {row["phase"]} {row["key_window"]}; counters retained in CSV', flush=True)
                continue
            print(f'OK: {row["phase"]} {row["key_window"]}; isolated counters only; msprof Task Duration={row["task_duration_us"]:.6g} us', flush=True)
        except Exception as error:
            row.update(status='failed', error=str(error))
            print(f'FAILED: {row["phase"]} {row["key_window"]}: {error}', flush=True)
            if row['phase'] == 'baseline' and stage_binary is None:
                for pending in plan:
                    if pending['status'] == 'pending':
                        pending.update(status='not_measured', error='cached baseline failed; instrumentation was not compiled')
                save_csv(output, plan)
                raise
        finally:
            row['collection_wall_seconds'] = time.monotonic() - start
            for item in plan:
                baseline = next((saved for saved in plan if saved['case'] == item['case'] and saved['phase'] == 'baseline' and saved['status'] == 'ok'), None)
                control = next((saved for saved in plan if saved['case'] == item['case'] and saved['phase'] == 'control' and saved['status'] == 'ok'), None)
                if baseline is not None and control is not None:
                    item['unmarked_instrumentation_delta_percent'] = 100 * (float(control['task_duration_us']) / float(baseline['task_duration_us']) - 1)
            save_csv(output, plan)
            tables = folder / 'tables'
            if tables.is_dir():
                shutil.rmtree(tables)
            tables.mkdir(exist_ok=True)
            for index, table in enumerate(sorted(scratch.rglob('*.csv'))):
                if re.fullmatch(r'(OpBasicInfo|PipeUtilization|Memory|ArithmeticUtilization)_\d{17}\.csv', table.name):
                    shutil.copy2(table, tables / f'{index}_{table.name}')
            shutil.rmtree(scratch)
            for path in (folder / 'output').glob('*.bin'):
                path.unlink()
    ok = sum(row['status'] == 'ok' for row in plan)
    print(f'CSV={output}\nCOMPLETED={ok}/{len(plan)}; elapsed_seconds={LIMIT_SECONDS - max(0, deadline - time.monotonic()):.1f}', flush=True)
    if ok != len(plan):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
