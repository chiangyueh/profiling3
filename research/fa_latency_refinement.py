from __future__ import annotations
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from prediction.fa_latency_refinement import (
    CompressedGraphArray,
    compressed_array,
    signature,
    matmul_microtiles,
    refine_matmul_job,
    execution_conditions,
    Work,
    workload,
    batch_program,
    s1_program,
    general_program,
    varlen_gm_program,
    release_graph_pages,
    graph_time,
    BranchGraph,
    branch_graph,
    primitive_class,
    primitive_pools,
    graph_entry,
    predict_work,
    predict,
    refinement_status,
    micro_context,
    critical_pipe_times,
    predict_selected,
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from research import fa_branch_latency as base
source = base.source
ROOT = Path(__file__).resolve().parents[1]
VERSION = "fa_branch_execution_v4"
MODEL = ROOT / "output/fa_micro_candidate.json"
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
records = base.records
_summary = base._summary






def prior_for(priors, pipe, key):
    candidates = primitive_pools(pipe, key)
    for candidate in candidates:
        entry = priors.get(candidate)
        if entry is not None and (len(entry['inputs']) >= len(entry['observed']) or candidate == candidates[-1]):
            return candidate, entry
    return None, None


def primitive_priors(samples):
    from scipy.optimize import nnls
    pooled=defaultdict(list)
    for row in samples:
        for pipe in PIPES:
            if row['targets'].get(pipe,0)>0:
                for candidate in primitive_pools(pipe, row['cost_class']):
                    pooled[candidate].append(row)
    result={}
    for key,rows in pooled.items():
        pipe=key[0];names=PIPES[pipe][1]
        x=np.array([[r['features'][pipe].get(n,0) for n in names] for r in rows],dtype=float)
        y=np.array([r['targets'][pipe] for r in rows],dtype=float)
        scales=np.maximum(np.max(x/y[:,None],axis=0),1e-30)
        active=np.any(x>0,axis=0)
        rates=np.zeros(len(names))
        if np.any(active):
            weights=np.sqrt(np.array([r.get('pmu_weight',1) for r in rows]))
            solution,_=nnls((x/y[:,None]/scales)[:,active]*weights[:,None],weights,maxiter=10000)
            rates[active]=solution/scales[active]
        bounds=np.full(len(names),np.inf)
        if pipe in ('cscalar','vscalar'):
            for i,name in enumerate(names):
                good=x[:,i]>0
                if np.any(good):bounds[i]=np.min(y[good]/x[good,i])
        result[key]={'names':names,'rates':rates,'upper':bounds,'observed':[n for n,a in zip(names,active) if a],
                     'inputs':sorted({r['input_sha256'] for r in rows}),'matrix':x,'targets':y}
    return result


def fit_costs(samples, hardware, class_keys=None, fit_inputs=None, initial_classes=None):
    from scipy.optimize import least_squares
    names=parameter_names();positions={key:i for i,key in enumerate(names)}
    priors=primitive_priors(samples)
    grouped=defaultdict(list)
    for row in samples:
        if fit_inputs is None or row['input_sha256'] in fit_inputs:
            grouped[row['cost_class']].append(row)
    keys=set(grouped) if class_keys is None else set(grouped).intersection(class_keys)
    model={'version':VERSION,'source_sha256':signature(),'hardware':hardware,'classes':{},
           'latency_used_for_cost_fit':True,'calibration_target':'training kernel latency and training PMU; Scalar PMU is an upper constraint',
           'coefficient_interpretation':'nonnegative operation costs on branch-specific dependency graphs; no per-shape lookup or total-time correction factor'}
    for key in sorted(keys):
        rows=grouped[key]
        initial=np.zeros(len(names));scales=np.ones(len(names));upper=np.full(len(names),np.inf)
        pipe_info={}
        for pipe,(_,fields) in PIPES.items():
            parent_key,parent=prior_for(priors,pipe,key)
            if parent is None:continue
            local=np.array([[r['features'][pipe].get(n,0) for n in fields] for r in rows],dtype=float)
            seen=np.any(parent['matrix']!=0,axis=0)
            rank=int(np.linalg.matrix_rank(parent['matrix'][:,seen]/np.maximum(np.max(parent['matrix'][:,seen],axis=0),1e-30))) if any(seen) else 0
            pipe_info[pipe]={'rank':rank,'terms':int(sum(seen)),'samples':len(parent['inputs']),
                             'observed_features':list(parent['observed']),'primitive_calibration_class':list(parent_key),
                             'primitive_calibration_input_count':len(parent['inputs'])}
            budget=statistics.median(parent['targets'])/max(1,sum(seen))
            maxima=np.maximum(np.max(parent['matrix'],axis=0),1e-12)
            for j,n in enumerate(fields):
                i=positions[pipe,n];initial[i]=parent['rates'][j];scales[i]=max(initial[i],budget/maxima[j],1e-12)
                if pipe in ('cscalar','vscalar'):
                    if fit_inputs is None:
                        upper[i]=parent['upper'][j]
                    else:
                        values=[row['targets'][pipe]/row['features'][pipe][n] for row in rows if row['targets'][pipe]>0 and row['features'][pipe].get(n,0)>0]
                        if values:upper[i]=min(values)
        if initial_classes and key in initial_classes:
            initial=np.array(initial_classes[key]['rates'],dtype=float)
        active=np.flatnonzero(np.any(np.vstack([r['graph']['templates'] for r in rows])!=0,axis=0))
        limits=upper[active]/scales[active]
        start=np.minimum(initial[active]/scales[active],limits*.95)
        timing=np.array([r['duration_us'] for r in rows])
        matrices=[];upper_only=[];counter_weights=[]
        for row in rows:
            for pipe in PIPES:
                target=row['targets'].get(pipe,0)
                if target<=0:continue
                vec=np.zeros(len(names))
                for n,value in row['features'][pipe].items():
                    if (pipe,n) in positions:vec[positions[pipe,n]]=value/target
                matrices.append(vec);upper_only.append(pipe in ('cscalar','vscalar'));counter_weights.append(math.sqrt(row.get('pmu_weight',1)/len(PIPES)))
        counters=np.array(matrices);upper_only=np.array(upper_only);counter_weights=np.array(counter_weights)
        scalar_rows={}
        for pipe in ('cscalar','vscalar'):
            fields=PIPES[pipe][1];indexes=[positions[pipe,n] for n in fields]
            scalar_rows[pipe]=(indexes,np.array([[r['features'][pipe].get(n,0)/r['targets'][pipe] for n in fields] for r in rows if r['targets'].get(pipe,0)>0]))
        def effective_rates(parameters):
            rates=np.zeros(len(names));rates[active]=parameters*scales[active]
            derivative=np.zeros((len(names),len(active)));derivative[active,np.arange(len(active))]=scales[active]
            for indexes,matrix in scalar_rows.values():
                if not len(matrix):continue
                ratios=matrix@rates[indexes];at=int(np.argmax(ratios));ratio=ratios[at]
                if ratio>1:
                    original=rates[indexes].copy();normalization=matrix[at]@derivative[indexes]
                    derivative[indexes]=derivative[indexes]/ratio-np.outer(original,normalization)/ratio**2
                    rates[indexes]=original/ratio
            return rates,derivative
        last={}
        def residual(parameters):
            if 'x' in last and np.array_equal(parameters,last['x']):return last['r']
            rates,rate_jacobian=effective_rates(parameters)
            predictions=[];jac=[]
            for row in rows:
                val,deriv=graph_time(row['graph'],rates,True);predictions.append(val);jac.append(deriv@rate_jacobian)
            times=np.array(predictions)/timing-1
            time_jac=np.array(jac)/timing[:,None]
            pmu=counters@rates-1;pmu_jac=counters@rate_jacobian
            disabled=upper_only&(pmu<=0);pmu[disabled]=0;pmu_jac[disabled]=0
            weight=counter_weights
            regularization=1/math.sqrt(max(1,len(rows)))
            values=np.concatenate((times,weight*pmu,regularization*(parameters-start)))
            derivatives=np.vstack((time_jac,weight[:,None]*pmu_jac,regularization*np.eye(len(active))))
            last.update(x=parameters.copy(),r=values,j=derivatives)
            return values
        optimum=least_squares(residual,start,jac=lambda x:(residual(x),last['j'])[1],bounds=(np.zeros(len(active)),limits),
                              loss='soft_l1',f_scale=.05,max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        rates,_=effective_rates(optimum.x)
        pipes={}
        for pipe,(_,fields) in PIPES.items():
            data=pipe_info.get(pipe,{'rank':0,'terms':0,'samples':0,'observed_features':[]})
            data['coefficients']={n:float(rates[positions[pipe,n]]) for n in fields};pipes[pipe]=data
        model['classes'][key]={'pipes':pipes,'rates':rates.tolist(),'calibration_inputs':[r['input_sha256'] for r in rows],
                               'optimization_success':bool(optimum.success),'optimization_evaluations':int(optimum.nfev),
                               'scalar_counter_scope':'strict training inequality: useful issue service <= measured busy including polling',
                               'parameter_names':[list(x) for x in names]}
        model['classes'][key]['local_identifiability'] = {pipe: feature_identifiability(rows, pipe) for pipe in PIPES}
        support = defaultdict(lambda: defaultdict(int))
        for row in rows:
            for name, value in row.get('conditions', {}).items():
                support[name][value] += 1
        model['classes'][key]['source_condition_support'] = {name: dict(values) for name, values in support.items()}
        contexts = [micro_context(row['input_features']) for row in rows]
        model['classes'][key]['micro_context_ranges'] = {name: [min(ctx[name] for ctx in contexts), max(ctx[name] for ctx in contexts)] for name in contexts[0]}
        print(f"MICRO FIT {key}: samples={len(rows)} evaluations={optimum.nfev} converged={optimum.success}", flush=True)
    for parent_key,parent in priors.items():
        if parent_key[0] not in ('cscalar','vscalar'):continue
        selected=[(entry,len(entry['calibration_inputs'])) for key,entry in model['classes'].items() if primitive_class(parent_key[0],key)==parent_key]
        for index,name in enumerate(parent['names']):
            values=sorted((entry['pipes'][parent_key[0]]['coefficients'][name],weight) for entry,weight in selected)
            if values:
                middle=sum(weight for _,weight in values)/2;accumulated=0
                for value,weight in values:
                    accumulated+=weight
                    if accumulated>=middle:
                        parent['rates'][index]=value
                        break
    model['primitive_priors']={json.dumps(key):{'rates':entry['rates'].tolist(),'names':list(entry['names']),
                                             'observed_features':entry['observed'],'calibration_inputs':entry['inputs'],
                                             'rank':int(np.linalg.matrix_rank(entry['matrix'][:,np.any(entry['matrix']>0,axis=0)])),'terms':len(entry['observed'])}
                               for key,entry in priors.items()}
    return model


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def prepare_graph_samples(rows, hardware):
    import zlib
    folder = ROOT / 'out/fa_branch_host'
    folder.mkdir(parents=True, exist_ok=True)
    binary, manifest = folder / 'micro_graphs.bin', folder / 'micro_graphs.json'
    identity = source.cost.sha256({'inputs_and_measurements': rows, 'hardware': hardware, 'api': tiler()[1], 'base_source':base.signature(),
                                  'version': VERSION, 'primitives': source.source_hash(),
                                  'functions': [inspect.getsource(f) for f in
                                      (matmul_microtiles, refine_matmul_job, Work, matmul_cache_trace,
                                       workload, batch_work, s1_work, general_work, varlen_work, adapter_jobs,
                                       active_features, BranchGraph, branch_graph)]})
    saved = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if saved.get('identity') != identity or not binary.is_file():
        temporary = binary.with_suffix('.building')
        samples, rejected = [], []
        with temporary.open('wb') as stream:
            def save_graph(graph):
                result = {k: v for k, v in graph.items() if k not in ('a', 'b', 'which', 'templates')}
                result['arrays'] = {}
                for name in ('a', 'b', 'which', 'templates'):
                    array = np.ascontiguousarray(graph[name])
                    packed = zlib.compress(memoryview(array).cast('B'), level=1)
                    offset = stream.tell()
                    stream.write(packed)
                    result['arrays'][name] = {'offset': offset, 'shape': array.shape, 'dtype': str(array.dtype),
                                             'compressed_bytes': len(packed), 'raw_bytes': array.nbytes}
                return result
            for index, row in enumerate(rows, 1):
                try:
                    work = workload(row['inputs'], hardware)
                    if int(row['priority']) != work.s['priority']:
                        raise ValueError('input-derived route differs from official Host trace')
                    actual, differences = [], []
                    for stage, mm in enumerate(work.mm, 1):
                        fields = {n: int(row[f'bmm{stage}_{n}']) for n in mm if _number(row.get(f'bmm{stage}_{n}')) is not None}
                        if any(fields.get(n, mm[n]) != mm[n] for n in ('base_m', 'base_n', 'base_k')):
                            raise ValueError('input-derived Matmul base differs from official Host')
                        differences.extend(f'bmm{stage}_{n}:{mm[n]}!={v}' for n, v in fields.items() if mm[n] != v)
                        actual.append(fields)
                    input_graph = save_graph(branch_graph(work))
                    input_features = active_features(work)
                    conditions = execution_conditions(work)
                    if differences:
                        work = workload(row['inputs'], hardware, actual)
                        graph = save_graph(branch_graph(work))
                        features = active_features(work)
                    else:
                        graph, features = input_graph, input_features
                    samples.append({'input_sha256': row['input_sha256'], 'sequence_group': row['sequence_group'],
                                    'inputs': row['inputs'], 'case': row['case'],
                                    'collection_group': row.get('collection_group') or row['family'],
                                    'cost_class': cost_class(work.p, work.s), 'features': features,
                                    'input_features': input_features, 'conditions': conditions,
                                    'duration_us': float(row['duration_us']),
                                    'targets': {pipe: float(row[target]) for pipe, (target, _) in PIPES.items()},
                                    'graph': graph, 'input_graph': input_graph, 'geometry_differences': differences})
                except (ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
                    rejected.append({'case': row['case'], 'input_sha256': row['input_sha256'], 'error': str(error)})
                if index % 25 == 0:
                    print(f'MICRO GRAPHS {index}/{len(rows)} accepted={len(samples)} compressed_mb={stream.tell()/1024**2:.1f}', flush=True)
        temporary.replace(binary)
        saved = {'identity': identity, 'samples': samples, 'rejected': rejected,
                 'format': 'per-array lossless zlib, bounded 320 MiB decompression cache'}
        source.cost.save_json(manifest, saved)
    else:
        print(f'MICRO GRAPH CACHE HIT samples={len(saved["samples"])}', flush=True)
    condition_signature = hashlib.sha256(inspect.getsource(execution_conditions).encode()).hexdigest()
    if saved.get('condition_signature') != condition_signature:
        for item in saved['samples']:
            item['conditions'] = execution_conditions(workload(item['inputs'], hardware))
        saved['condition_signature'] = condition_signature
        source.cost.save_json(manifest,saved)
        print('MICRO SOURCE-CONDITION METADATA UPDATED; timing graphs reused',flush=True)
    measurements, result = {r['input_sha256']: r for r in rows}, []
    for row in saved['samples']:
        item = {**row}
        measured = measurements[row['input_sha256']]
        clocks = [float(measured[n]) for n in ('aic_time_max_us', 'aiv_time_max_us') if _number(measured.get(n)) is not None]
        interval = [float(measured[n]) for n in ('duration_us', 'round1_us', 'round2_us', 'round3_us', 'median_ci95_high_us')
                    if (_number(measured.get(n)) or 0) > 0]
        reference = max(interval)
        item['pmu_time_ratio'] = max(clocks, default=reference) / reference
        item['pmu_weight'] = min(1, 1 / item['pmu_time_ratio'])**2
        low,high=repeat_interval(measured)
        item['repeat_noise_percent']=100*(high-low)/float(measured['duration_us'])
        for key in ('graph', 'input_graph'):
            description = row[key]
            graph = {k: v for k, v in description.items() if k != 'arrays'}
            graph.update({n: CompressedGraphArray(binary, d) for n, d in description['arrays'].items()})
            item[key] = graph
        result.append(item)
    return result, saved['rejected']


def build_and_evaluate(paths, evaluation_path, output_dir, model_path=MODEL, reference_path=None):
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    inventory=[];historical,hardware=records(paths,inventory);evaluation,observed=records([evaluation_path],inventory)
    if hardware!=observed:raise ValueError('calibration and evaluation hardware differ')
    raw={r['input_sha256']:r for r in historical+evaluation}
    prepared,rejected=prepare_graph_samples(list(raw.values()),hardware)
    indexed={r['input_sha256']:r for r in prepared};results=[]
    for fold in range(4):
        training=[r for r in prepared if int(r['sequence_group'][:8],16)%4!=fold]
        testing=[r for r in evaluation if int(r['sequence_group'][:8],16)%4==fold]
        print(f'BRANCH FIT fold={fold+1}/4 training={len(training)} evaluation={len(testing)}',flush=True)
        model=fit_refined_costs(training,hardware)
        model['sdk_api_sha256']=tiler()[1]
        training_groups={r['sequence_group'] for r in training}
        for row in testing:
            assert row['sequence_group'] not in training_groups
            result={**row,'branch_status':'unavailable','branch_evaluation':'complete sequence-pair heldout, source graph and training-only primitive calibration',
                    'branch_predicted_us':None,'branch_ape_percent':None,'branch_error':'','branch_fold':fold,
                    'total_latency_used_for_fit':True,'heldout_latency_used_for_fit':False,'branch_model_sha256':source.cost.sha256(model)}
            if str(row.get('prediction_before_measurement','')).lower()=='true' and row.get('predicted_us'):
                result.update(original_frozen_predicted_us=float(row['predicted_us']),original_frozen_model_sha256=row.get('model_sha256'),
                              original_frozen_ape_percent=100*abs(float(row['predicted_us'])/float(row['duration_us'])-1))
            try:
                sample=indexed[row['input_sha256']];entry=graph_entry(model,sample['cost_class'],sample['conditions'])
                result.update(branch_pmu_time_ratio=sample['pmu_time_ratio'],branch_pmu_calibration_weight=sample['pmu_weight'])
                for pipe,values in sample['input_features'].items():
                    missing=[n for n,v in values.items() if v>0 and n not in entry['pipes'][pipe]['observed_features']]
                    if missing:raise ValueError('unmeasured source primitive '+pipe+'/'+','.join(missing))
                val,gradient=graph_time(sample['input_graph'],entry['rates'],True);replay=graph_time(sample['graph'],entry['rates'])
                if not math.isfinite(val) or val<=0:raise ValueError('invalid source graph prediction')
                critical=source.critical_phases(sample['input_graph'],entry['rates']);release_graph_pages(sample['input_graph'])
                result.update(branch_status='predicted',branch_predicted_us=val,branch_ape_percent=100*abs(val/float(row['duration_us'])-1),
                              branch_declared_tiling_predicted_us=replay,branch_declared_tiling_ape_percent=100*abs(replay/float(row['duration_us'])-1),
                              branch_implementation=sample['cost_class'].split('|')[0],branch_cost_class=sample['cost_class'],
                              branch_unidentified_pipe_terms='|'.join(p for p,x in entry['pipes'].items() if x['rank']<x['terms']),
                              branch_direct_class_samples=len(entry['calibration_inputs']),
                              branch_adapter_us=critical.get('drop_adapter',0),branch_startup_us=critical.get('startup',0),
                              branch_critical_path_phase_us=json.dumps(critical,sort_keys=True),
                              branch_critical_path_pipe_us=json.dumps(critical_pipe_times(gradient,entry['rates']),sort_keys=True),
                              branch_matmul_geometry_differences='|'.join(sample['geometry_differences']))
                for pipe,values in sample['input_features'].items():
                    estimated=sum(v*entry['pipes'][pipe]['coefficients'].get(n,0) for n,v in values.items());actual=sample['targets'][pipe]
                    result[f'branch_{pipe}_active_sum_us']=estimated
                    result[f'branch_{pipe}_ape_percent']=100*abs(estimated/actual-1) if actual>0 else None
                    if pipe in ('cscalar','vscalar'):result[f'branch_{pipe}_polling_and_unmodeled_busy_us']=actual-estimated
            except (ValueError,RuntimeError,KeyError) as error:result['branch_error']=str(error)
            result.pop('inputs',None);results.append(result)
        print(f'BRANCH EVALUATED {len(results)}/{len(evaluation)}',flush=True)
    if reference_path is not None and Path(reference_path).is_file():
        with Path(reference_path).open(newline='') as stream:
            reference = {row['input_sha256']:row for row in csv.DictReader(stream)}
        for result in results:
            old = reference.get(result['input_sha256'],{})
            if _number(old.get('branch_predicted_us')) is not None:
                result['reference_predicted_us'] = float(old['branch_predicted_us'])
                result['reference_ape_percent'] = float(old['branch_ape_percent'])
                result['reference_evaluation'] = old.get('branch_evaluation','')
    final_model=fit_refined_costs(prepared,hardware)
    final_model.update(training_inputs=sorted(r['input_sha256'] for r in prepared),sdk_api_sha256=tiler()[1],source_files=FILES,
                       validation='four folds by all inputs sharing complete (S1,S2); all calibration priors exclude heldout groups',
                       calibration_geometry='recorded Host fields; CPU API fills fields not present in the exported trace',
                       inference_calibration='immutable artifact; inputs and CPU tiling only; no runtime refitting',
                       scope='dense FA and uniform TND; per-implementation controller, per-buffer hazards, operand cache and engine queues',
                       stage_costs_independently_measured=False,qualification='cross-validated source-operation model; accuracy is reported per family and input',
                       optimizer_policy='nonnegative operation rates; Scalar PMU upper constraints; source-matched primitive priors; no per-shape lookup or total-time scale factor')
    output_dir.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in results for k in r))
    with (output_dir/'branch_predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(sorted(results,key=lambda r:r['case']))
    report={'groups':{},'source_sha256':signature(),'feature_failures':rejected,'new_npu_measurements':0,
            'kernel_latency_used_for_cost_fit':True,'heldout_latency_used_for_fit':False,'development_replay_after_inspection':True,
            'hardware':hardware,'data_inventory':inventory,'evaluation':final_model['validation']}
    for group in sorted({r.get('collection_group') or r['family'] for r in results}):
        selected=[r for r in results if (r.get('collection_group') or r['family'])==group]
        report['groups'][group]=_summary(selected)
        old_errors = [row['reference_ape_percent'] for row in selected if 'reference_ape_percent' in row]
        if old_errors:
            report['groups'][group]['reference'] = {'cases':len(old_errors),'median_ape_percent':statistics.median(old_errors),
                'above10_count':sum(value>10 for value in old_errors),'above20_count':sum(value>20 for value in old_errors),
                'above30_count':sum(value>30 for value in old_errors)}
        report['groups'][group]['by_layout']={layout:_summary([r for r in selected if r['layout']==layout]) for layout in sorted({r['layout'] for r in selected})}
        summary=report['groups'][group]
        print(f"BRANCH RESULT {group}: predicted={summary['predicted']}/{summary['cases']} MdAPE={summary['median_ape_percent']}% "+' '.join(f'>{limit}%={v["count"]}/{summary["predicted"]}' for limit,v in summary['above'].items()),flush=True)
    final_model['micro_cost_contract'] = micro_cost_contract()
    final_model['refinement_candidates'] = refinement_analysis(results,prepared)
    final_model['refinement_evidence'] = 'development sequence-group cross-validation; suggestions do not activate additional splits'
    final_model['cross_validation']=report['groups'];source.cost.save_json(model_path,final_model);source.cost.save_json(output_dir/'branch_evaluation.json',report)
    write_model_audit(results,final_model,report,ROOT/'research' / 'reports' / 'fa_branch_audit.csv')
    print(f'BRANCH_MODEL={model_path}\nBRANCH_PREDICTIONS={output_dir/"branch_predictions.csv"}',flush=True)
    return report


def feature_identifiability(rows, pipe):
    fields = PIPES[pipe][1]
    matrix = np.array([[row['features'][pipe].get(name, 0) for name in fields] for row in rows], dtype=float)
    active = np.any(matrix != 0, axis=0)
    if not np.any(active):
        return {'rank': 0, 'terms': 0, 'condition_number': None, 'identifiable': True}
    normalized = matrix[:, active] / np.maximum(np.max(np.abs(matrix[:, active]), axis=0), 1e-30)
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.linalg.matrix_rank(normalized))
    condition = float(singular[0] / singular[rank-1]) if rank else None
    return {'rank': rank, 'terms': int(sum(active)), 'condition_number': condition,
            'identifiable': rank == int(sum(active)), 'parameter_names': [n for n, a in zip(fields, active) if a]}


def repeat_interval(row):
    rounds = [_number(row.get(name)) for name in ('round1_us', 'round2_us', 'round3_us')]
    rounds = [value for value in rounds if value is not None and value > 0]
    low = _number(row.get('ci95_low_us')) or _number(row.get('median_ci95_low_us'))
    high = _number(row.get('ci95_high_us')) or _number(row.get('median_ci95_high_us'))
    actual = float(row['duration_us'])
    return min(rounds + ([low] if low else []) + [actual]), max(rounds + ([high] if high else []) + [actual])


def refinement_analysis(results, samples):
    from scipy.stats import binomtest
    indexed = {row['input_sha256']: row for row in samples}
    pooled = defaultdict(list)
    for row in results:
        if row.get('branch_predicted_us') is None:
            continue
        sample = indexed[row['input_sha256']]
        for name, value in sample['conditions'].items():
            pooled[sample['cost_class'], name, value].append(row)
    candidates = []
    for (key, name, value), rows in sorted(pooled.items()):
        parent = [row for row in results if row.get('branch_cost_class') == key]
        if len(rows) == len(parent):
            continue
        signed = [100 * (row['branch_predicted_us'] / float(row['duration_us']) - 1) for row in rows]
        group_signs, group_residuals = defaultdict(list), defaultdict(list)
        outside = stable_outside = 0
        for row, residual in zip(rows, signed):
            lo, hi = repeat_interval(row)
            prediction = row['branch_predicted_us']
            exceeds = prediction < lo or prediction > hi
            outside += exceeds
            stable_outside += exceeds and str(row.get('stable')).lower() == 'true'
            group_residuals[row['sequence_group']].append(residual)
            if exceeds:
                group_signs[row['sequence_group']].append(1 if prediction > hi else -1)
        signs = [statistics.median(values) for values in group_signs.values()]
        signs = [value for value in signs if value != 0]
        positive, negative = sum(value > 0 for value in signs), sum(value < 0 for value in signs)
        p = float(binomtest(positive, positive + negative, .5).pvalue) if signs else 1.0
        group_errors = [statistics.median(values) for values in group_residuals.values()]
        record = {'cost_class': key, 'condition': name, 'value': value, 'cases': len(rows),
                  'sequence_groups': len(group_residuals), 'median_signed_error_percent': statistics.median(signed),
                  'median_group_signed_error_percent': statistics.median(group_errors),
                  'median_ape_percent': statistics.median(map(abs, signed)),
                  'above10_count': sum(abs(v) > 10 for v in signed), 'above20_count': sum(abs(v) > 20 for v in signed),
                  'outside_repeat_interval_count': outside, 'stable_outside_repeat_interval_count': stable_outside,
                  'positive_independent_groups': positive, 'negative_independent_groups': negative,
                  'sign_test_p': p, 'split_validation': 'not_run; observation is a candidate, not approval for a new branch',
                  'evidence_scope': 'development cross-validation; source-condition cohort, not independent stage wall times'}
        if not signs:
            record['suggested_action'] = 'measurement_intervals_cover_predictions'
        elif len(signs) < 2:
            record['suggested_action'] = 'insufficient_independent_sequence_groups'
        else:
            record['suggested_action'] = 'compare_cost_and_timeline_for_this_source_condition'
        candidates.append(record)
    order = sorted(range(len(candidates)), key=lambda i: candidates[i]['sign_test_p'])
    adjusted = 1.0
    for position in range(len(order)-1, -1, -1):
        index = order[position]
        adjusted = min(adjusted, candidates[index]['sign_test_p'] * len(order) / (position + 1))
        candidates[index]['sign_test_fdr_q'] = min(1.0, adjusted)
        candidates[index]['systematic_bias_evidence'] = adjusted < .05
    return candidates


def write_model_audit(results, model, report, path):
    output = []
    for family, summary in report['groups'].items():
        output.append({'record_type': 'summary', 'family': family, 'cases': summary['cases'],
                       'predicted': summary['predicted'], 'median_ape_percent': summary['median_ape_percent'],
                       'above10_count': summary['above']['10']['count'], 'above20_count': summary['above']['20']['count'],
                       'above30_count': summary['above']['30']['count'], 'max_ape_percent': summary['max_ape_percent'],
                       'evaluation': model['validation']})
    for row in results:
        item = {'record_type': 'kernel_case', 'family': row.get('collection_group') or row['family'], 'case': row['case'],
                'input_sha256': row['input_sha256'], 'actual_us': row['duration_us'], 'predicted_us': row.get('branch_predicted_us'),
                'ape_percent': row.get('branch_ape_percent'), 'cost_class': row.get('branch_cost_class'),
                'fold': row['branch_fold'], 'status': row['branch_status'], 'error': row.get('branch_error'),
                'heldout_latency_used_for_fit': False, 'direct_class_samples': row.get('branch_direct_class_samples'),
                'reference_predicted_us':row.get('reference_predicted_us'), 'reference_ape_percent':row.get('reference_ape_percent'),
                'critical_path_pipe_us':row.get('branch_critical_path_pipe_us')}
        item.update({name: row[name] for name in source.cost.INPUT_FIELDS})
        if row.get('branch_predicted_us'):
            item['signed_error_percent'] = 100 * (row['branch_predicted_us'] / float(row['duration_us']) - 1)
            lo, hi = repeat_interval(row)
            item.update(measurement_interval_low_us=lo, measurement_interval_high_us=hi,
                        outside_repeat_interval=row['branch_predicted_us'] < lo or row['branch_predicted_us'] > hi)
        for pipe in PIPES:
            item[f'{pipe}_service_ape_percent'] = row.get(f'branch_{pipe}_ape_percent')
        output.append(item)
    for candidate in model.get('refinement_candidates', []):
        output.append({'record_type': 'refinement_candidate', **candidate})
    for key, entry in model['classes'].items():
        for pipe, stats in entry.get('local_identifiability', {}).items():
            output.append({'record_type': 'cost_identifiability', 'cost_class': key, 'pipe': pipe,
                           'cost_rank': stats['rank'], 'cost_terms': stats['terms'],
                           'normalized_condition_number': stats['condition_number'], 'independently_identifiable': stats['identifiable']})
    for decision in model.get('source_refinement_decisions',[]):
        output.append({'record_type':'automatic_source_split',**decision})
    for name, rule in model['micro_cost_contract'].items():
        output.append({'record_type': 'source_rule', 'condition': name, 'note': rule})
    fields = list(dict.fromkeys(name for row in output for name in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(output)


def micro_cost_contract():
    return {'microtile_arithmetic': 'MMAD M/N/K from Matmul loops; real tail dimensions; 16-element compute padding; first K accumulator setup separated from continuation',
            'pipe_m_barrier': 'SDK MmadCompute: floor(m/16)*floor(n/16)<10 inserts PIPE_M; m=1 is fixed to 16 for non-GEMV Matmul. Barrier synchronization is scoped to PIPE_M, not a global Scalar/MTE stall; its issue is counted explicitly.',
            'batch_preload': 'B IterateBatch: BatchLoad A/B completes before the inner batch MMAD loop. After each output, BatchScheduler sets/waits M_MTE1, so the next MTE1 operation waits for that MMAD output; independent L0 ping-pong does not remove this explicit event.',
            'manual_l1': 'Varlen L1: MM1 partitions N by 256; MM2 partitions K by 256, accumulates, then GetTensorC. Varlen SameAB L1: MM2 partitions K by 192 and output N by 128.',
            'scalar_scope': 'commands count source API steps, MMAD issue and conditional PIPE_M issue; calibrated useful issue service is bounded by measured Scalar busy including polling',
            'pipe_cost_context': 'batch, manual L1, direct SameAB and KFC callers have distinct empirical primitive priors when supported; sparse contexts borrow the matching instruction primitive',
            'stage_scope': 'BMM1/Vec1/BMM2/Vec2 critical-path contributions are modeled; normal PMU contains whole-kernel pipe service sums. Independent undisturbed stage wall times remain unavailable.',
            'refinement_policy': 'Refinement candidates are diagnosed by source conditions and sequence-group residual sign tests with FDR correction. No new split is activated merely because a single shape has a large error.',
            'measurement_reuse': 'Existing measurements only; four complete S1/S2 group folds; no runtime coefficient fitting, no measured target input to predict, no per-shape latency lookup'}


def fit_refined_costs(samples, hardware, class_keys=None):
    model = fit_costs(samples, hardware, class_keys=class_keys)
    grouped = defaultdict(list)
    for row in samples:
        if row['cost_class'] in model['classes']:
            grouped[row['cost_class']].append(row)
    decisions = []
    for key, rows in sorted(grouped.items()):
        if len({row['sequence_group'] for row in rows}) < 6:
            continue
        parent = model['classes'][key]
        residual = {row['input_sha256']: 100*(graph_time(row['graph'],parent['rates'])/row['duration_us']-1) for row in rows}
        candidates = []
        for condition in sorted(rows[0]['conditions']):
            values = sorted({row['conditions'][condition] for row in rows})
            if len(values) != 2:
                continue
            children = {value:[row for row in rows if row['conditions'][condition]==value] for value in values}
            if any(len(child)<4 or len({row['sequence_group'] for row in child})<3 for child in children.values()):
                continue
            medians = [statistics.median(residual[row['input_sha256']] for row in children[value]) for value in values]
            contrast = abs(medians[0]-medians[1])
            noise = statistics.median(row.get('repeat_noise_percent',0) for row in rows)
            if contrast <= noise:
                continue
            score = contrast * min(map(len,children.values())) / len(rows)
            candidates.append((score,condition,values))
        if not candidates:
            continue
        _, condition, values = max(candidates)
        parent_errors, child_errors = [], []
        parent_service, child_service = [], []
        supported = True
        for fold in (0,1):
            train = [row for row in samples if int(row['sequence_group'][8:16],16)%2!=fold]
            test = [row for row in rows if int(row['sequence_group'][8:16],16)%2==fold]
            if not test:
                supported=False
                break
            baseline = fit_costs(train,hardware,class_keys={key})
            if key not in baseline['classes']:
                supported=False
                break
            fitted = {}
            for value in values:
                selected = {row['input_sha256'] for row in train if row['cost_class']==key and row['conditions'][condition]==value}
                sequences = {row['sequence_group'] for row in train if row['input_sha256'] in selected}
                if len(selected)<3 or len(sequences)<2:
                    supported=False
                    break
                child = fit_costs(train,hardware,class_keys={key},fit_inputs=selected,initial_classes=baseline['classes'])
                fitted[value]=child['classes'][key]
            if not supported:
                break
            for row in test:
                entry=fitted[row['conditions'][condition]]
                parent_errors.append(100*abs(graph_time(row['input_graph'],baseline['classes'][key]['rates'])/row['duration_us']-1))
                child_errors.append(100*abs(graph_time(row['input_graph'],entry['rates'])/row['duration_us']-1))
                for pipe in PIPES:
                    if pipe in ('cscalar','vscalar') or row['targets'][pipe]<=0:continue
                    target=row['targets'][pipe]
                    base_service=sum(v*baseline['classes'][key]['pipes'][pipe]['coefficients'].get(n,0) for n,v in row['features'][pipe].items())
                    refined_service=sum(v*entry['pipes'][pipe]['coefficients'].get(n,0) for n,v in row['features'][pipe].items())
                    weight=row.get('pmu_weight',1)
                    parent_service.append((100*abs(base_service/target-1),weight))
                    child_service.append((100*abs(refined_service/target-1),weight))
        if not supported or not parent_errors:
            continue
        gain = statistics.median(parent_errors)-statistics.median(child_errors)
        noise = statistics.median(row.get('repeat_noise_percent',0) for row in rows)/2
        def weighted_median(values):
            cumulative=0;middle=sum(weight for _,weight in values)/2
            for value,weight in sorted(values):
                cumulative+=weight
                if cumulative>=middle:return value
        parent_service_error=weighted_median(parent_service)
        child_service_error=weighted_median(child_service)
        accepted = (gain>noise and np.quantile(child_errors,.9)<=np.quantile(parent_errors,.9)
                    and sum(value>20 for value in child_errors)<=sum(value>20 for value in parent_errors)
                    and child_service_error<=parent_service_error)
        decision={'cost_class':key,'condition':condition,'inner_validation_cases':len(parent_errors),
                  'parent_median_ape_percent':statistics.median(parent_errors),'child_median_ape_percent':statistics.median(child_errors),
                  'parent_p90_ape_percent':float(np.quantile(parent_errors,.9)),'child_p90_ape_percent':float(np.quantile(child_errors,.9)),
                  'parent_above20_count':sum(value>20 for value in parent_errors),'child_above20_count':sum(value>20 for value in child_errors),
                  'measurement_noise_halfspan_percent':noise,'accepted':bool(accepted),
                  'parent_pipe_service_median_ape_percent':parent_service_error,'child_pipe_service_median_ape_percent':child_service_error,
                  'selection_scope':'all selection and inner fits are inside the outer training groups; source-condition screen uses outer training residuals'}
        decisions.append(decision)
        print('SOURCE REFINEMENT '+key+' condition='+condition+' accepted='+str(accepted)+' median_gain='+str(round(gain,3)),flush=True)
        if not accepted:
            continue
        children={}
        for value in values:
            selected={row['input_sha256'] for row in rows if row['conditions'][condition]==value}
            child=fit_costs(samples,hardware,class_keys={key},fit_inputs=selected,initial_classes={key:parent})
            children[value]=child['classes'][key]
        parent['source_refinement']={'condition':condition,'children':children,'inner_validation':decision}
    model['source_refinement_decisions']=decisions
    model['source_refinement_policy']='at most one input-derived source-condition split per class; inner heldout median improves beyond repeat halfspan, p90 and >20% count do not worsen; outer sequence-group evaluation remains separate'
    return model


def select_source_models(baseline_model, refined_model, validation_rows):
    decisions = {}
    for impl in sorted({row['branch_implementation'] for row in validation_rows}):
        rows = [row for row in validation_rows if row['branch_implementation']==impl]
        old = [float(row['reference_ape_percent']) for row in rows]
        new = [float(row['branch_ape_percent']) for row in rows]
        def metrics(errors):
            return {'cases':len(errors),'median_ape_percent':statistics.median(errors),
                    'p90_ape_percent':float(np.quantile(errors,.9)),
                    'above10_count':sum(value>10 for value in errors),
                    'above20_count':sum(value>20 for value in errors),
                    'above30_count':sum(value>30 for value in errors)}
        before, after = metrics(old), metrics(new)
        fields = ('median_ape_percent','p90_ape_percent','above10_count','above20_count','above30_count')
        improved = all(after[name]<=before[name] for name in fields) and any(after[name]<before[name] for name in fields)
        decisions[impl] = {'approved':bool(improved),'baseline':before,'micro_costs':after,
                           'selection_scope':'development sequence-group validation; promotion used these validation results; prospective new-shape accuracy is not guaranteed',
                           'rule':'median, P90 and >10/>20/>30 counts must all not worsen; at least one improves'}
    model={'version':VERSION,'source_sha256':base.signature(),'hardware':baseline_model['hardware'],
           'baseline':baseline_model,'refined':refined_model,'selection':decisions,
           'inference_inputs':'operator input fields and fixed hardware only; no latency lookup or runtime fitting',
           'validation':'source implementations are calibrated with complete S1/S2 groups held out; the same development validation is used for model selection; prospective validation is still required',
           'stage_costs_independently_measured':False}
    return model
