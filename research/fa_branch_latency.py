from __future__ import annotations
import copy
import csv
import hashlib
import heapq
import json
import inspect
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from prediction.fa_branch_latency import (
    signature,
    tiler,
    geometry,
    implementation,
    cost_class,
    _fake,
    matmul_cache_trace,
    Work,
    batch_work,
    s1_work,
    general_work,
    varlen_work,
    workload,
    adapter_jobs,
    active_features,
    _stages,
    batch_program,
    s1_program,
    general_program,
    varlen_gm_program,
    varlen_l1_program,
    release_graph_pages,
    graph_time,
    parameter_names,
    BranchGraph,
    compact_engine,
    compact_graph,
    branch_graph,
    primitive_class,
    graph_entry,
    phase_features,
    predict_work,
    predict,
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from research import fa_latency_model as source
ROOT = Path(__file__).resolve().parents[1]
VERSION = "fa_branch_execution_v3"
MODEL = ROOT / "prediction" / "models" / "fa_branch_model.json"
PIPES = dict(source.PIPES)
PIPES["vector"] = (PIPES["vector"][0], PIPES["vector"][1] + ("scale_k", "duplicate_k"))
FILES = {"b": ("flash_attention_score_bn2gs1s2_b.h", 544),
         "s1": ("flash_attention_score_s1_bn2gs1.h", 880),
         "general": ("flash_attention_score_s1s2_bn2gs1.h", 770),
         "varlen_gm": ("flash_attention_var_len_score.h", 473),
         "varlen_l1": ("flash_attention_var_len_score.h", 560),
         "varlen_same_ab": ("flash_attention_var_len_score_sab.h", 435),
         "same_ab": ("flash_attention_score_s1s2_bn2gs1_sab.h", 810),
         "drop_adapter": ("flash_attention_score_drop_mask_adapter.h", 83)}


def compact_graph_cache(binary,manifest,saved):
    key=hashlib.sha256((ROOT/'prediction' / 'native' / 'fa_branch_graph.cpp').read_bytes()).hexdigest()
    if saved.get('compactor_sha256')==key:return saved
    unique={}
    for row in saved['samples']:
        for name in ('graph','input_graph'):
            desc=row[name];identity=desc['arrays']['a']['offset']
            unique.setdefault(identity,desc)
    replacements={}
    with binary.open('r+b') as stream:
        for index,(offset,description) in enumerate(sorted(unique.items()),1):
            graph={k:v for k,v in description.items() if k!='arrays'}
            graph.update({n:np.memmap(binary,mode='r',dtype=d['dtype'],offset=d['offset'],shape=tuple(d['shape'])) for n,d in description['arrays'].items()})
            compressed=compact_graph(graph)
            rates=np.arange(1,len(parameter_names())+1,dtype=float)/100
            before=graph_time(graph,rates);after=graph_time(compressed,rates)
            if not math.isclose(before,after,rel_tol=1e-11,abs_tol=1e-8):raise ValueError('compaction changed source graph latency')
            replacement={k:v for k,v in compressed.items() if k not in ('a','b','which','templates')};replacement['arrays']={}
            for name in ('a','b','which','templates'):
                values=np.asarray(compressed[name]);padding=(-stream.tell())%8;stream.write(bytes(padding));position=stream.tell()
                if name=='templates':values=values.copy()
                values.tofile(stream);replacement['arrays'][name]={'offset':position,'shape':values.shape,'dtype':str(values.dtype)}
            replacements[offset]=replacement
            if index%100==0:print(f'BRANCH GRAPH COMPACT {index}/{len(unique)} bytes={stream.tell()}',flush=True)
            del graph,compressed
        stream.truncate(stream.tell())
    for row in saved['samples']:
        for name in ('graph','input_graph'):row[name]=replacements[row[name]['arrays']['a']['offset']]
    saved['compactor_sha256']=key;source.cost.save_json(manifest,saved)
    return saved


def primitive_priors(samples):
    from scipy.optimize import nnls
    pooled=defaultdict(list)
    for row in samples:
        for pipe in PIPES:
            if row['targets'].get(pipe,0)>0:
                pooled[primitive_class(pipe,row['cost_class'])].append(row)
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


def fit_costs(samples, hardware):
    from scipy.optimize import least_squares
    names=parameter_names();positions={key:i for i,key in enumerate(names)}
    priors=primitive_priors(samples)
    grouped=defaultdict(list)
    for row in samples:grouped[row['cost_class']].append(row)
    keys=set(grouped)
    model={'version':VERSION,'source_sha256':signature(),'hardware':hardware,'classes':{},
           'latency_used_for_cost_fit':True,'calibration_target':'training kernel latency and training PMU; Scalar PMU is an upper constraint',
           'coefficient_interpretation':'nonnegative operation costs on branch-specific dependency graphs; no per-shape lookup or total-time correction factor'}
    for key in sorted(keys):
        rows=grouped[key]
        initial=np.zeros(len(names));scales=np.ones(len(names));upper=np.full(len(names),np.inf)
        pipe_info={}
        for pipe,(_,fields) in PIPES.items():
            parent=priors.get(primitive_class(pipe,key))
            if parent is None:continue
            local=np.array([[r['features'][pipe].get(n,0) for n in fields] for r in rows],dtype=float)
            seen=np.any(parent['matrix']!=0,axis=0)
            rank=int(np.linalg.matrix_rank(parent['matrix'][:,seen]/np.maximum(np.max(parent['matrix'][:,seen],axis=0),1e-30))) if any(seen) else 0
            pipe_info[pipe]={'rank':rank,'terms':int(sum(seen)),'samples':len(parent['inputs']),
                             'observed_features':list(parent['observed']),'primitive_calibration_class':list(primitive_class(pipe,key)),
                             'primitive_calibration_input_count':len(parent['inputs'])}
            budget=statistics.median(parent['targets'])/max(1,sum(seen))
            maxima=np.maximum(np.max(parent['matrix'],axis=0),1e-12)
            for j,n in enumerate(fields):
                i=positions[pipe,n];initial[i]=parent['rates'][j];scales[i]=max(initial[i],budget/maxima[j],1e-12)
                if pipe in ('cscalar','vscalar'):upper[i]=parent['upper'][j]
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


def records(paths, audit=None):
    rows, hardware, pending = {}, None, []
    names = ("soc", "aic_cores", "aiv_cores", "l1_bytes", "l2_bytes", "ub_bytes")
    for path in paths:
        if not path.is_file():
            continue
        counts = Counter()
        with path.open(newline="") as stream:
            loaded = json.load(stream).get("measurements", []) if path.suffix == ".json" else csv.DictReader(stream)
            for row in loaded:
                if row.get("status") != "ok" or (_number(row.get("checked_elements")) or 0) <= 0:
                    counts["failed_or_no_numerical_check"] += 1
                    continue
                if (_number(row.get("duration_us")) or 0) <= 0:
                    counts["no_latency"] += 1
                    continue
                if any(_number(row.get(target)) is None for target, _ in PIPES.values()):
                    counts["incomplete_pmu"] += 1
                    continue
                if str(row.get("official_tiling", "true")).lower() == "false":
                    counts["candidate_tiling_not_determined_by_inputs"] += 1
                    continue
                observed = {name: row[name] if name == "soc" else int(row[name]) for name in names if row.get(name) not in (None, "")}
                if len(observed) == len(names):
                    if hardware is not None and observed != hardware:
                        raise ValueError("branch data mix different hardware")
                    hardware = observed
                pending.append((row, observed))
                counts["accepted_measurements"] += 1
        if audit is not None:
            audit.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "counts": dict(counts)})
    if hardware is None:
        raise ValueError("no complete hardware metadata with valid numerical checks and PMU measurements")
    for row, observed in pending:
        if any(hardware[name] != value for name, value in observed.items()):
            raise ValueError("branch data mix different hardware")
        p = source.cost.normalize({name: row[name] for name in source.cost.INPUT_FIELDS})
        identity = source.cost.sha256(p)
        rows[identity] = {**row, "case": row.get("case") or row.get("name") or "input_" + identity[:16], "inputs": p, "input_sha256": identity,
                          "sequence_group": source.cost.sha256({name: p[name] for name in ("S1", "S2")})}
    return list(rows.values()), hardware


def _summary(rows):
    good = [r for r in rows if r.get("branch_ape_percent") is not None]
    errors = [r["branch_ape_percent"] for r in good]
    return {"cases": len(rows), "predicted": len(good), "unavailable": len(rows) - len(good),
            "median_ape_percent": statistics.median(errors) if errors else None,
            "max_ape_percent": max(errors, default=None),
            "above": {str(limit): {"count": sum(value > limit for value in errors),
                                   "percent": 100 * sum(value > limit for value in errors) / len(errors) if errors else None}
                      for limit in (10, 20, 30)}}


def prepare_graph_samples(rows, hardware):
    folder=ROOT/'out/fa_branch_host';folder.mkdir(parents=True,exist_ok=True)
    binary,manifest=folder/'graph_cache.bin',folder/'graph_cache.json'
    identity=source.cost.sha256({'rows':rows,'hardware':hardware,'pipes':PIPES,'api':tiler()[1],
        'primitives':source.source_hash(),'functions':[inspect.getsource(f) for f in
            (Work,matmul_cache_trace,workload,batch_work,s1_work,general_work,varlen_work,adapter_jobs,active_features,BranchGraph)]})
    saved=json.loads(manifest.read_text()) if manifest.is_file() else {}
    archive=folder/'graph_cache.bin.gz'
    if saved.get('identity')==identity and not binary.is_file() and archive.is_file():
        import gzip
        import shutil
        temporary=binary.with_suffix('.restoring')
        try:
            with gzip.open(archive,'rb') as compressed,temporary.open('wb') as restored:
                shutil.copyfileobj(compressed,restored,1024*1024)
            temporary.replace(binary)
        finally:
            temporary.unlink(missing_ok=True)
    if saved.get('identity')!=identity or not binary.is_file():
        samples,rejected=[],[]
        with binary.open('wb') as stream:
            def save_graph(graph):
                result={k:v for k,v in graph.items() if k not in ('a','b','which','templates')}
                result['arrays']={}
                for key in ('a','b','which','templates'):
                    values=graph[key];padding=(-stream.tell())%8;stream.write(bytes(padding));offset=stream.tell();values.tofile(stream)
                    result['arrays'][key]={'offset':offset,'shape':values.shape,'dtype':str(values.dtype)}
                return result
            for index,row in enumerate(rows,1):
                try:
                    work=workload(row['inputs'],hardware)
                    if int(row['priority'])!=work.s['priority']:raise ValueError('input-derived route differs from official Host trace')
                    actual=[];differences=[]
                    for stage,mm in enumerate(work.mm,1):
                        fields={name:int(row[f'bmm{stage}_{name}']) for name in mm if _number(row.get(f'bmm{stage}_{name}')) is not None}
                        for name in ('base_m','base_n','base_k'):
                            if name in fields and fields[name]!=mm[name]:raise ValueError('input-derived Matmul base differs from official Host')
                        differences.extend(f'bmm{stage}_{n}:{mm[n]}!={v}' for n,v in fields.items() if mm[n]!=v)
                        actual.append(fields)
                    input_graph=save_graph(branch_graph(work));input_features=active_features(work)
                    if differences:
                        work=workload(row['inputs'],hardware,actual)
                        graph=save_graph(branch_graph(work));features=active_features(work)
                    else:graph=input_graph;features=input_features
                    samples.append({'input_sha256':row['input_sha256'],'sequence_group':row['sequence_group'],
                        'inputs':row['inputs'],'case':row['case'],'collection_group':row.get('collection_group') or row['family'],
                        'cost_class':cost_class(work.p,work.s),'features':features,'input_features':input_features,
                        'duration_us':float(row['duration_us']),'targets':{pipe:float(row[target]) for pipe,(target,_) in PIPES.items()},
                        'graph':graph,'input_graph':input_graph,'geometry_differences':differences})
                except (ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as error:
                    rejected.append({'input_sha256':row['input_sha256'],'case':row['case'],'error':str(error)})
                if index%25==0:print(f'BRANCH SOURCE GRAPHS {index}/{len(rows)} accepted={len(samples)} bytes={stream.tell()}',flush=True)
        saved={'identity':identity,'samples':samples,'rejected':rejected}
        source.cost.save_json(manifest,saved)
    else:print(f'BRANCH GRAPH CACHE HIT samples={len(saved["samples"])}',flush=True)
    saved=compact_graph_cache(binary,manifest,saved)
    result=[]
    measurements={r['input_sha256']:r for r in rows}
    for row in saved['samples']:
        item={**row}
        measured=measurements[row['input_sha256']]
        clocks=[float(measured[n]) for n in ('aic_time_max_us','aiv_time_max_us') if _number(measured.get(n)) is not None]
        interval=[float(measured[n]) for n in ('duration_us','round1_us','round2_us','round3_us','median_ci95_high_us') if (_number(measured.get(n)) or 0)>0]
        reference=max(interval)
        item['pmu_time_ratio']=max(clocks,default=reference)/reference
        item['pmu_weight']=min(1,1/item['pmu_time_ratio'])**2
        for key in ('graph','input_graph'):
            graph={k:v for k,v in row[key].items() if k!='arrays'}
            graph.update({name:np.memmap(binary,mode='r',dtype=data['dtype'],offset=data['offset'],shape=tuple(data['shape']))
                          for name,data in row[key]['arrays'].items()})
            item[key]=graph
        result.append(item)
    return result,saved['rejected']


def build_and_evaluate(paths, evaluation_path, output_dir, model_path=MODEL):
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
        model=fit_costs(training,hardware)
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
                sample=indexed[row['input_sha256']];entry=graph_entry(model,sample['cost_class'])
                result.update(branch_pmu_time_ratio=sample['pmu_time_ratio'],branch_pmu_calibration_weight=sample['pmu_weight'])
                for pipe,values in sample['input_features'].items():
                    missing=[n for n,v in values.items() if v>0 and n not in entry['pipes'][pipe]['observed_features']]
                    if missing:raise ValueError('unmeasured source primitive '+pipe+'/'+','.join(missing))
                val=graph_time(sample['input_graph'],entry['rates']);replay=graph_time(sample['graph'],entry['rates'])
                if not math.isfinite(val) or val<=0:raise ValueError('invalid source graph prediction')
                critical=source.critical_phases(sample['input_graph'],entry['rates']);release_graph_pages(sample['input_graph'])
                result.update(branch_status='predicted',branch_predicted_us=val,branch_ape_percent=100*abs(val/float(row['duration_us'])-1),
                              branch_declared_tiling_predicted_us=replay,branch_declared_tiling_ape_percent=100*abs(replay/float(row['duration_us'])-1),
                              branch_implementation=sample['cost_class'].split('|')[0],branch_cost_class=sample['cost_class'],
                              branch_unidentified_pipe_terms='|'.join(p for p,x in entry['pipes'].items() if x['rank']<x['terms']),
                              branch_direct_class_samples=len(entry['calibration_inputs']),
                              branch_adapter_us=critical.get('drop_adapter',0),branch_startup_us=critical.get('startup',0),
                              branch_critical_path_phase_us=json.dumps(critical,sort_keys=True),
                              branch_matmul_geometry_differences='|'.join(sample['geometry_differences']))
                for pipe,values in sample['input_features'].items():
                    estimated=sum(v*entry['pipes'][pipe]['coefficients'].get(n,0) for n,v in values.items());actual=sample['targets'][pipe]
                    result[f'branch_{pipe}_active_sum_us']=estimated
                    result[f'branch_{pipe}_ape_percent']=100*abs(estimated/actual-1) if actual>0 else None
                    if pipe in ('cscalar','vscalar'):result[f'branch_{pipe}_polling_and_unmodeled_busy_us']=actual-estimated
            except (ValueError,RuntimeError,KeyError) as error:result['branch_error']=str(error)
            result.pop('inputs',None);results.append(result)
        print(f'BRANCH EVALUATED {len(results)}/{len(evaluation)}',flush=True)
    final_model=fit_costs(prepared,hardware)
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
        report['groups'][group]['by_layout']={layout:_summary([r for r in selected if r['layout']==layout]) for layout in sorted({r['layout'] for r in selected})}
        summary=report['groups'][group]
        print(f"BRANCH RESULT {group}: predicted={summary['predicted']}/{summary['cases']} MdAPE={summary['median_ape_percent']}% "+' '.join(f'>{limit}%={v["count"]}/{summary["predicted"]}' for limit,v in summary['above'].items()),flush=True)
    final_model['cross_validation']=report['groups'];source.cost.save_json(model_path,final_model);source.cost.save_json(output_dir/'branch_evaluation.json',report)
    print(f'BRANCH_MODEL={model_path}\nBRANCH_PREDICTIONS={output_dir/"branch_predictions.csv"}',flush=True)
    return report


@lru_cache(maxsize=1)
def _forecast_state(directory):
    path = Path(directory) / "branch_predictions_before_measurement.json"
    return json.loads(path.read_text()) if path.is_file() else {"models": {}, "predictions": {}}


def _load_forecast_model():
    model = json.loads(MODEL.read_text())
    if model.get("source_sha256") != signature():
        raise ValueError("branch model and execution code differ")
    return model


@lru_cache(maxsize=1)
def _forecast_model():
    try:
        return _load_forecast_model()
    except (ValueError, RuntimeError, KeyError, OSError, subprocess.SubprocessError) as error:
        return {"unavailable_reason": str(error)}


def freeze_case(case, hardware, directory):
    state = _forecast_state(str(directory))
    identity = source.cost.sha256(case["inputs"])
    saved = state["predictions"].get(case["name"])
    if saved is not None:
        if saved["input_sha256"] != identity:
            raise ValueError("saved branch forecast belongs to another input")
        saved_model = state["models"].get(saved.get("model_sha256"))
        if saved.get("status") == "predicted" and (saved_model is None or any(
                str(hardware.get(name)) != str(value) for name, value in saved_model["hardware"].items())):
            raise ValueError("saved branch forecast belongs to another hardware configuration")
        return saved, saved_model
    item = {"input_sha256": identity, "status": "unavailable"}
    model = None
    try:
        model = _forecast_model()
        if model.get("unavailable_reason"):
            raise ValueError(model["unavailable_reason"])
        for name, value in model["hardware"].items():
            if str(hardware.get(name)) != str(value):
                raise ValueError(f"branch model hardware differs: {name}")
        model_sha = source.cost.sha256(model)
        item["model_sha256"] = model_sha
        result = predict(case["inputs"], model)
        result["prediction_quality"] = "source_operation_graph_with_grouped_validation"
        item.update(status="predicted", result=result)
        state["models"][model_sha] = model
    except (ValueError, RuntimeError, KeyError, OSError, subprocess.SubprocessError) as error:
        item["error"] = str(error)
    state["predictions"][case["name"]] = item
    source.cost.save_json(Path(directory) / "branch_predictions_before_measurement.json", state)
    return item, model
