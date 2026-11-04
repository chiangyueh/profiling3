from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv,json,math,re,statistics,collections,gc,time,hashlib,warnings
from pathlib import Path
import numpy as np
from research import fa_latency_transfer as t
from scipy.optimize import linprog,OptimizeWarning
ROOT=Path(__file__).resolve().parents[1]
EXTRA=('unique_kv_share_kb','evicted_repeat_kv_kb')
NAMES=t.PARAMETERS+tuple(('mte2',x) for x in EXTRA)
DMA=t.FEATURES+EXTRA
MODELS={
 'current_form_refit':t.FEATURES,
 'copy_only_refit':t.BASE_FEATURES,
 'reuse_traffic_refit':t.BASE_FEATURES+('nd2nz_split_sector_kb',)+EXTRA,
}


def fraction(w):
    return 1/((w.p['N1']//w.p['N2'])*w.s['s1_outer'])

def extra(item,w):
    kv=item.get('source') in ('K','V')
    share=fraction(w)
    footprint=2*w.p['B']*w.p['N2']*w.p['S2']*(w.p['D']+w.p['DV'])
    pressure=max(0,1-w.hardware['l2_bytes']/footprint)
    return dict(unique_kv_share_kb=item['payload_kb']*share if kv else 0,
        evicted_repeat_kv_kb=item['payload_kb']*(1-share)*pressure if kv else 0)

class Graph(t.Graph):
    def __init__(self,w):
        super().__init__(w)
        self.names=NAMES;self.columns={k:i for i,k in enumerate(NAMES)}
        self.templates=[tuple(x)+(0.,)*len(EXTRA) for x in self.templates]
        self.template_ids={(p,x):i for i,(p,x) in enumerate(zip(self.phases,self.templates))}
    def template(self,pipe,values,phase,divisor=1):
        if pipe!='mte2':return super().template(pipe,values,phase,divisor)
        value={**values,**extra(values,self.work)}
        row=[0.]*len(self.names)
        for name in DMA:row[self.columns[pipe,name]]=value.get(name,0)/divisor
        key=(phase,tuple(row))
        if key not in self.template_ids:
            self.template_ids[key]=len(self.templates);self.templates.append(key[1]);self.phases.append(phase)
        return self.template_ids[key]

def features(w):
    result=collections.Counter(t.features(w))
    for pair in w.pairs:
        for tasks in pair:
            for unit in tasks:
                for job in unit['cube']:
                    for item in job['dma_query']+[c for group in job['dma_groups'] for c in group]:result.update(extra(item,w))
    return [float(result[x]) for x in DMA]

def host_geometry(r,w):
    result=[]
    for stage,mm in enumerate(w.mm,1):
        fields={}
        for name in mm:
            key='FA_BMM'+str(stage)+'_'+name.upper()
            if r.get(key) not in ('',None):fields[name]=int(r[key])
        result.append(fields)
    return result

def quality(r,p,w):
    errors=[];duration=float(r['duration_us'])
    if r.get('pmu_complete')!='True':errors.append('incomplete_pmu')
    if float(r.get('checked_elements') or 0)<=0 or float(r.get('failed_elements') or 0)>0:errors.append('numerical_check')
    if r.get('actual_implementation')!='varlen_l1' or r.get('schedule_disagreements')!='{}':errors.append('route_or_schedule')
    if any(float(r.get(k) or 0)>duration*1.02 for k in ('aic_time_max_us','aiv_time_max_us')):errors.append('per_core_time_exceeds_task')
    if any(float(r.get(k) or 0)>duration*1.02 for k in r if k.endswith('_active_max_us')):errors.append('pipe_time_exceeds_task')
    if not 0<=float(r.get('aic_mte2_ratio') or 0)<=1.02:errors.append('invalid_mte2_ratio')
    actual=t.workload(p,w.hardware,host_geometry(r,w))
    count=t.micro.active_features(actual)['cube']['tiles']
    if abs(float(r['aic_cube_instructions'])-count)>.5:errors.append('mmad_count_mismatch')
    payload=t.features(w)['payload_kb']
    if abs(float(r['gm_to_l1_kb'])/payload-1)>.1:errors.append('source_payload_difference')
    return errors

def fit(x,y,cols):
    xx=x[:,cols];active=np.max(xx,axis=0)>0;xx=xx[:,active]
    ranks=int(np.linalg.matrix_rank(xx/np.maximum(np.max(xx,axis=0),1e-30)))
    scale=np.maximum(np.max(xx/y[:,None],axis=0),1e-30);a=xx/y[:,None]/scale;n=len(y)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning)
        result=linprog(np.r_[np.zeros(xx.shape[1]),np.ones(n)/n],A_ub=np.vstack((np.c_[a,-np.eye(n)],np.c_[-a,-np.eye(n)])),b_ub=np.r_[np.ones(n),-np.ones(n)],bounds=(0,None),method='highs',options={'threads':1})
    if not result.success:raise RuntimeError(result.message)
    rates=np.zeros(len(DMA));rates[np.array(cols)[active]]=result.x[:xx.shape[1]]/scale
    return rates,ranks,set(np.array(cols)[~active])

def stats(values):
    a=np.array(values);return {'n':len(a),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(max(a)),'above10':int(sum(a>10)),'above20':int(sum(a>20)),'above30':int(sum(a>30))}

def run(paths, destination):
    model=json.load(open(ROOT / 'prediction/models/fa_branch_model.json'));hw=model['hardware']
    samples=[];raw=[]
    reference_contract={k:v for k,v in model['varlen_l1_transfer'].items() if k not in ('api_input_normalization','vector_accumulator_read_policy')}
    for path in paths:
        grouped=collections.defaultdict(list)
        for r in csv.DictReader(path.open()):
            if r['status']!='ok':continue
            p=t.micro.source.cost.normalize({k:r[k] for k in t.micro.source.cost.INPUT_FIELDS})
            grouped[t.micro.source.cost.sha256(p)].append((r,p))
        for identity,rs in grouped.items():
            p=rs[0][1];eff=t.api_kernel_inputs(p,model['varlen_l1_transfer']['api_input_normalization']);w=t.workload(eff,hw)
            flags=sorted({e for r,_ in rs for e in quality(r,eff,w)})
            item={'id':identity,'p':p,'eff':eff,'dataset':path.name,'actual':statistics.median(float(r['duration_us']) for r,_ in rs),'recorded_prediction':float(rs[0][0]['predicted_us']),'published':t.predict(p,model['refined'],reference_contract)['predicted_us'],
                'flags':flags,'target':statistics.median(float(r['aic_mte2_active_sum_us']) for r,_ in rs),'x':features(w)}
            raw.append(item)
            if not flags:samples.append(item)
    print('DATA',len(raw),'ELIGIBLE',len(samples),'FLAGS',dict(collections.Counter(flag for s in raw for flag in s['flags'])),flush=True)
    x=np.array([r['x'] for r in samples]);y=np.array([r['target'] for r in samples])
    axes={
      'sequence_pair':lambda p:(p['S1'],p['S2']),
      'head_combination':lambda p:(p['N1'],p['N2']),
      'dimension_pair':lambda p:(p['D'],p['DV']),
    }
    allrates={};support={}
    for axis,group in axes.items():
        labels=[group(r['eff']) for r in samples]
        for key in sorted(set(labels)):
            train=np.array([label!=key for label in labels]);test=~train
            for name,cols in MODELS.items():
                indices=[DMA.index(c) for c in cols]
                rates,rank,missing=fit(x[train],y[train],indices)
                allrates[axis,key,name]=rates
                support[axis,key,name]={'rank':rank,'missing':list(missing),'test_has_missing':bool(missing and np.any(x[test][:,list(missing)]>0))}
        print('FIT',axis,'groups',len(set(labels)),flush=True)
    results=[];start=time.monotonic()
    for i,r in enumerate(samples,1):
        w=t.workload(r['eff'],hw)
        entry=t.micro.graph_entry(model['refined'],t.micro.cost_class(w.p,w.s),t.micro.execution_conditions(w))
        pair_seen=set();graphs=[]
        for pair in w.pairs:
            identity=tuple(tuple(id(u) for u in tasks) for tasks in pair)
            if identity in pair_seen:continue
            pair_seen.add(identity)
            import copy
            component=copy.copy(w);component.pairs=[pair]
            graphs.append(t.micro.compact_graph(Graph(component).build()))
        olddma=np.array([model['varlen_l1_transfer']['coefficients'].get(k,0) for k in DMA])
        def predict(dma):
            rates=list(entry['rates'])+list(dma)
            return max(t.micro.graph_time(g,rates) for g in graphs)
        api=predict(olddma)
        output={k:r[k] for k in ('id','p','dataset','actual','published')};output['api_source']=api;output['predictions']={}
        for axis,group in axes.items():
            for name in MODELS:
                key=axis,group(r['eff']),name
                if support[key]['test_has_missing']:value=None
                else:value=predict(allrates[key])
                output['predictions'][axis+'/'+name]=value
        results.append(output)
        del graphs,w;gc.collect()
        if i%10==0:print('GRAPH',i,'/',len(samples),'seconds',round(time.monotonic()-start,1),flush=True)
    report={'scope':'retrospective grouped cross validation; MTE2 activity targets only; no kernel latency target; no NPU measurement; baseline preserves the same coefficients without API normalization','quality_eligible':len(samples),'total_measured_inputs':len(raw),'published':stats([100*abs(r['published']/r['actual']-1) for r in results]),'api_source':stats([100*abs(r['api_source']/r['actual']-1) for r in results]),'cross_validation':{}}
    for axis in axes:
        for name in MODELS:
            key=axis+'/'+name;errors=[100*abs(r['predictions'][key]/r['actual']-1) for r in results if r['predictions'][key] is not None]
            record=stats(errors) if errors else {'n':0};record['unavailable']=len(results)-len(errors);report['cross_validation'][key]=record
    report['latest_quality_passed']={}
    latest=[r for r in results if r['dataset']==paths[-1].name]
    for name in ('published','api_source'):
        report['latest_quality_passed'][name]=stats([100*abs(r[name]/r['actual']-1) for r in latest])
    for axis in axes:
        for name in MODELS:
            key=axis+'/'+name;vals=[100*abs(r['predictions'][key]/r['actual']-1) for r in latest if r['predictions'][key] is not None]
            report['latest_quality_passed'][key]=stats(vals)
    print('REPORT',json.dumps(report),flush=True)
    write_audit(destination, paths, model, report, results, raw)

def promotion_decisions(report):
    metrics=report['latest_quality_passed'];reference=metrics['api_source'];decisions={}
    for name in MODELS:
        reasons=[]
        for axis in ('sequence_pair','head_combination','dimension_pair'):
            result=metrics[axis+'/'+name];control=metrics[axis+'/current_form_refit']
            if result['n']!=reference['n']:reasons.append(axis+': incomplete coverage')
            for field in ('median','p90','above20','above30'):
                if result[field]>reference[field]+1e-9:reasons.append(axis+': worse '+field+' than frozen source correction')
                if result[field]>control[field]+1e-9:reasons.append(axis+': worse '+field+' than same-form refit control')
        decisions[name]={'grouped_checks_passed':not reasons,'activated':False,'reasons':reasons,'prospective_validation_required':True}
    return decisions


def write_audit(destination, paths, model, report, results, raw):
    indexes={r['id']:r for r in results}
    api_predictions={}
    api_contract={k:v for k,v in model['varlen_l1_transfer'].items() if k!='vector_accumulator_read_policy'}
    methods=list(next(iter(results))['predictions']) if results else []
    fields=['record_kind','dataset','case','inputs','dtype','quality_passed','quality_flags','reference_us','api_corrected_us','measured_us','reference_ape_percent','api_ape_percent','effective_Dqk','effective_Dv',*methods,'method','cases','median_ape_percent','p90_ape_percent','max_ape_percent','above10','above20','above30','notes']
    with destination.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerow(dict(record_kind='methodology',method='source_API_normalization_and_grouped_CV',notes=json.dumps({'model_sha256':t.micro.source.cost.sha256(model),'study_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policy':model['varlen_l1_transfer']['api_input_normalization'],'datasets':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'quality_policy':'PMU fields present; numerical check; correct route; each per-core and pipe maximum within 2% of task duration; MTE2 ratio in [0,1.02]; exact source/Host MMAD count; payload within 10%; no APE-based filtering','fit_target':'whole-device MTE2 active-sum only; pipeline dependencies and other costs unchanged','held_out_units':'entire S1/S2 pair, head combination, or effective Dqk/Dv pair; all repeats and dtypes of the same group excluded from training','candidate_cost_refinements_activated':False},sort_keys=True)))
        for item in raw:
            r=indexes.get(item['id']);api=r['api_source'] if r else t.predict(item['p'],model['refined'],api_contract)['predicted_us']
            api_predictions[item['id']]=api
            result=dict(record_kind='sample',dataset=item['dataset'],case='varlen_'+item['id'][:16],inputs=json.dumps({k:item['p'][k] for k in t.micro.source.cost.INPUT_FIELDS[:7]},sort_keys=True),dtype=item['p']['dtype'],quality_passed=not item['flags'],quality_flags=';'.join(item['flags']),reference_us=item['published'],api_corrected_us=api,measured_us=item['actual'],reference_ape_percent=100*abs(item['published']/item['actual']-1),api_ape_percent=100*abs(api/item['actual']-1),effective_Dqk=item['eff']['D'],effective_Dv=item['eff']['DV'])
            if r:result.update(r['predictions'])
            writer.writerow(result)
        for path in paths:
            for row in csv.DictReader(path.open()):
                if row['status']=='ok':continue
                writer.writerow(dict(record_kind='failed_collection',dataset=path.name,case=row['case'],dtype=row['dtype'],quality_passed=False,quality_flags=row.get('error','')))
        for cohort,metrics in [('quality_all',{'reference':report['published'],'api_source':report['api_source'],**report['cross_validation']}),('quality_latest',report['latest_quality_passed'])]:
            for name,values in metrics.items():
                writer.writerow(dict(record_kind='summary',dataset=cohort,method=name,cases=values['n'],median_ape_percent=values.get('median'),p90_ape_percent=values.get('p90'),max_ape_percent=values.get('max'),above10=values.get('above10'),above20=values.get('above20'),above30=values.get('above30'),notes='retrospective grouped evaluation; not prospective accuracy'))
        for name,decision in promotion_decisions(report).items():
            writer.writerow(dict(record_kind='decision',method=name,notes=json.dumps(decision,sort_keys=True)))
        for dataset in dict.fromkeys(r['dataset'] for r in raw):
            part=[r for r in raw if r['dataset']==dataset]
            for name in ('reference','api_source'):
                values=[100*abs((r['published'] if name=='reference' else api_predictions[r['id']])/r['actual']-1) for r in part]
                v=stats(values);writer.writerow(dict(record_kind='summary',dataset=dataset+'/all_recorded_times',method=name,cases=v['n'],median_ape_percent=v['median'],p90_ape_percent=v['p90'],max_ape_percent=v['max'],above10=v['above10'],above20=v['above20'],above30=v['above30'],notes='all successful timing records retained, including PMU diagnostic flags'))
    print('AUDIT_CSV='+str(destination),flush=True)


if __name__=='__main__':
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(1100*1024**2,1100*1024**2))
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('paths',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'research' / 'reports' / 'fa_condition_audit.csv')
    args=parser.parse_args()
    run(args.paths,args.output)
