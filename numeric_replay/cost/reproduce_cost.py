"""Portable, standard-library-only replay of archived PFE-UNet cost records.
No model execution, imports of archived scripts, device query or original path access.
"""
from pathlib import Path, PurePosixPath
import argparse,csv,hashlib,json,math,statistics,sys,zipfile
import xml.etree.ElementTree as ET
METHODS=['direct_reference','pfe_reference','learned_fusion_reference','global_film_style','wide_direct']
SEEDS=[42,2026,3407,7819,12011]
LABELS=['Direct','PFE','Learned fusion','Global FiLM style','Widened Direct']

def need(v,message):
    if not v:raise ValueError(message)
def unique_object(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,'duplicate JSON key: '+k);d[k]=v
    return d
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'),object_pairs_hook=unique_object,parse_constant=lambda x:(_ for _ in []).throw(ValueError('nonfinite JSON: '+x)))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def located(root,rel):
    parts=PurePosixPath(rel)
    need(not parts.is_absolute() and bool(parts.parts) and '\\' not in rel and ':' not in rel and '..' not in parts.parts,'unsafe relative path')
    p=root.joinpath(*parts.parts)
    need(p.resolve().is_relative_to(root.resolve()),'path escaped package')
    return p

def verify_package(root):
    manifest=read(root/'MANIFEST.json');seen=set()
    for item in manifest['files']:
        rel=item['path'];need(rel not in seen and rel!='MANIFEST.json','duplicate/reserved manifest entry');seen.add(rel)
        p=located(root,rel);need(p.is_file() and not p.is_symlink(),'missing/symlink package member: '+rel)
        need(p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'checksum mismatch: '+rel)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    need(actual==seen,'unlisted or missing package members')
    return len(seen)

def finite(v):
    x=float(v);need(math.isfinite(x),'nonfinite value');return x
def close(a,b,label):
    need(math.isclose(finite(a),finite(b),rel_tol=1e-12,abs_tol=1e-12),'numeric mismatch: '+label)
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def stat(xs):
    need(len(xs)>=2,'need at least two values')
    return {'mean':statistics.mean(xs),'sample_sd':statistics.stdev(xs),'minimum':min(xs),'maximum':max(xs)}
def quantile(xs,p):
    values=sorted(xs);need(bool(values) and 0<=p<=1,'invalid quantile')
    pos=(len(values)-1)*p;lo=math.floor(pos);hi=math.ceil(pos)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)

def history_totals(history):
    need(len(history)==50 and [int(x['epoch']) for x in history]==list(range(1,51)),'history must contain ordered epochs 1..50 exactly once')
    need(all(int(x['train_images'])==3104 and int(x['val_images'])==776 for x in history),'training/validation population changed')
    training=[finite(x['train_seconds']) for x in history];validation=[finite(x['validation_seconds']) for x in history]
    need(all(x>0 for x in training+validation),'nonpositive loop interval')
    return {'train_seconds_sum':math.fsum(training),'validation_seconds_sum':math.fsum(validation),'loop_seconds_sum':math.fsum(training)+math.fsum(validation)}

def architectural_counts(method):
    need(method in METHODS,'unknown method')
    width=41 if method=='wide_direct' else 32
    ch=[width*2**i for i in range(5)];params=0;macs=0
    def block(ci,co,s):return 9*ci*co+9*co*co+4*co,s*s*(9*ci*co+9*co*co)
    for inp in ([4] if method in ('direct_reference','wide_direct') else [3,1]):
        ci=inp
        for level,co in enumerate(ch):
            p,m=block(ci,co,352//2**level);params+=p;macs+=m;ci=co
    for level in range(4):
        co=ch[level];p,m=block(co+ch[level+1],co,352//2**level);params+=p;macs+=m
    params+=width+1;macs+=352*352*width
    for level,co in enumerate(ch):
        if method=='pfe_reference':
            mid=max(co//4,8);params+=2*co*mid+2*mid+co;macs+=2*co*mid*(352//2**level)**2
        elif method in ('learned_fusion_reference','global_film_style'):
            params+=2*co*co+2*co;macs+=2*co*co*(1 if method=='global_film_style' else (352//2**level)**2)
    return params,macs

def capacity(root,registry):
    expected={x['method']:x for x in read(root/'records/stage83/analysis_v1/capacity_compute_timing.json')}
    oldtimes={x['run_id']:x for x in rows(root/'records/stage83/analysis_v1/timing_per_run.csv')}
    per_run=[]
    for entry in registry:
        rid=entry['run_id'];folder=root/'records/training'/rid
        cfg=read(folder/'config.json');receipt=read(folder/'portable_complete.json')
        need(cfg['run_id']==rid and cfg['method']==entry['method'] and cfg['seed']==entry['seed'],'training identity mismatch')
        need(receipt['run_id']==rid and receipt['protocol_sha256']==read(root/'records/stage83/contract.json')['formal_protocol_sha256'],'training protocol mismatch')
        need(digest(folder/'history.csv')==receipt['artifact_sha256']['history.csv'],'history receipt mismatch')
        v=history_totals(rows(folder/'history.csv'));v.update(run_id=rid,method=entry['method'],seed=entry['seed'])
        for k in ['train_seconds_sum','validation_seconds_sum','loop_seconds_sum']:close(v[k],oldtimes[rid][k],rid+'/'+k)
        per_run.append(v)
    summary=[]
    for method in METHODS:
        detail=read(root/('records/stage83/analysis_v1/'+method+'_detail.json'))
        observed=0
        for op in detail['convolutions']:
            n,co,ho,wo=op['output_shape'];ci=op['in_channels'];g=op['groups'];kh,kw=op['kernel']
            need(all(isinstance(v,int) and not isinstance(v,bool) and v>0 for v in [n,co,ho,wo,ci,g,kh,kw]) and ci%g==0,'invalid convolution dimensions')
            mac=n*co*ho*wo*(ci//g)*kh*kw;need(mac==op['MACs'],'per-call MAC mismatch');observed+=mac
        params,formula_macs=architectural_counts(method)
        need(observed==formula_macs==detail['summary']['conv_MACs_per_image'],'architecture/hook arithmetic mismatch')
        need(params==detail['summary']['trainable_parameters']==detail['summary']['graph_connected_parameters'],'parameter formula mismatch')
        result={'method':method,'n':5,'trainable_parameters':params,'conv_MACs_per_image':observed,'conv_GMACs':observed/1e9}
        values=[x for x in per_run if x['method']==method];need([x['seed'] for x in values]==SEEDS,'incomplete capacity seeds')
        for old,new in [('train_seconds_sum','train_hours'),('validation_seconds_sum','validation_hours'),('loop_seconds_sum','loop_hours')]:
            for key,val in stat([x[old]/3600 for x in values]).items():result[new+'_'+key]=val
        for key,val in result.items():
            if key!='method':close(val,expected[method][key],method+'/'+key)
        summary.append(result)
    return summary,per_run

def validate_iterations(records,phase,identity):
    need(len(records)==35,'forward count must be 35')
    last=0;warm=[];timed=[]
    for i,v in enumerate(records):
        need(v['iteration']==i and v['kind']==('warmup' if i<5 else 'measured') and v['phase']==phase,'iteration order/phase changed')
        need(v['identity']==identity,'iteration worker identity changed')
        interval=v['interval'];start,end,elapsed=[interval[k] for k in ['start_ns','end_ns','elapsed_ns']]
        need(all(type(x) is int for x in [start,end,elapsed]) and end>start>=last and elapsed==end-start,'invalid ordered time interval')
        need(v['finite'] is True and v['readback_completed'] is True and v['requires_grad'] is False and v['gradient_enabled'] is False,'completed readback receipt invalid')
        need(v['output_shape']==[1,1,352,352] and v['output_bytes']==495616,'invalid output shape/bytes')
        (warm if i<5 else timed).append(elapsed/1e6);last=end
    return warm,timed

def timing(root,registry,phase):
    contract=read(root/'records/stage87/contract.json');order=contract['qualification_order'] if phase=='qualification' else contract['protocol']['run_order']
    expected=read(root/('records/stage87/'+phase+'_assessment.json'))
    reference={x['run_id']:x for x in expected['summaries']};mapping={x['run_id']:x for x in registry}
    complete=read(root/('records/stage87/'+phase+'_v1/complete.json'))
    need(complete['passed'] is True and complete['completed_run_ids']==order,'completion population/order mismatch')
    summaries=[];raw=[]
    for rid in order:
        p=root/('records/stage87/'+phase+'_v1')/rid
        done=read(p/'done.json');result=read(p/'result.json');identity=read(p/'binding.json')['worker']
        entry=mapping[rid];need(done['run_id']==rid and done['method']==entry['method'] and done['seed']==entry['seed'] and done['phase']==phase,'timing identity mismatch')
        need(done['forwards']==35 and done['warmups']==5 and done['measurements']==30 and done['training_steps']==0 and not done['test_images_read'] and not done['numerical_comparison'],'forward scope changed')
        need(done['input_sha256']==contract['input_sha256'],'synthetic input receipt mismatch')
        need(result['passed'] and result['done_sha256']==digest(p/'done.json') and result['samples_sha256']==digest(p/'samples.json'),'run result binding mismatch')
        need({x.name for x in p.glob('iteration_*.json')}=={'iteration_%02d.json'%i for i in range(35)},'missing or extra iteration file')
        records=[read(p/('iteration_%02d.json'%i)) for i in range(35)]
        warm,times=validate_iterations(records,phase,identity)
        need(done['monotonic_ns']>=records[-1]['interval']['end_ns'],'premature completion')
        samples=read(p/'samples.json');need(bool(samples) and samples[0]['phase']=='before_payload','missing lifecycle baseline')
        need(all(x['worker']==identity for x in samples),'resource worker changed')
        s={'run_id':rid,'method':entry['method'],'seed':entry['seed'],'median_ms':statistics.median(times),'p25_ms':quantile(times,.25),'p75_ms':quantile(times,.75),'minimum_ms':min(times),'maximum_ms':max(times),'min_available_ram_gib':min(finite(x['host']['available_ram_gib']) for x in samples),'max_private_gib':max(finite(x['worker_memory']['private_bytes']) for x in samples)/2**30,'max_working_set_gib':max(finite(x['worker_memory']['working_set_bytes']) for x in samples)/2**30}
        for k,v in s.items():
            if k not in ('run_id','method'):close(v,reference[rid][k],phase+'/'+rid+'/'+k)
        need(warm==reference[rid]['warmup_ms'] and times==reference[rid]['measured_ms'],'raw timing population mismatch')
        for i,v in enumerate(records):raw.append(dict(phase=phase,run_id=rid,method=entry['method'],seed=entry['seed'],iteration=i,kind=v['kind'],**v['interval']))
        summaries.append(s)
    return summaries,raw

def aggregate_timing(per_run):
    result=[]
    for method in METHODS:
        values=[x for x in per_run if x['method']==method]
        need(len(values)==5 and {x['seed'] for x in values}==set(SEEDS),'incomplete timing seeds')
        xs=[x['median_ms'] for x in values]
        result.append({'method':method,'n_checkpoints':5,'mean_run_median_ms':statistics.mean(xs),'sample_sd_of_run_medians_ms':statistics.stdev(xs),'min_run_median_ms':min(xs),'max_run_median_ms':max(xs),'min_available_ram_gib':min(x['min_available_ram_gib'] for x in values),'max_private_gib':max(x['max_private_gib'] for x in values),'max_working_set_gib':max(x['max_working_set_gib'] for x in values)})
    return result

def paper_cells(cap,latency):
    s25=[['Method','Params(M)','Conv.GMAC','Training(h)','Validation(h)','Sum of loops(h)']]
    s26=[['Method','Latency(ms)','Median range(ms)','Max private(GiB)','Max WS(GiB)']]
    for label,c,t in zip(LABELS,cap,latency):
        s25.append([label,f"{c['trainable_parameters']/1e6:.3f}",f"{c['conv_GMACs']:.3f}"]+[f"{c[x+'_mean']:.3f} ± {c[x+'_sample_sd']:.3f}" for x in ['train_hours','validation_hours','loop_hours']])
        s26.append([label,f"{t['mean_run_median_ms']:.3f} ± {t['sample_sd_of_run_medians_ms']:.3f}",f"{t['min_run_median_ms']:.3f}–{t['max_run_median_ms']:.3f}",f"{t['max_private_gib']:.3f}",f"{t['max_working_set_gib']:.3f}"])
    return s25,s26

def prepare_output(root,out):
    root=root.resolve();out=out.resolve()
    need(not out.exists(),'output must not already exist')
    need(not out.is_relative_to(root) and not root.is_relative_to(out),'output must be separate from package')
    need(out.parent.is_dir(),'output parent must exist')
    return out

def replay(root,out):
    root=root.resolve();out=prepare_output(root,out);count=verify_package(root)
    registry=read(root/'registry.json')['runs']
    need(len(registry)==25 and len({x['run_id'] for x in registry})==25,'duplicate/missing registry run')
    need([(x['method'],x['seed']) for x in registry]==[(m,s) for m in METHODS for s in SEEDS],'registry population/order changed')
    cap,history=capacity(root,registry);per,raw=timing(root,registry,'matrix');qualification,qraw=timing(root,registry,'qualification');latency=aggregate_timing(per)
    expected=read(root/'records/stage87/matrix_summary.json')
    need(expected['warmups_preserved']==125 and expected['measured_observations_preserved']==750 and expected['qualified_observations_excluded_by_design']==175,'count summary mismatch')
    for actual,old in zip(latency,expected['groups']):
        need(actual['method']==old['method'],'method order mismatch')
        for k,v in actual.items():
            if k!='method':close(v,old[k],actual['method']+'/'+k)
    d,p,w=cap[0],cap[1],cap[4];dt,pt,wt=latency[0],latency[1],latency[4]
    ratios={'wide_parameter_excess_vs_pfe':w['trainable_parameters']/p['trainable_parameters']-1,'pfe_MAC_reduction_vs_wide':1-p['conv_MACs_per_image']/w['conv_MACs_per_image'],'pfe_MAC_increase_vs_direct':p['conv_MACs_per_image']/d['conv_MACs_per_image']-1,'pfe_latency_increase_vs_direct':pt['mean_run_median_ms']/dt['mean_run_median_ms']-1,'pfe_latency_reduction_vs_wide':1-pt['mean_run_median_ms']/wt['mean_run_median_ms']}
    paper={'current_manuscript_checked':False,'historical_Word_not_included':True,'scope':'numeric table strings generated; manuscript QA is separate'}
    report={'passed':True,'package_files_verified':count,'training_runs':25,'training_epochs':1250,'capacity_methods':5,'formal_runs':25,'formal_warmups':125,'formal_measurements':750,'qualification_runs':5,'qualification_observations':175,'ratios':ratios,'paper':paper,'comparison_tolerance':{'absolute':1e-12,'relative':1e-12},'scope':'archived records to statistics and numeric table strings; no current manuscript comparison; not re-execution of model, graph, training, timing, weight provenance or device equivalence','new_model_or_GPU_execution':False}
    out.mkdir()
    def save(name,value):
        with (out/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
    save('capacity_summary.json',cap);save('latency_summary.json',latency);save('cost_ratios.json',ratios);save('verification.json',report)
    for name,data in [('training_per_run.csv',history),('latency_per_run.csv',per),('qualification_per_run.csv',qualification),('timing_observations_all1050.csv',raw+qraw)]:
        with (out/name).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    for name,table in zip(['table_S25.md','table_S26.md'],paper_cells(cap,latency)):
        lines=['| '+' | '.join(table[0])+' |','| '+' | '.join('---' for _ in table[0])+' |']+['| '+' | '.join(row)+' |' for row in table[1:]]
        (out/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False));return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    replay(Path(__file__).resolve().parent,args.out)