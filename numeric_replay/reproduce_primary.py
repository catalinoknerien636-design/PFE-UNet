"""Local review replay adapted from stage 78; numerical records only; no model execution."""
from pathlib import Path
from fractions import Fraction
import argparse, collections, csv, hashlib, itertools, json, math, statistics, sys

METRICS = ('iou','f1','image_mae','boundary_f1_2px_image_mean','boundary_iou_pooled')
BOUNDARY = ('boundary_f1_1px','boundary_f1_2px','boundary_f1_4px','boundary_iou')
PROTOCOLS = ('pcod_record_group','official_dateblock_reference','association_quarantine_sensitivity_v2')

def require(ok, message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def compare(a,b,label='root'):
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),label+': keys differ')
        for key in b: compare(a[key],b[key],label+'.'+key)
    elif isinstance(b,list):
        require(isinstance(a,list) and len(a)==len(b),label+': list length differs')
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,f'{label}[{i}]')
    elif isinstance(b,bool): require(type(a) is bool and a==b,label+': boolean differs')
    elif isinstance(b,(int,float)):
        require(type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),label+': numeric mismatch')
    else: require(a==b,label+': value differs')

def verify_manifest(root):
    manifest=read(root/'MANIFEST.json'); seen=set()
    for row in manifest['files']:
        name=row['path']; p=(root/name).resolve()
        require(name not in seen and p.is_relative_to(root.resolve()),'invalid package path')
        require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'package checksum mismatch: '+name)
        seen.add(name)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual==seen|{'MANIFEST.json'},'unexpected or missing package files')
    return manifest

def membership(rows,expected):
    ids=[r['sample_id'] for r in rows]
    require(len(ids)==len(set(ids)) and ids==expected,'missing, duplicate, extra or reordered sample IDs')

def aggregate(rows):
    require(bool(rows),'empty stratum cannot be averaged')
    require(len({r['sample_id'] for r in rows})==len(rows),'duplicate image')
    for r in rows:
        for k in ('tp','fp','fn','tn','boundary_intersection','boundary_union','evaluated_pixels'):
            require(type(r[k]) is int and r[k]>=0,'invalid count: '+k)
        require(sum(r[k] for k in ('tp','fp','fn','tn'))==r['evaluated_pixels']==352*352,'wrong evaluation grid')
        require(r['boundary_intersection']<=r['boundary_union'],'invalid boundary counts')
        for k in ('mae',*BOUNDARY): require(type(r[k]) in (int,float) and math.isfinite(r[k]) and 0<=r[k]<=1,'nonfinite or out-of-range metric')
        require(r['pred_empty'] in (0,1) and r['gt_empty'] in (0,1),'invalid empty flag')
        require(bool(r['pred_empty'])==(r['tp']+r['fp']==0),'prediction empty flag mismatch')
        require(bool(r['gt_empty'])==(r['tp']+r['fn']==0),'target empty flag mismatch')
    tp,fp,fn,tn=(sum(r[k] for r in rows) for k in ('tp','fp','fn','tn'))
    inter,union=(sum(r[k] for r in rows) for k in ('boundary_intersection','boundary_union'))
    return {'images':len(rows),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'iou':tp/(tp+fp+fn+1e-7),
        'f1':2*tp/(2*tp+fp+fn+1e-7),'precision':tp/(tp+fp+1e-7),'recall':tp/(tp+fn+1e-7),
        'image_mae':sum(r['mae'] for r in rows)/len(rows),
        **{k+'_image_mean':sum(r[k] for r in rows)/len(rows) for k in BOUNDARY},
        'boundary_iou_pooled':inter/union if union else 1.,
        'both_empty_images':sum(r['pred_empty'] and r['gt_empty'] for r in rows),
        'exactly_one_empty_images':sum(r['pred_empty']!=r['gt_empty'] for r in rows)}

def exact_sign_flip(differences):
    require(bool(differences) and all(math.isfinite(x) for x in differences),'invalid paired differences')
    values=[Fraction.from_float(float(x)) for x in differences]
    observed=abs(sum(values)); count=0
    for signs in itertools.product((-1,1),repeat=len(values)):
        count+=abs(sum(s*x for s,x in zip(signs,values)))>=observed
    return count/(2**len(values))

def summaries(rows):
    groups=collections.defaultdict(list)
    for r in rows: groups[(r['protocol'],r['method'])].append(r)
    result=[]
    for (protocol,method),part in groups.items():
        expected=[42,2026,3407] if protocol!='pcod_record_group' or method=='deeplabv3plus_resnet50' else [42,2026,3407,7819,12011]
        require([r['seed'] for r in part]==expected,'seed membership/order mismatch')
        out={'protocol':protocol,'method':method,'n':len(part),'seeds':expected}
        for k in METRICS:
            values=[r[k] for r in part];out[k+'_mean']=statistics.mean(values);out[k+'_sample_sd']=statistics.stdev(values)
        result.append(out)
    require(len(result)==16,'expected 16 method/protocol groups')
    contrasts=[]; pairs=[]
    for protocol in PROTOCOLS:
        dm,pm=('direct_reference','pfe_reference') if protocol==PROTOCOLS[0] else ('direct','pfe_multiply')
        direct={r['seed']:r for r in groups[(protocol,dm)]};pfe={r['seed']:r for r in groups[(protocol,pm)]}
        require(set(direct)==set(pfe),'unpaired seeds')
        for k in METRICS:
            seeds=sorted(direct); diffs=[pfe[s][k]-direct[s][k] for s in seeds]
            contrasts.append({'protocol':protocol,'metric':k,'seeds':seeds,'differences':diffs,'mean':statistics.mean(diffs),
                'sample_sd':statistics.stdev(diffs),'minimum':min(diffs),'maximum':max(diffs),
                'positive':sum(x>0 for x in diffs),'negative':sum(x<0 for x in diffs),'n':len(diffs),
                'two_sided_non_mid_p':exact_sign_flip(diffs),'confirmatory_endpoint':k=='iou'})
            if k=='iou': pairs.extend({'protocol':protocol,'seed':s,'direct_test_iou':direct[s][k],
                'pfe_test_iou':pfe[s][k],'pfe_minus_direct':pfe[s][k]-direct[s][k]} for s in seeds)
    raw=sorted((c['two_sided_non_mid_p'],c['protocol']) for c in contrasts if c['metric']=='iou' and c['protocol']!=PROTOCOLS[0])
    first=min(1.,2*raw[0][0]); holm={raw[0][1]:first,raw[1][1]:max(first,raw[1][0])}
    return result,contrasts,pairs,holm

def backbone_subset(rows):
    seeds=[42,2026,3407];part=[r for r in rows if r['protocol']==PROTOCOLS[0]]
    pf={r['seed']:r['iou'] for r in part if r['method']=='pfe_reference'}
    bb={r['seed']:r['iou'] for r in part if r['method']=='deeplabv3plus_resnet50'}
    p=[pf[s] for s in seeds];b=[bb[s] for s in seeds];d=[x-y for x,y in zip(p,b)]
    return {'seeds':seeds,'pfe_iou':p,'backbone_iou':b,'differences':d,'pfe_mean':statistics.mean(p),
        'pfe_sd':statistics.stdev(p),'backbone_mean':statistics.mean(b),'backbone_sd':statistics.stdev(b),
        'difference_mean':statistics.mean(d),'difference_sd':statistics.stdev(d),
        'interpretation':'descriptive matched three-seed secondary comparison; no new model inference'}

def csv_check(path,rows):
    with path.open(newline='',encoding='utf-8-sig') as f: expected=list(csv.DictReader(f))
    require(len(expected)==len(rows),'CSV row count differs: '+path.name)
    for i,(a,b) in enumerate(zip(rows,expected)):
        require(set(a)==set(b),'CSV schema mismatch')
        for k,x in a.items():
            y=b[k]
            if isinstance(x,bool): require(y==str(x),'CSV flag mismatch')
            elif isinstance(x,(int,float)): compare(x,float(y),f'{path.name}:{i}:{k}')
            elif isinstance(x,list): compare(x,json.loads(y),'CSV seed list')
            else: require(str(x)==y,'CSV value differs')

def analyze(root):
    manifest=verify_manifest(root); contract=read(root/'evidence/contract_v2.json');completion=read(root/'evidence/complete.json')
    require(sha(root/'evidence/contract_v2.json')=='1b93b1d2edb9a8a2a9a6f9b71d7f5d919e7684fd2253f653bcfc2f0c9d04d35e','wrong frozen contract')
    require(sha(root/'evidence/summary.json')==completion['summary_sha256'],'executor summary binding differs')
    runs=contract['runs'];ids=[r['entry']['config']['run_id'] for r in runs]
    require(len(ids)==58 and len(set(ids))==58 and ids==completion['runs'],'run population differs')
    compare(contract['original_parity_counts'],{'passed':35,'failed':23})
    for dataset,n in [('PCOD',230),('RGBP',1304)]:
        require(len(contract['members'][dataset])==n,'test population differs')
    rows=[]; strata_out={}; total=0; source_bindings=read(root/'expected/source_bindings.json')
    for run in runs:
        cfg=run['entry']['config'];rid=cfg['run_id'];base=root/'records'/rid
        receipt=read(base/'run_complete.json');metric_receipt=read(base/'metrics_complete.json')
        require(sha(base/'run_complete.json')==completion['run_receipts_sha256'][rid],'run receipt binding differs')
        require(receipt['contract_sha256']==sha(root/'evidence/contract_v2.json') and receipt['run_id']==rid,'run contract differs')
        require(sha(base/'metrics_complete.json')==receipt['metrics_sha256'],'metric receipt differs')
        for name in ('per_image.json','summary.json'): require(sha(base/name)==metric_receipt[name],'metric input checksum differs')
        require(sha(base/'export_complete.json')==receipt['export_sha256'],'export receipt differs')
        ex=read(base/'export_complete.json');bundle=read(base/'bundle.json')
        require(sha(base/'bundle.json')==ex['bundle_sha256'],'bundle binding differs')
        expected=source_bindings[rid]
        for field,name in [('summary_sha256','summary.json'),('per_image_sha256','per_image.json'),('bundle_sha256','bundle.json')]:
            require(sha(base/name)==expected[field],'stage74 source binding differs')
        per=read(base/'per_image.json');report=read(base/'summary.json');members=contract['members'][run['dataset']]
        ids_expected=[m['sample_id'] for m in members];membership(per,ids_expected);membership(bundle['items'],ids_expected)
        require(report['best_checkpoint_sha256']==bundle['best_checkpoint_sha256']==run['best_sha256'],'checkpoint identity differs')
        require(report['prediction_bundle_sha256']==sha(base/'bundle.json'),'prediction manifest identity differs')
        require(report['evaluation_contract_sha256']==receipt['contract_sha256'],'report contract differs')
        require(report['run_id']==rid and report['dataset']==run['dataset'] and report['method']==cfg['method'] and report['seed']==cfg['seed'],'run identity differs')
        require(report['original_parity_result']==run['original_parity_result'],'original failure flag changed')
        require(report['cross_device_equivalence_claimed'] is False and report['scene_independence_established'] is False,'claim boundary changed')
        agg=aggregate(per);compare(agg,report['all_test'],rid+'.all_test');total+=len(per)
        if run['dataset']=='RGBP':
            strata={m['sample_id']:m['stratum'] for m in members}; out={}
            for name in ('train','validation','none_detected'):
                part=[r for r in per if strata[r['sample_id']]==name];out[name]=aggregate(part) if part else None
            compare(out,report['association_strata'],rid+'.strata');strata_out[rid]=out
        rows.append({'dataset':run['dataset'],'protocol':cfg.get('protocol_variant','pcod_record_group'),
            'method':cfg['method'],'seed':cfg['seed'],'selected_epoch':run['selected_epoch'],
            'original_parity_passed':run['original_parity_result'],**agg})
    require(total==45560 and sum(r['original_parity_passed'] for r in rows)==35,'record coverage differs')
    groups,contrasts,pairs,holm=summaries(rows);bb=backbone_subset(rows)
    for name,actual in [('run_metrics.json',rows),('group_summary.json',groups),('paired_comparisons.json',contrasts),('standard_backbone_matched_subset.json',bb)]:
        compare(actual,read(root/'expected'/name),name)
    for name,actual in [('all58_test_results.csv',rows),('group_summary.csv',groups),('source_data_fig8.csv',pairs)]: csv_check(root/'expected'/name,actual)
    compare(holm,read(root/'expected/independent_result_audit.json')['rgbp_primary_holm'],'Holm')
    report={'passed':True,'runs':58,'per_image_records':total,'method_protocol_groups':len(groups),
        'descriptive_rgbp_strata':len(strata_out)*3,'paired_metric_comparisons':len(contrasts),'figure8_pairs':len(pairs),
        'primary':[c for c in contrasts if c['metric']=='iou'],'rgbp_primary_holm':holm,
        'original_device_check_counts':{'passed':35,'failed':23},'package_files_verified':len(manifest['files']),
        'tolerance':{'relative':1e-12,'absolute':1e-12},'new_inference':False,'training':False,
        'probability_arrays_rehashed':False,'image_or_mask_decoding':False,
        'scope':'reaggregation of saved metric records; not fresh pixel metrics, model replay, or cross-device equivalence'}
    return {'verification.json':report,'run_metrics.json':rows,'group_summary.json':groups,
        'paired_comparisons.json':contrasts,'association_strata.json':strata_out,
        'standard_backbone_matched_subset.json':bb,'rgbp_primary_holm.json':holm},pairs

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parent;out=args.out.resolve()
    require(not out.exists(),'output already exists; choose a new directory')
    require(not out.is_relative_to(root),'output must be outside the immutable package')
    results,pairs=analyze(root);out.mkdir(parents=True,exist_ok=False)
    for name,value in results.items():
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,value in [('all58_test_results.csv',results['run_metrics.json']),('group_summary.csv',results['group_summary.json']),('source_data_fig8.csv',pairs)]:
        with (out/name).open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(value[0]));writer.writeheader();writer.writerows(value)
    print(json.dumps({'passed':True,'runs':58,'records':45560,'groups':16,'output':str(out)},ensure_ascii=False))

if __name__=='__main__':
    try: main()
    except (ValueError,KeyError,OSError) as error:
        print('REPRODUCTION FAILED: '+str(error),file=sys.stderr);sys.exit(1)
