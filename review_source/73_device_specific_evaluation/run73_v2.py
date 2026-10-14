"""Versioned device-specific evaluation; never changes original failed parity gates."""
from pathlib import Path
import sys,json,hashlib,os,gc,contextlib,traceback,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'66_pcod_device_checks'));import run66 as old
sys.path.insert(0,str(ROOT/'46_rgbp_prediction_export'));import formal_export as rgbp
pcod=old.f;x=old.x;torch=old.torch;sha=old.sha;read=old.read
save=pcod.e.save_new
METRICS=('iou','f1','image_mae','boundary_f1_2px_image_mean','boundary_iou_pooled')

CONTRACT=HERE/'contract_v2.json'
_sealed_sha=None

def load_contract():
 global _sealed_sha
 payload=CONTRACT.read_bytes()
 def pairs(items):
  out={}
  for k,v in items:
   if k in out:raise ValueError('Duplicate contract key')
   out[k]=v
  return out
 value=json.loads(payload.decode('utf-8-sig'),object_pairs_hook=pairs)
 _sealed_sha=hashlib.sha256(payload).hexdigest()
 return value

def contract_sha():
 if _sealed_sha is None or sha(CONTRACT)!=_sealed_sha:raise ValueError('Contract changed since sealed read')
 return _sealed_sha

def frozen_members():
 pc=read(pcod.CONTRACT)['members'];rm=read(rgbp.CONTRACT)['members'];em=read(rgbp.e.offline.HERE/'evaluation_contract.json')['members']
 if [m['sample_id'] for m in rm]!=[m['sample_id'] for m in em]:raise ValueError('Original RGBP input/evaluation population mismatch')
 combined=[]
 for im,mm in zip(rm,em):
  if any(im[k]!=mm[k] for k in im.keys()&mm.keys()):raise ValueError('Conflicting input/mask membership')
  member={**im,**mm}
  if not {'sample_id','image','image_sha256','dolp','dolp_sha256','mask','mask_sha256','stratum'}<=member.keys():raise ValueError('Complete RGBP input/mask/stratum provenance required')
  combined.append(member)
 return {'PCOD':pc,'RGBP':combined}

def exact_population(items,expected):
 ids=[r['sample_id'] for r in items]
 if ids!=expected or len(ids)!=len(set(ids)):raise ValueError('Exact complete ordered population required')

def source_check(c):
 contract_sha()
 for path,h in c['sources'].items():
  if sha(path)!=h:raise ValueError('Frozen source changed: '+path)
 if pcod.e.read(pcod.CONTRACT)['full_inference_path_accepted'] is not False:raise ValueError('Original53 disabled boundary changed')

def contract_check(c):
 if c['members']!=frozen_members():raise ValueError('Complete original input/mask/stratum member mapping required')
 p=read(pcod.protocol.PROTOCOL)['configs'];r=read(x.p.PROTOCOL)['configs']
 if len(p)!=28 or len(r)!=30 or [q['entry'] for q in c['runs']]!=p+r:raise ValueError('All original58 entries in original order required')
 if c['prediction_records']!=45560 or c['original_parity_counts']!={'passed':35,'failed':23}:raise ValueError('Amendment population/count mismatch')
 if c['amendment']['original_failures_relabelled'] or c['amendment']['cross_device_equivalence_claimed']:raise ValueError('Unchanged evidence boundary required')
 if c['statistics']!=statistics_rule():raise ValueError('Statistics specification changed')
 for family,count in [('PCOD',230),('RGBP',1304)]:
  members=c['members'][family]
  if len(members)!=count:raise ValueError('Full test population required')
  exact_population(members,[m['sample_id'] for m in members])
 for row in c['runs']:
  cfg=row['entry']['config'];d=Path(cfg['output_dir']);marker=read(d/'portable_complete.json');status=read(d/'status.json')
  if marker['identity']!=row['entry']['identity'] or marker['test_evaluated'] is not False:raise ValueError('Original completion differs')
  if sha(d/'portable_complete.json')!=row['completion_sha256'] or marker['artifact_sha256']['best.pt']!=row['best_sha256']:raise ValueError('Completion binding differs')
  if row['selected_epoch']!=status['best_epoch'] or row['selected_validation_iou']!=status['best_validation_iou']:raise ValueError('Selection changed')
 return c

def statistics_rule():
 return {'pcod_primary':'PFE minus Direct pooled IoU, five paired seeds, one primary family','rgbp_primary':'PFE minus Direct pooled IoU, three paired seeds within each of two protocols; Holm across those two p-values','seed_order':[42,2026,3407,7819,12011],'standard_backbone_seed_order':[42,2026,3407],'metrics':list(METRICS),'secondary':'Descriptive only: other methods, other metrics, all association strata; no confirmatory superiority claims','test':'Two-sided exhaustive non-mid-p sign-flip; joint sign exchangeability is an assumption; minimum p 0.0625 at n5, 0.25 at n3','reported':'Every pair, mean, sample SD, range, every seed metric; no deletion or test-based selection','independence':'Images are not training repeats; two RGBP variants reuse one test set; none_detected is not scene-independent; no causal leakage claim','probability':'Original53 PCOD device sigmoid then CPU FP32; original46 RGBP detach float CPU then sigmoid; strict >0.5 at352; original45/53 masks/metrics/empty rules unchanged'}

def safe_best(row):
 entry=row['entry'];cfg=entry['config'];d=Path(cfg['output_dir'])
 if sha(d/'best.pt')!=row['best_sha256']:raise ValueError('Checkpoint bytes differ')
 reader=x.safe.Reader(entries=[entry]);best=reader(d/'best.pt',map_location='cpu')
 if set(best)!={'identity','epoch','model_state','validation_iou'} or best['identity']!=entry['identity'] or best['epoch']!=row['selected_epoch'] or best['validation_iou']!=row['selected_validation_iou']:raise ValueError('Validation best identity differs')
 if row['dataset']=='PCOD':old.tensor_check(best['model_state'],read(row['tensor_evidence'])['best_tensor_values'])
 if sha(d/'best.pt')!=row['best_sha256']:raise ValueError('Checkpoint changed during safe read')
 return best,reader.calls

class GuardedLoader:
 def __init__(self,loader):self.loader=loader
 def __iter__(self):
  for b in self.loader:
   x.q.require_resources()
   yield b


def evaluate_bundle(row,c,bundle_path):
 family=row['dataset'];cfg=row['entry']['config'];ep=pcod.e if family=='PCOD' else rgbp.e;members=c['members'][family];ids=[m['sample_id'] for m in members]
 bundle=ep.verify_receipt(bundle_path,contract_sha(),ids)
 if bundle.get('run_id')!=cfg['run_id'] or bundle.get('best_checkpoint_sha256')!=row['best_sha256'] or bundle.get('evaluation_version')!='device_specific73_v2':raise ValueError('Wrong bundle provenance')
 output=bundle_path.parent.parent/'metrics';output.mkdir(exist_ok=False);rows=[]
 try:
  for member,item in zip(members,bundle['items']):
   x.q.require_resources();p=pcod.e.read_probability(bundle_path.parent/item['path'],item['sha256'],[352,352])
   if family=='PCOD':result=pcod.e.image_metrics(item['sample_id'],p,pcod.e.mask_grid(pcod.e.bound_payload(cfg['dataset_root'],member,'mask')))
   else:
    path=(Path(cfg['dataset_root'])/member['mask']).resolve(strict=True)
    if not path.is_relative_to(Path(cfg['dataset_root']).resolve()) or path.stat().st_size>64*1024**2:raise ValueError('Mask path or size')
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=member['mask_sha256']:raise ValueError('Frozen mask changed')
    mask=rgbp.e.cv2.imdecode(np.frombuffer(payload,np.uint8),rgbp.e.cv2.IMREAD_UNCHANGED)
    if mask is None or mask.size>16000000:raise ValueError('Mask decode or size')
    result=pcod.e.metrics.image_metrics(item['sample_id'],p,mask,raw_grid=[352,352])
   rows.append(result)
  exact_population(rows,ids)
  if family=='PCOD':report={'all_test':pcod.e.aggregate(rows)}
  else:report=pcod.e.metrics.complete_report(rows,ids,{m['sample_id']:m['stratum'] for m in members})
  report.update(run_id=cfg['run_id'],dataset=family,method=cfg['method'],seed=cfg['seed'],protocol_variant=cfg.get('protocol_variant','pcod_record_group'),best_checkpoint_sha256=row['best_sha256'],prediction_bundle_sha256=sha(bundle_path),evaluation_contract_sha256=contract_sha(),cross_device_equivalence_claimed=False,original_parity_result=row['original_parity_result'],scene_independence_established=False)
  save(output/'per_image.json',rows);save(output/'summary.json',report)
  save(output/'complete.json',{n:sha(output/n) for n in ['per_image.json','summary.json']})
  return report
 except BaseException as error:
  save(output/'failure.json',{'error':str(error),'rows':len(rows),'automatic_retry':False});raise

def summarize(reports):
 if len(reports)!=58 or len({r['run_id'] for r in reports})!=58:raise ValueError('All58 unique reports required')
 cp=pcod.e.statistics([r for r in reports if r['dataset']=='PCOD']);ev=pcod.e.metrics;rr={};raw={}
 for variant in ev.VARIANTS:
  rows=[r for r in reports if r['dataset']=='RGBP' and r['protocol_variant']==variant]
  expected={(m,s) for m in ('rgb_only','direct','pfe_multiply','learned_fusion','global_film_style') for s in ev.SEEDS}
  if len(rows)!=15 or {(r['method'],r['seed']) for r in rows}!=expected:raise ValueError('Complete RGBP method-seed matrix required')
  index={(r['method'],r['seed']):r['all_test'] for r in rows}
  paired={k:ev.paired_three_seed({s:index['pfe_multiply',s][k] for s in ev.SEEDS},{s:index['direct',s][k] for s in ev.SEEDS}) for k in METRICS}
  raw[variant]=paired['iou']['two_sided_non_mid_p'];rr[variant]={'primary_and_secondary_metrics':paired,'all_method_seed_reports':rows}
 return {'pcod':cp,'rgbp':rr,'rgbp_primary_holm':ev.holm_two_primary(raw),'statistics_rule':statistics_rule(),'completed':58,'prediction_records':45560,'interpretation':'Device-specific test evidence; not cross-device equivalence, not scene-independent evidence.'}

def main():
 torch.set_num_threads(1);x.q.require_resources();x.q.guard.require_idle();c=load_contract();source_check(c);contract_check(c)
 tests=read(HERE/'cpu_tests_v2/report.json');review=read(HERE/'execution_review_v2.json')
 if not tests['passed'] or tests['runner_sha256']!=sha(__file__) or tests['contract_sha256']!=contract_sha():raise ValueError('Bound passing CPU tests required')
 if review.get('approved') is not True or review['contract_sha256']!=contract_sha() or review['tests_sha256']!=sha(HERE/'cpu_tests_v2/report.json'):raise ValueError('Independent reviewed execution contract required')
 with contextlib.ExitStack() as stack:
  old.locks(stack);before=old.b.protection()
  if shutil.disk_usage(HERE).free < (20*2**30+45560*(352*352*4+128)):raise RuntimeError('Need complete export capacity plus original20GiB reserve')
  out=HERE/'attempt_v2';out.mkdir(exist_ok=False);owner=x.q.guard.process_identity(os.getpid());done=[];reports=[];runid=None
  save(out/'identity.json',{'owner':owner,'contract_sha256':contract_sha(),'utc':x.c.t.utc_now(),'original_parity_gate_modified':False})
  try:
   for row in c['runs']:
    x.q.require_resources();cfg=row['entry']['config'];runid=cfg['run_id'];family=row['dataset'];core=pcod.core if family=='PCOD' else x.c
    target=out/runid;target.mkdir(exist_ok=False);ep=pcod.e if family=='PCOD' else rgbp.e;members=c['members'][family];ids=[m['sample_id'] for m in members]
    x.c.t.write_json(out/'progress.json',{'state':'running','current':runid,'completed':done,'owner':owner,'utc':x.c.t.utc_now()})
    best,calls=safe_best(row)
    with core.installed_runtime():
     model=core.t.build_model(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed']);model.load_state_dict(best['model_state'],strict=True)
     (core.backend.h.compare_tree if family=='PCOD' else core.backend.compare_tree)(model.state_dict(),best['model_state'],{'atol':0,'rtol':0})
     device,name=core.t.choose_device(cfg)
     if name!=cfg['expected_device_name'] or cfg['device']!='directml':raise ValueError('Original physical DirectML adapter required')
     model.to(device);model.eval();del best;gc.collect()
     data=ep.BoundInputs(cfg['dataset_root'],members) if family=='PCOD' else ep.BoundInputs(cfg['dataset_root'],members,'rgb' if cfg['method']=='rgb_only' else 'rgb_dolp',352)
     loader=GuardedLoader(torch.utils.data.DataLoader(data,batch_size=4,shuffle=False,drop_last=False,num_workers=0,generator=torch.Generator().manual_seed(53019 if family=='PCOD' else 46027)))
     prov={'evaluation_version':'device_specific73_v2','run_id':runid,'source_stage':25 if family=='PCOD' else 41,'exporter_stage':73,'contract_sha256':contract_sha(),'export_contract_sha256':contract_sha(),'best_checkpoint_sha256':row['best_sha256'],'completion_sha256':row['completion_sha256'],'selected_epoch':row['selected_epoch'],'original_parity_result':row['original_parity_result'],'original_parity_evidence':row['original_parity_evidence'],'cross_device_equivalence_claimed':False,'device_name':name,'device_index':cfg['device_index'],'owner':owner,'reader_calls':calls,'utc':x.c.t.utc_now()}
     def final_check():
      source_check(c)
      if sha(Path(cfg['output_dir'])/'best.pt')!=row['best_sha256'] or sha(Path(cfg['output_dir'])/'portable_complete.json')!=row['completion_sha256']:raise ValueError('Checkpoint changed during export')
     if family=='PCOD':ep.export_batches(model,loader,device,target/'export',ids,prov,final_check)
     else:ep.export_batches(model,loader,device,target/'export',ids,[352,352],prov,final_check)
     del model,device,loader,data;gc.collect()
    report=evaluate_bundle(row,c,target/'export/bundle.json');reports.append(report);done.append(runid)
    save(target/'complete.json',{'export_sha256':sha(target/'export/export_complete.json'),'metrics_sha256':sha(target/'metrics/complete.json'),'run_id':runid,'contract_sha256':contract_sha()})
    print(json.dumps({'completed':len(done),'run_id':runid,'test_iou':report['all_test']['iou']}),flush=True)
   summary=summarize(reports);source_check(c);after=old.b.protection();save(out/'summary.json',summary)
   save(out/'complete.json',{'runs':done,'summary_sha256':sha(out/'summary.json'),'run_receipts_sha256':{rid:sha(out/rid/'complete.json') for rid in done},'original_parity_counts':c['original_parity_counts'],'before_protection':before,'after_protection':after,'utc':x.c.t.utc_now()})
   x.c.t.write_json(out/'progress.json',{'state':'completed','completed':done,'owner':owner,'utc':x.c.t.utc_now()})
  except BaseException as error:
   save(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),'current':runid,'completed':done,'owner':owner,'automatic_retry':False,'original_parity_failure_is_not_execution_failure':True,'utc':x.c.t.utc_now()});raise
if __name__=='__main__':main()
