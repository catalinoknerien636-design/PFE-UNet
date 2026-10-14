from pathlib import Path
import sys,json,hashlib,csv
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
import run73_v2 as a
read,sha=a.read,a.sha
pc=read(a.pcod.CONTRACT);rc=read(a.rgbp.CONTRACT);ev=read(a.rgbp.e.offline.HERE/'evaluation_contract.json')
pf=read(a.pcod.protocol.PROTOCOL);rf=read(a.x.p.PROTOCOL);evidence=read(ROOT/'66_pcod_device_checks/contract.json')['evidence_paths']
sources={};runs=[];devices={}
def bind(p,expected=None):
 p=Path(p).resolve(strict=True);h=sha(p)
 if expected is not None and h!=expected:raise ValueError('Source changed '+str(p))
 sources[str(p)]=h;return h
for stage in ['66_pcod_device_checks','61_rgbp_checkpoint_device_acceptance','65_rgbp_remaining_device_checks']:
 d=ROOT/stage/'device_attempt_v1'
 for p in d.glob('*.json'):
  j=read(p)
  if j.get('run_id') and type(j.get('passed')) is bool:devices[j['run_id']]={'passed':j['passed'],'path':str(p),'sha256':bind(p)}
 if stage=='61_rgbp_checkpoint_device_acceptance':
  p=d/'failure.json';j=read(p);devices[j['run_id']]={'passed':False,'path':str(p),'sha256':bind(p),'interpretation':'Original first failed device gate; numerical diagnosis is stopped; original failure retained.'}
for family,protocol,pp,stage in [('PCOD',pf,a.pcod.protocol.PROTOCOL,'25_pcod_portable_runner'),('RGBP',rf,a.x.p.PROTOCOL,'41_rgbp_portable_runner')]:
 bind(pp);q=read(ROOT/stage/'queue_status.json');ids=[e['config']['run_id'] for e in protocol['configs']]
 if q['state']!='completed' or q['completed']!=ids:raise ValueError('Complete frozen queue required')
 bind(ROOT/stage/'queue_status.json')
 for entry in protocol['configs']:
  cfg=entry['config'];d=Path(cfg['output_dir']);m=read(d/'portable_complete.json');st=read(d/'status.json');rid=cfg['run_id']
  if m['identity']!=entry['identity'] or m['test_evaluated'] is not False:raise ValueError('Original identity/boundary')
  for n in ['config.json','status.json','portable_complete.json','history.csv']:bind(d/n)
  row={'dataset':family,'entry':entry,'best_path':str(d/'best.pt'),'best_sha256':m['artifact_sha256']['best.pt'],'completion_sha256':sha(d/'portable_complete.json'),'selected_epoch':st['best_epoch'],'selected_validation_iou':st['best_validation_iou'],'original_parity_result':devices[rid]['passed'],'original_parity_evidence':devices[rid]}
  if family=='PCOD':row['tensor_evidence']=evidence[rid];bind(evidence[rid])
  runs.append(row)
if len(devices)!=58 or sum(d['passed'] for d in devices.values())!=35:raise ValueError('58 original device result bindings required')
# Bind existing internal acceptance chain and owned safe-reader tests; no new checkpoint reads during preparation.
for p in [ROOT/'61_rgbp_checkpoint_device_acceptance/contract.json',ROOT/'66_pcod_device_checks/contract.json',ROOT/'57_rgbp_safe_runtime/contract.json',ROOT/'72_manuscript_completion/paper_rewriting_output/evidence/training_inventory.json',ROOT/'72_manuscript_completion/paper_rewriting_output/evidence/training_sources_sha256.json',ROOT/'72_manuscript_completion/paper_rewriting_output/evaluation_science_review.md',a.pcod.CONTRACT,a.rgbp.CONTRACT,a.rgbp.e.offline.HERE/'evaluation_contract.json',ROOT/'36_pgsnet_pretrained_contract/contract.json',ROOT/'31_pgsnet_checkpoint_safety/active_before.json']:
 bind(p)
ac=read(ROOT/'61_rgbp_checkpoint_device_acceptance/contract.json');bind(ac['acceptance_report'],ac['acceptance_report_sha256'])
if not read(ac['acceptance_report'])['passed'] or ac['rgbp_internal_acceptance']!=30 or ac['pcod_internal_acceptance']!=28:raise ValueError('Internal acceptance incomplete')
for base,c in [(ROOT,pc),(ROOT,rc),(ROOT,ev)]:
 for p,h in c['source_sha256'].items():bind(base/p,h)
for m in tuple(sys.modules.values()):
 p=getattr(m,'__file__',None)
 if p:
  p=Path(p).resolve()
  if p.suffix=='.py' and (p.is_relative_to(ROOT) or (str(p).lower().startswith('<LOCAL_HOST_PATH_05>') and '.venv' not in str(p))):bind(p)
for p in [HERE/'run73_v2.py',HERE/'prepare73_v2.py',HERE/'test73_v2.py']:bind(p)
assert pc['full_inference_path_accepted'] is False
c={'version':'device_specific73_v2','utc':a.x.c.t.utc_now(),'amendment':{'previous_requirement':'All actual checkpoints must pass original CPU/DirectML tolerance before original53/46 formal export','new_requirement':'Separate device-specific evaluation of every validation-best on its original corrected training device/runtime; original tolerance records remain unchanged','reason':'User stopped further numerical-difference diagnosis and authorized continued core research/manuscript. Same original training backend answers a bounded empirical-performance question without asserting cross-device equivalence. All58 remain included.','original53_modified':False,'original_failures_relabelled':False,'cross_device_equivalence_claimed':False,'training_or_selection_or_test_population_changed':False,'test_pixels_read_before_freeze':False},'original_parity_counts':{'passed':35,'failed':23},'runs':runs,'members':a.frozen_members(),'prediction_records':45560,'statistics':a.statistics_rule(),'execution':{'order':'Original PCOD28 then original RGBP30','device':'Original cfg DirectML index and expected adapter name','grid':[352,352],'batch_size':4,'tail_batch':'retain','workers':0,'shuffle':False,'mode':'eval and no_grad, no autocast, original FP32 state','data':'original byte-bound inputs and masks, no TTA, no exclusion','resource_thresholds_gib':{'physical_ram':8,'disk':20},'locks':'original four exclusive locks held for whole serial attempt','export_bytes_projection':45560*(352*352*4+128),'read_mode':'owned symbolic Reader57, no executable pickle','complete_condition':'All58 transactions, exactly45560 predictions and complete metrics with bound SHA receipts','fault_policy':'stop and preserve any partial output, no auto-retry or overwrite; later continuation only via separate reviewed version binding completed and failed attempts; no score-based retry','diagnostics':'No additional cross-device, FP64 or operator-difference checks','writer_receipt_schema':'Original53/46 primitive receipt stage IDs identify immutable serialization implementation; bundle exporter_stage73 and evaluation_version identify new execution and amendment'},'sources':sources}
a.contract_check(c);a.save(HERE/'contract_v2.json',c)
with (HERE/'run_manifest_v2.csv').open('x',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['dataset','run_id','method','seed','variant','selected_epoch','selected_validation_iou','best_path','best_sha256','original_parity_result']);w.writeheader()
 for r in runs:
  cfg=r['entry']['config'];w.writerow({'dataset':r['dataset'],'run_id':cfg['run_id'],'method':cfg['method'],'seed':cfg['seed'],'variant':cfg.get('protocol_variant','pcod_record_group'),**{k:r[k] for k in ['selected_epoch','selected_validation_iou','best_path','best_sha256','original_parity_result']}})
a.save(HERE/'preparation_v2.json',{'contract_sha256':sha(HERE/'contract_v2.json'),'sources':len(sources),'runs':len(runs),'prediction_records':45560,'GPU_used':False,'checkpoint_loaded':False,'test_pixels_read':False})
print(json.dumps({'contract_sha256':sha(HERE/'contract_v2.json'),'sources':len(sources),'runs':len(runs)}))
