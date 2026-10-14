"""Synthetic-only checkpoint-specific RGBP device acceptance; never test inference."""
import pathlib,sys,os,json,gc,contextlib,traceback
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'57_rgbp_safe_runtime'));import run57 as x
sys.path.insert(0,str(ROOT/'46_rgbp_prediction_export'));import formal_export as f
read,sha=x.safe.read_json,x.safe.digest

def require_coverage(evidence,queue,ids):
 if not evidence.get('passed') or evidence.get('verified_internal_run_ids')!=ids or len(ids)!=30 or len(set(ids))!=30:raise ValueError('Exact30 internal acceptance required')
 if queue.get('state')!='completed' or queue.get('completed')!=ids or queue.get('test_evaluated') is not False:raise ValueError('Exact30 completed queue required')
def require_tolerance(value):
 if value!={'atol':1e-5,'rtol':1e-4}:raise ValueError('Frozen device tolerance required')
def validate():
 contract=read(HERE/'contract.json')
 for p,h in contract['sources'].items():
  if sha(p)!=h:raise ValueError('Source changed: '+p)
 tests=read(HERE/'cpu_attempt_v1/report.json')
 if not tests['passed'] or tests['gpu_started'] or tests['source_sha256']!=contract['tested_sources']:raise ValueError('CPU gates must pass')
 frozen,_=x.validate();export,_=f.verify_contract();require_tolerance(export['device_parity_tolerance'])
 require_coverage(read(contract['acceptance_report']),read(x.q.STATE),[e['config']['run_id'] for e in frozen['configs']])
 return frozen,contract,export

def main():
 frozen,contract,export=validate();x.c.t.torch.set_num_threads(1)
 with contextlib.ExitStack() as locks:
  for p in [x.c.HERE/'queue.lock',*x.q.SHARED_LOCKS]:locks.enter_context(x.q.byte_lock(p))
  x.q.guard.require_idle();x.q.require_resources()
  out=HERE/'device_attempt_v1';out.mkdir(exist_ok=False)
  owner=x.q.guard.process_identity(os.getpid());passed=[];runid=None
  x.c.candidate.save_new(out/'identity.json',dict(owner=owner,contract_sha256=sha(HERE/'contract.json'),utc=x.c.t.utc_now(),synthetic_only=True,test_pixels_read=False))
  try:
   for entry in frozen['configs']:
    x.q.require_resources();cfg,identity=entry['config'],entry['identity'];runid=cfg['run_id'];d=pathlib.Path(cfg['output_dir'])
    x.c.t.write_json(out/'progress.json',dict(state='running',current_run=runid,completed=passed,owner=owner,utc=x.c.t.utc_now(),test_pixels_read=False))
    marker=read(d/'portable_complete.json');status=read(d/'status.json')
    if marker['identity']!=identity or marker['protocol_sha256']!=sha(x.p.PROTOCOL) or marker['test_evaluated'] is not False:raise ValueError('Completion identity changed')
    for p,h in marker['artifact_sha256'].items():
     if sha(d/p)!=h:raise ValueError('Completed artifact changed')
    reader=x.safe.Reader(entries=[entry]);best=reader(d/'best.pt',map_location='cpu')
    if best['identity']!=identity or best['epoch']!=status['best_epoch'] or best['validation_iou']!=status['best_validation_iou']:raise ValueError('Validation-selected checkpoint changed')
    cpu=x.c.backend.ORIGINAL_BUILD(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed'])
    cpu.load_state_dict(best['model_state'],strict=True);x.c.backend.compare_tree(cpu.state_dict(),best['model_state'],{'atol':0,'rtol':0})
    with x.c.installed_runtime():
     gpu=x.c.t.build_model(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed'])
     gpu.load_state_dict(best['model_state'],strict=True);x.c.backend.compare_tree(gpu.state_dict(),best['model_state'],{'atol':0,'rtol':0})
     device,name=x.c.t.choose_device(cfg);gpu.to(device)
     result=f.device_parity(cpu,gpu,cfg,device,export['device_parity_tolerance'])
    if sha(d/'best.pt')!=marker['artifact_sha256']['best.pt']:raise ValueError('Checkpoint changed during device check')
    result.update(run_id=runid,owner=owner,device=name,best_checkpoint_sha256=sha(d/'best.pt'),completion_sha256=sha(d/'portable_complete.json'),export_contract_sha256=sha(f.CONTRACT),companion_contract_sha256=sha(HERE/'contract.json'),reader_calls=reader.calls,utc=x.c.t.utc_now(),test_pixels_read=False)
    x.c.candidate.save_new(out/(runid+'.json'),result);passed.append(runid)
    del cpu,gpu,best,reader,device;gc.collect()
   x.q.require_resources()
   for p,h in contract['sources'].items():
    if sha(p)!=h:raise ValueError('Source changed after device checks: '+p)
   x.c.candidate.save_new(out/'report.json',dict(passed=True,completed=passed,owner=owner,contract_sha256=sha(HERE/'contract.json'),utc=x.c.t.utc_now(),test_pixels_read=False,formal_inference=False))
   x.c.t.write_json(out/'progress.json',dict(state='completed',completed=passed,owner=owner,utc=x.c.t.utc_now(),test_pixels_read=False))
  except BaseException as error:
   x.c.candidate.save_new(out/'failure.json',dict(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),run_id=runid,completed=passed,owner=owner,contract_sha256=sha(HERE/'contract.json'),utc=x.c.t.utc_now(),automatic_retry=False,test_pixels_read=False))
   raise
if __name__=='__main__':main()