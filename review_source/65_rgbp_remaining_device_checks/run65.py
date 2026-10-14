"""One pass over 16 previously untested checkpoints; preserve original failed gate."""
import pathlib,sys,json,os,gc,contextlib,traceback
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'63_rgbp_early_operator_diagnosis'));import run63 as b
a=b.a.a;x=a.x;f=a.f;torch=b.torch;read,sha=b.read,b.sha

def remaining(ids,failure):
 if len(ids)!=30 or len(set(ids))!=30 or failure['completed']!=ids[:13] or failure['run_id']!=ids[13] or failure['automatic_retry'] is not False:raise ValueError('Exact original 13-pass/1-fail boundary required')
 return ids[14:]

def compare_outputs(left,right,tolerance):
 a.require_tolerance(tolerance)
 if left.shape!=right.shape or left.dtype!=right.dtype or left.dtype!=torch.float32:raise ValueError('Shape/dtype mismatch is an execution failure')
 if not torch.isfinite(left).all() or not torch.isfinite(right).all():raise ValueError('Nonfinite synthetic device parity')
 error=None
 try:torch.testing.assert_close(left,right,**tolerance)
 except AssertionError as e:error=str(e)
 mismatches=int((~torch.isclose(left,right,**tolerance)).sum())
 if (error is None)!=(mismatches==0):raise ValueError('Comparison implementations disagree')
 return dict(passed=error is None,mismatched_elements=mismatches,elements=left.numel(),logit_max_abs_difference=float((left-right).abs().max()),assertion=error)

def parity(cpu,gpu,cfg,device,tolerance):
 channels=3 if cfg['method']=='rgb_only' else 6
 image=torch.rand(2,channels,352,352,generator=torch.Generator().manual_seed(46027))
 cpu.eval();gpu.eval()
 with torch.no_grad():left=cpu(image).float().cpu();right=gpu(image.to(device)).float().cpu()
 r=compare_outputs(left,right,tolerance)
 pa,pb=f.e.probabilities(left,2,[352,352]),f.e.probabilities(right,2,[352,352])
 r.update(raw_grid=[352,352],samples=2,seed=46027,probability_max_abs_difference=float(abs(pa-pb).max()),tolerance=tolerance,test_pixels_read=False)
 return r,left,right

def validate():
 c=read(HERE/'contract.json');b.sources(c);tests=read(HERE/'cpu_tests_v1/report.json')
 if not tests['passed'] or tests['source_sha256']!=c['tested_sources']:raise ValueError('Bound CPU checks required')
 frozen,old,export=a.validate();ids=[e['config']['run_id'] for e in frozen['configs']];failure=read(a.HERE/'device_attempt_v1/failure.json')
 if remaining(ids,failure)!=c['remaining_run_ids']:raise ValueError('Changed remaining scope')
 for runid in ids[:13]:
  r=read(a.HERE/'device_attempt_v1'/(runid+'.json'))
  if r['run_id']!=runid or r['passed'] is not True or r['tolerance']!=export['device_parity_tolerance'] or r['test_pixels_read'] is not False:raise ValueError('Prior pass evidence changed')
 return frozen,c,export

def main():
 torch.set_num_threads(1);x.q.require_resources();x.q.guard.require_idle();frozen,c,export=validate()
 with contextlib.ExitStack() as stack:
  for lock in [x.c.HERE/'queue.lock',*x.q.SHARED_LOCKS]:stack.enter_context(x.q.byte_lock(lock))
  x.q.guard.require_idle();x.q.require_resources();before=b.protection();out=HERE/'device_attempt_v1';out.mkdir(exist_ok=False)
  owner=x.q.guard.process_identity(os.getpid());done=[];passed=[];failed=[];runid=None
  x.c.candidate.save_new(out/'identity.json',dict(owner=owner,utc=x.c.t.utc_now(),contract_sha256=sha(HERE/'contract.json'),synthetic_only=True,test_pixels_read=False))
  try:
   for entry in frozen['configs'][14:]:
    x.q.require_resources();cfg,identity=entry['config'],entry['identity'];runid=cfg['run_id'];d=pathlib.Path(cfg['output_dir'])
    x.c.t.write_json(out/'progress.json',dict(state='running',current_run=runid,completed=done,passed=passed,numerical_failed=failed,owner=owner,utc=x.c.t.utc_now()))
    marker=read(d/'portable_complete.json');status=read(d/'status.json')
    if marker['identity']!=identity or marker['protocol_sha256']!=sha(x.p.PROTOCOL) or marker['test_evaluated'] is not False:raise ValueError('Completion identity changed')
    for p,h in marker['artifact_sha256'].items():
     if sha(d/p)!=h:raise ValueError('Completed artifact changed')
    reader=x.safe.Reader(entries=[entry]);best=reader(d/'best.pt',map_location='cpu')
    if best['identity']!=identity or best['epoch']!=status['best_epoch'] or best['validation_iou']!=status['best_validation_iou']:raise ValueError('Validation-selected checkpoint changed')
    cpu=x.c.backend.ORIGINAL_BUILD(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed']);cpu.load_state_dict(best['model_state'],strict=True);x.c.backend.compare_tree(cpu.state_dict(),best['model_state'],{'atol':0,'rtol':0})
    with x.c.installed_runtime():
     gpu=x.c.t.build_model(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed']);gpu.load_state_dict(best['model_state'],strict=True);x.c.backend.compare_tree(gpu.state_dict(),best['model_state'],{'atol':0,'rtol':0})
     device,name=x.c.t.choose_device(cfg);gpu.to(device);result,left,right=parity(cpu,gpu,cfg,device,export['device_parity_tolerance'])
    if sha(d/'best.pt')!=marker['artifact_sha256']['best.pt']:raise ValueError('Checkpoint changed during device check')
    arrays={}
    if not result['passed']:
     for label,value in [('cpu',left),('directml',right)]:
      path=out/(runid+'_'+label+'.npy')
      with path.open('xb') as stream:np.lib.format.write_array(stream,value.numpy(),version=(1,0),allow_pickle=False)
      arrays[path.name]=sha(path)
    result.update(run_id=runid,owner=owner,device=name,best_checkpoint_sha256=sha(d/'best.pt'),completion_sha256=sha(d/'portable_complete.json'),export_contract_sha256=sha(f.CONTRACT),companion_contract_sha256=sha(HERE/'contract.json'),reader_calls=reader.calls,arrays_sha256=arrays,utc=x.c.t.utc_now(),formal_inference=False)
    x.c.candidate.save_new(out/(runid+'.json'),result);done.append(runid);(passed if result['passed'] else failed).append(runid)
    print(json.dumps(dict(completed=len(done),run_id=runid,passed=result['passed'],mismatches=result['mismatched_elements'])),flush=True)
    del cpu,gpu,best,reader,device,left,right;gc.collect()
   if done!=c['remaining_run_ids']:raise ValueError('Incomplete inspected scope')
   b.sources(c);after=b.protection();x.q.require_resources()
   r=dict(execution_completed=True,completed=done,passed_run_ids=passed,numerical_failed_run_ids=failed,combined_rgbp_counts=dict(checked=30,passed=13+len(passed),failed=1+len(failed),unattempted=0),all_rgbp_admitted=False,prior_failed_run_id=c['prior_failed_run_id'],test_pixels_read=False,formal_inference=False,owner=owner,contract_sha256=sha(HERE/'contract.json'),results_sha256={runid:sha(out/(runid+'.json')) for runid in done},before_protection=before,after_protection=after,available_ram_gib=x.q.available_memory()/2**30,utc=x.c.t.utc_now())
   x.c.candidate.save_new(out/'report.json',r);x.c.t.write_json(out/'progress.json',dict(state='checks_completed_admission_blocked',completed=done,passed=passed,numerical_failed=failed,owner=owner,utc=x.c.t.utc_now()))
   print(json.dumps(dict(report=str(out/'report.json'),sha256=sha(out/'report.json'),combined=r['combined_rgbp_counts'])))
  except BaseException as error:
   x.c.candidate.save_new(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),run_id=runid,completed=done,passed=passed,numerical_failed=failed,owner=owner,automatic_retry=False,test_pixels_read=False,utc=x.c.t.utc_now()));raise
if __name__=='__main__':main()
