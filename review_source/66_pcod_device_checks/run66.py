"""Safe owned PCOD best reads and synthetic-only original53 device checks."""
import pathlib,sys,json,hashlib,gc,contextlib,os,traceback
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'65_rgbp_remaining_device_checks'));import run65 as prior
sys.path.insert(0,str(ROOT/'53_pcod_formal_evaluation'));import formal53 as f
x=prior.x;b=prior.b;torch=prior.torch;read,sha=prior.read,prior.sha
assert f.core.HERE.resolve()==(ROOT/'25_pcod_portable_runner').resolve()

def tensor_check(state,expected):
 if set(state)!=set(expected):raise ValueError('Tensor key coverage differs')
 for key,tensor in state.items():
  v=tensor.detach().cpu().numpy();e=expected[key]
  if list(v.shape)!=e['shape'] or v.dtype.str!=e['dtype'] or hashlib.sha256(v.tobytes(order='C')).hexdigest()!=e['sha256']:raise ValueError('Independent tensor evidence differs '+key)
 return len(state)

def selected_best(entry,evidence):
 cfg=entry['config'];d=pathlib.Path(cfg['output_dir']);marker=read(d/'portable_complete.json');status=read(d/'status.json')
 if marker['identity']!=entry['identity'] or marker['protocol_sha256']!=sha(f.protocol.PROTOCOL) or marker['test_evaluated'] is not False:raise ValueError('Formal completion differs')
 for name,h in marker['artifact_sha256'].items():
  if sha(d/name)!=h:raise ValueError('Completed artifact changed')
 if sha(d/'portable_complete.json')!=evidence['marker_sha256'] or marker['artifact_sha256']!=evidence['artifacts_sha256']:raise ValueError('Internal evidence does not bind completion')
 reader=x.safe.Reader(entries=[entry]);best=reader(d/'best.pt',map_location='cpu')
 if set(best)!={'identity','epoch','model_state','validation_iou'} or best['identity']!=entry['identity'] or best['epoch']!=status['best_epoch'] or best['validation_iou']!=status['best_validation_iou']:raise ValueError('Selected best identity differs')
 count=tensor_check(best['model_state'],evidence['best_tensor_values'])
 return best,dict(run_id=cfg['run_id'],best_checkpoint_sha256=sha(d/'best.pt'),evidence_tensor_count=count,completion_sha256=sha(d/'portable_complete.json'),reader_calls=reader.calls)

def validate():
 c=read(HERE/'contract.json');b.sources(c);t=read(HERE/'cpu_tests_v1/report.json')
 if not t['passed'] or t['source_sha256']!=c['tested_sources']:raise ValueError('Bound tests required')
 x.validate();formal=f.protocol.validate(f.protocol.read(f.protocol.PROTOCOL));ids=[e['config']['run_id'] for e in formal['configs']]
 q=read(f.core.HERE/'queue_status.json')
 if len(ids)!=28 or ids!=c['run_ids'] or q['state']!='completed' or q['completed']!=ids:raise ValueError('Exact completed28 required')
 export=f.verify_contract();prior.a.require_tolerance(export['device_tolerance'])
 if export['full_inference_path_accepted'] is not False or export['checkpoint_metadata_audit']['restricted_compatibility_verified'] is not False:raise ValueError('Original53 disabled boundary changed')
 return c,formal,export

def locks(stack):
 for p in [x.c.HERE/'queue.lock',*x.q.SHARED_LOCKS]:stack.enter_context(x.q.byte_lock(p))
 x.q.guard.require_idle();x.q.require_resources()

def safe_review():
 torch.set_num_threads(1);x.q.require_resources();x.q.guard.require_idle();c,formal,_=validate()
 with contextlib.ExitStack() as stack:
  locks(stack);before=b.protection();out=HERE/'safe_read_v1';out.mkdir(exist_ok=False);rows=[]
  try:
   for entry in formal['configs']:
    x.q.require_resources();evidence=read(c['evidence_paths'][entry['config']['run_id']]);best,r=selected_best(entry,evidence);rows.append(r);del best;gc.collect()
   b.sources(c);after=b.protection();x.q.require_resources()
   x.c.candidate.save_new(out/'report.json',dict(passed=True,run_ids=c['run_ids'],rows=rows,before_protection=before,after_protection=after,gpu_started=False,pickle_executed=False,contract_sha256=sha(HERE/'contract.json'),utc=x.c.t.utc_now()))
   print(json.dumps(dict(safe_reads=len(rows),tensors=sum(r['evidence_tensor_count'] for r in rows),sha256=sha(out/'report.json'))))
  except BaseException as e:
   x.c.candidate.save_new(out/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),completed=[r['run_id'] for r in rows],automatic_retry=False));raise

def parity(cpu,gpu,device,tol):
 inputs=torch.rand(2,4,352,352,generator=torch.Generator().manual_seed(53019));cpu.eval();gpu.eval()
 with torch.no_grad():
  left,right=cpu(inputs),gpu(inputs.to(device));pa,pb=f.e.probabilities(left,2),f.e.probabilities(right,2)
 left,right=left.float().cpu(),right.float().cpu();logits=prior.compare_outputs(left,right,tol);probs=prior.compare_outputs(torch.from_numpy(pa),torch.from_numpy(pb),tol)
 return dict(passed=logits['passed'] and probs['passed'],samples=2,seed=53019,grid=f.e.GRID,tolerance=tol,test_pixels_read=False,sigmoid_order=f.e.ENCODING,logit_max_abs_difference=logits['logit_max_abs_difference'],probability_max_abs_difference=probs['logit_max_abs_difference'],logit_comparison=logits,probability_comparison=probs),left,right

def device_checks():
 torch.set_num_threads(1);x.q.require_resources();x.q.guard.require_idle();c,formal,export=validate();review=read(HERE/'device_review.json');safe=read(HERE/'safe_read_v1/report.json')
 if review['contract_sha256']!=sha(HERE/'contract.json') or review['safe_report_sha256']!=sha(HERE/'safe_read_v1/report.json') or not safe['passed'] or safe['run_ids']!=c['run_ids']:raise ValueError('Exact safe-read review required')
 with contextlib.ExitStack() as stack:
  locks(stack);before=b.protection();out=HERE/'device_attempt_v1';out.mkdir(exist_ok=False);owner=x.q.guard.process_identity(os.getpid());done=[];passed=[];failed=[];rid=None
  x.c.candidate.save_new(out/'identity.json',dict(owner=owner,contract_sha256=sha(HERE/'contract.json'),review_sha256=sha(HERE/'device_review.json'),utc=x.c.t.utc_now()))
  try:
   for entry in formal['configs']:
    x.q.require_resources();cfg=entry['config'];rid=cfg['run_id'];evidence=read(c['evidence_paths'][rid]);d=pathlib.Path(cfg['output_dir'])
    x.c.t.write_json(out/'progress.json',dict(state='running',current=rid,completed=done,passed=passed,numerical_failed=failed,owner=owner,utc=x.c.t.utc_now()))
    best,binding=selected_best(entry,evidence)
    cpu=f.core.backend.ORIGINAL_BUILD(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed']);cpu.load_state_dict(best['model_state'],strict=True);f.core.backend.h.compare_tree(cpu.state_dict(),best['model_state'],{'atol':0,'rtol':0})
    with f.core.installed_runtime():
     gpu=f.core.t.build_model(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed']);gpu.load_state_dict(best['model_state'],strict=True);f.core.backend.h.compare_tree(gpu.state_dict(),best['model_state'],{'atol':0,'rtol':0})
     device,name=f.core.t.choose_device(cfg);gpu.to(device);result,left,right=parity(cpu,gpu,device,export['device_tolerance'])
    if sha(d/'best.pt')!=binding['best_checkpoint_sha256']:raise ValueError('Best changed during device check')
    arrays={}
    if not result['passed']:
     for label,v in [('cpu',left),('directml',right)]:
      p=out/(rid+'_'+label+'.npy')
      with p.open('xb') as stream:np.lib.format.write_array(stream,v.numpy(),version=(1,0),allow_pickle=False)
      arrays[p.name]=sha(p)
    result.update(**binding,owner=owner,device=name,arrays_sha256=arrays,contract_sha256=sha(HERE/'contract.json'),original53_contract_sha256=sha(f.CONTRACT),utc=x.c.t.utc_now(),formal_inference=False)
    x.c.candidate.save_new(out/(rid+'.json'),result);done.append(rid);(passed if result['passed'] else failed).append(rid)
    print(json.dumps(dict(completed=len(done),run_id=rid,passed=result['passed'],logit_mismatches=result['logit_comparison']['mismatched_elements'],probability_mismatches=result['probability_comparison']['mismatched_elements'])),flush=True)
    del cpu,gpu,best,left,right,device;gc.collect()
   b.sources(c);after=b.protection();x.q.require_resources()
   report=dict(execution_completed=True,completed=done,passed=passed,numerical_failed=failed,all_pcod_checkpoint_checks_passed=len(passed)==28,formal_inference_enabled=False,test_pixels_read=False,contract_sha256=sha(HERE/'contract.json'),results_sha256={rid:sha(out/(rid+'.json')) for rid in done},before_protection=before,after_protection=after,available_ram_gib=x.q.available_memory()/2**30,utc=x.c.t.utc_now())
   x.c.candidate.save_new(out/'report.json',report);x.c.t.write_json(out/'progress.json',dict(state='checks_completed_formal_inference_disabled',completed=done,passed=passed,numerical_failed=failed,owner=owner,utc=x.c.t.utc_now()))
   print(json.dumps(dict(report_sha256=sha(out/'report.json'),checked=len(done),passed=len(passed),failed=len(failed))))
  except BaseException as e:
   x.c.candidate.save_new(out/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),run_id=rid,completed=done,passed=passed,numerical_failed=failed,owner=owner,automatic_retry=False,utc=x.c.t.utc_now()));raise
if __name__=='__main__':
 if sys.argv[1:] == ['safe-review']:safe_review()
 elif sys.argv[1:] == ['device-checks']:device_checks()
 else:raise SystemExit('Explicit safe-review or device-checks required')
