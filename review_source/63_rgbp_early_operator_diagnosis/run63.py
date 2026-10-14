"""CPU-only local operator decomposition using immutable stage62 saved operands."""
import pathlib,sys,json,copy,gc,os,contextlib,traceback,math,shutil
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'62_rgbp_film_numerical_diagnosis'));import run62 as a
x=a.x;torch=a.torch;read,sha=a.read,a.sha

def fp64_error(value,reference):
 if value.shape!=reference.shape or reference.dtype!=torch.float64:raise ValueError('Matched FP64 reference required')
 v=value.reshape(-1);r=reference.reshape(-1);maximum=0.;squares=0.;n=v.numel()
 for i in range(0,n,262144):
  delta=v[i:i+262144].double()-r[i:i+262144]
  if not torch.isfinite(delta).all():raise ValueError('Nonfinite diagnostic values')
  maximum=max(maximum,float(delta.abs().max()) if delta.numel() else 0.);squares+=float(delta.square().sum())
 return dict(elements=n,max_abs=maximum,rmse=math.sqrt(squares/n) if n else 0.)

def comparison(left,right):return a.metrics(left.detach().numpy(),right.detach().numpy())

def diagnose(op,ci,gi,co,go):
 if isinstance(op,torch.nn.ReLU):op=copy.deepcopy(op);op.inplace=False
 with torch.no_grad():
  replay=op(ci);local=op(gi)
  replay_error=comparison(replay,co)
  if replay_error['max_abs']!=0:raise ValueError('Saved CPU output not reproduced exactly')
  inherited=comparison(replay,local);local_error=comparison(local,go);total=comparison(co,go)
  reference=copy.deepcopy(op).double()(gi.double())
  return dict(cpu_replay_exact=True,inherited_input_difference=inherited,local_same_gpu_operands_difference=local_error,total_saved_path_difference=total,cpu_fp32_to_fp64=fp64_error(local,reference),saved_directml_to_fp64=fp64_error(go,reference),reference='CPU FP64 on identical saved DirectML input and original checkpoint parameters; numerical reference, not an exact-real oracle')

def protection():
 counts={}
 def verify(name,m,root=ROOT):
  for p,h in m.items():
   if sha(root/pathlib.Path(p))!=h:raise ValueError('Protected source changed '+str(p))
  counts[name]=len(m)
 verify('protected',read(ROOT/'36_pgsnet_pretrained_contract/contract.json')['protected_files'])
 for stage,manifest in [('53_pcod_formal_evaluation','artifact_manifest.json'),('54_pcod_safe_checkpoint_adapter','artifact_manifest.json'),('55_pcod_checkpoint_evidence','preparation_manifest.json')]:verify(stage,read(ROOT/stage/manifest)['files_sha256'],ROOT/stage)
 verify('environment',read(ROOT/'31_pgsnet_checkpoint_safety/active_before.json'))
 verify('six_manifests',{v['path']:v['sha256'] for parts in read(ROOT/'40_rgbp_formal_preparation/input_contract.json')['manifests'].values() for v in parts.values()})
 return counts

def sources(c):
 for p,h in c['sources'].items():
  if sha(p)!=h:raise ValueError('Bound source changed '+p)

def main():
 torch.set_num_threads(1);x.q.require_resources();x.q.guard.require_idle()
 c=read(HERE/'contract.json');sources(c)
 tests=read(HERE/'cpu_tests_v1/report.json')
 if not tests['passed'] or tests['source_sha256']!=c['tested_sources']:raise ValueError('Bound unit checks required')
 with contextlib.ExitStack() as stack:
  for p in [x.c.HERE/'queue.lock',*x.q.SHARED_LOCKS]:stack.enter_context(x.q.byte_lock(p))
  x.q.guard.require_idle();x.q.require_resources();before=protection()
  out=HERE/'analysis_attempt_v1';out.mkdir(exist_ok=False)
  x.c.candidate.save_new(out/'identity.json',dict(owner=x.q.guard.process_identity(os.getpid()),utc=x.c.t.utc_now(),contract_sha256=sha(HERE/'contract.json'),gpu_forwards=0))
  try:
   frozen,_,_=a.a.validate();entry=next(e for e in frozen['configs'] if e['config']['run_id']==c['run_id']);cfg=entry['config']
   reader=x.safe.Reader(entries=[entry]);best=reader(pathlib.Path(cfg['output_dir'])/'best.pt',map_location='cpu')
   if sha(pathlib.Path(cfg['output_dir'])/'best.pt')!=c['best_checkpoint_sha256']:raise ValueError('Checkpoint changed')
   model=x.c.backend.ORIGINAL_BUILD(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed']);model.load_state_dict(best['model_state'],strict=True);model.eval()
   mods=dict(model.named_modules());D=ROOT/'62_rgbp_film_numerical_diagnosis/diagnostic_attempt_v1'
   image=torch.from_numpy(np.load(D/'synthetic_input.npy',allow_pickle=False));rows=[]
   def load(route,name):return torch.from_numpy(np.load(D/route/(name+'.npy'),allow_pickle=False))
   for i in range(6):
    x.q.require_resources();name='backbone.rgb_inc.net.'+str(i);op=mods[name]
    if not isinstance(op,(torch.nn.Conv2d,torch.nn.BatchNorm2d,torch.nn.ReLU)):raise ValueError('Unexpected operator')
    if i==0:ci=image[:,:3];gi=image[:,:3]
    else:ci=load('corrected_cpu','backbone.rgb_inc.net.'+str(i-1));gi=load('corrected_directml','backbone.rgb_inc.net.'+str(i-1))
    co=load('corrected_cpu',name);go=load('corrected_directml',name)
    row=dict(module=name,type=type(op).__name__,shape=list(co.shape),parameters=str(op),metrics=diagnose(op,ci,gi,co,go))
    x.c.candidate.save_new(out/(str(i)+'.json'),row);rows.append(row)
    del ci,gi,co,go;gc.collect()
   sources(c);after=protection();x.q.require_resources()
   report=dict(completed=True,utc=x.c.t.utc_now(),contract_sha256=sha(HERE/'contract.json'),run_id=c['run_id'],rows=rows,rows_sha256={str(i)+'.json':sha(out/(str(i)+'.json')) for i in range(6)},before_protection=before,after_protection=after,available_ram_gib=x.q.available_memory()/2**30,free_disk_gib=shutil.disk_usage(ROOT).free/2**30,gpu_forwards=0,test_pixels_read=False,device_admission_changed=False,weights_or_environment_modified=False,reader_calls=reader.calls)
   x.c.candidate.save_new(out/'report.json',report);print(json.dumps(dict(report=str(out/'report.json'),sha256=sha(out/'report.json'),rows=rows)))
  except BaseException as error:
   x.c.candidate.save_new(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),automatic_retry=False,utc=x.c.t.utc_now()));raise
if __name__=='__main__':main()
