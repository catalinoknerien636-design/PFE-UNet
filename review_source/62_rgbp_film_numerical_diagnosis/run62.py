"""One bounded, instrumented three-route diagnosis; not a parity retry/admission."""
import pathlib,sys,os,json,gc,contextlib,traceback,math
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'61_rgbp_checkpoint_device_acceptance'));import run61 as a
x=a.x;torch=x.c.t.torch;read,sha=a.read,a.sha
TOL={'atol':1e-5,'rtol':1e-4}
def metrics(left,right):
 if left.shape!=right.shape or left.dtype!=np.float32 or right.dtype!=np.float32:raise ValueError('Matched FP32 shapes required')
 aa=left.reshape(-1);bb=right.reshape(-1);bad=0;maximum=0.;square=0.;n=aa.size
 for i in range(0,n,262144):
  u=torch.from_numpy(np.array(aa[i:i+262144],copy=True));v=torch.from_numpy(np.array(bb[i:i+262144],copy=True))
  if not torch.isfinite(u).all() or not torch.isfinite(v).all():raise ValueError('Nonfinite diagnostic output')
  d=(u-v).abs();bad+=int((~torch.isclose(u,v,**TOL)).sum());maximum=max(maximum,float(d.max()) if d.numel() else 0.);square+=float(d.double().square().sum())
 return dict(elements=n,mismatches=bad,max_abs=maximum,rmse=math.sqrt(square/n) if n else 0.,tolerance=TOL,reference='right')
def selected(model):
 return [(name,m) for name,m in model.named_modules() if name=='' or not list(m.children()) or type(m).__name__ in ('Up','GlobalAffine')]
def save_array(p,v):
 with p.open('xb') as stream:np.lib.format.write_array(stream,np.ascontiguousarray(v),version=(1,0),allow_pickle=False)
def validate():
 c=read(HERE/'contract.json')
 for p,h in c['sources'].items():
  if sha(p)!=h:raise ValueError('Bound source changed '+p)
 tests=read(HERE/'cpu_attempt_v1/report.json')
 if not tests['passed'] or tests['source_sha256']!=c['tested_sources']:raise ValueError('CPU diagnostic checks required')
 frozen,_,_=a.validate();failure=read(ROOT/'61_rgbp_checkpoint_device_acceptance/device_attempt_v1/failure.json')
 if failure['run_id']!=c['run_id'] or failure['automatic_retry'] is not False:raise ValueError('Only inspected failed checkpoint')
 entry=next(e for e in frozen['configs'] if e['config']['run_id']==c['run_id']);return c,entry

def main():
 c,entry=validate();cfg,identity=entry['config'],entry['identity'];torch.set_num_threads(1)
 with contextlib.ExitStack() as stack:
  for p in [x.c.HERE/'queue.lock',*x.q.SHARED_LOCKS]:stack.enter_context(x.q.byte_lock(p))
  x.q.guard.require_idle();x.q.require_resources();out=HERE/'diagnostic_attempt_v1';out.mkdir(exist_ok=False)
  owner=x.q.guard.process_identity(os.getpid());x.c.candidate.save_new(out/'identity.json',dict(owner=owner,contract_sha256=sha(HERE/'contract.json'),synthetic_only=True,test_pixels_read=False,utc=x.c.t.utc_now()))
  try:
   reader=x.safe.Reader(entries=[entry]);best=reader(pathlib.Path(cfg['output_dir'])/'best.pt',map_location='cpu');assert best['identity']==identity
   image=torch.rand(2,6,352,352,generator=torch.Generator().manual_seed(46027));save_array(out/'synthetic_input.npy',image.numpy());manifest={};comparisons={};orders={}
   for route in ('native_cpu','corrected_cpu','corrected_directml'):
    x.q.require_resources();target=out/route;target.mkdir();order=[];rows=[]
    context=contextlib.nullcontext() if route=='native_cpu' else x.c.installed_runtime()
    with context:
     model=x.c.backend.ORIGINAL_BUILD(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed']) if route=='native_cpu' else x.c.t.build_model(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed'])
     model.load_state_dict(best['model_state'],strict=True);x.c.backend.compare_tree(model.state_dict(),best['model_state'],{'atol':0,'rtol':0});model.eval()
     if route=='corrected_directml':device,device_name=x.c.t.choose_device(cfg);model.to(device);inp=image.to(device)
     else:inp=image
     handles=[]
     def hook(name):
      def record(module,args,value):
       if not torch.is_tensor(value):raise ValueError('Unexpected tuple diagnostic output')
       key=name or 'ROOT';array=value.detach().float().cpu().numpy().copy();path=target/(key+'.npy');save_array(path,array);manifest[str(path.relative_to(out))]=sha(path);order.append(key)
       row=dict(module=key,type=type(module).__name__,shape=list(array.shape))
       for refroute in ('native_cpu','corrected_cpu'):
        if refroute==route:break
        ref=np.load(out/refroute/(key+'.npy'),mmap_mode='r',allow_pickle=False);row[refroute+'_vs_'+route]=metrics(ref,array);del ref
       rows.append(row)
      return record
     for name,module in selected(model):handles.append(module.register_forward_hook(hook(name)))
     try:
      with torch.no_grad():result=model(inp)
     finally:
      for h in handles:h.remove()
     del result,inp,model;gc.collect()
    comparisons[route]=rows;orders[route]=order
    if route!='native_cpu' and order!=orders['native_cpu']:raise ValueError('Route module execution differs')
    x.c.candidate.save_new(out/(route+'_layers.json'),rows)
   root={route:np.load(out/route/'ROOT.npy',mmap_mode='r',allow_pickle=False) for route in comparisons}
   summary={pair:metrics(root[l],root[r]) for pair,l,r in [('native_vs_corrected_cpu','native_cpu','corrected_cpu'),('native_vs_directml','native_cpu','corrected_directml'),('corrected_cpu_vs_directml','corrected_cpu','corrected_directml')]}
   for p,h in c['sources'].items():
    if sha(p)!=h:raise ValueError('Source changed after diagnostic '+p)
   x.q.require_resources();x.c.candidate.save_new(out/'report.json',dict(diagnostic_completed=True,run_id=cfg['run_id'],owner=owner,device=device_name,input_sha256=sha(out/'synthetic_input.npy'),best_checkpoint_sha256=sha(pathlib.Path(cfg['output_dir'])/'best.pt'),contract_sha256=sha(HERE/'contract.json'),summary=summary,arrays_sha256=manifest,layer_report_sha256={r:sha(out/(r+'_layers.json')) for r in comparisons},reader_calls=reader.calls,device_admission_changed=False,test_pixels_read=False,training_or_weights_modified=False,utc=x.c.t.utc_now()));print(json.dumps(summary))
  except BaseException as error:
   x.c.candidate.save_new(out/'failure.json',dict(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),owner=owner,automatic_retry=False,test_pixels_read=False,utc=x.c.t.utc_now()));raise
if __name__=='__main__':main()