from pathlib import Path
import sys,json,hashlib,os,gc,shutil,contextlib,msvcrt,traceback,collections
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'36_pgsnet_pretrained_contract'));import adapter as a
sys.path.insert(0,str(R/'24_pcod_reference_acceptance'));import process_guard as guard

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,value):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def resources(minimum=8):
 ram=a.memory_guard(minimum);disk=shutil.disk_usage(H).free/2**30
 if disk<20:raise RuntimeError('Below original20GiB disk reserve')
 return {'available_ram_gib':ram,'disk_free_gib':disk}
def verify(c,digest):
 if a.digest(H/'contract_v2.json')!=digest:raise ValueError('Contract changed after read')
 for p,h in c['sources'].items():
  if a.digest(p)!=h:raise ValueError('Bound source changed '+p)
def protection():
 counts={}
 def check(name,m,root=R):
  for p,h in m.items():
   if a.digest(root/Path(p))!=h:raise ValueError('Protected source changed '+str(p))
  counts[name]=len(m)
 check('protected',read(R/'36_pgsnet_pretrained_contract/contract.json')['protected_files'])
 for stage,manifest in [('53_pcod_formal_evaluation','artifact_manifest.json'),('54_pcod_safe_checkpoint_adapter','artifact_manifest.json'),('55_pcod_checkpoint_evidence','preparation_manifest.json')]:check(stage,read(R/stage/manifest)['files_sha256'],R/stage)
 check('environment',read(R/'31_pgsnet_checkpoint_safety/active_before.json'))
 check('six_manifests',{v['path']:v['sha256'] for parts in read(R/'40_rgbp_formal_preparation/input_contract.json')['manifests'].values() for v in parts.values()})
 return counts

class SavedStorage:
 def __init__(self,state=(),meta_only=False):
  self.state=set(state);self.meta_only=meta_only;self.rows={};self.refs={};self.stack=[];self.calls=0
 def pack(self,t):
  if self.meta_only and t.device.type!='meta':raise ValueError('Non-meta full graph tensor')
  storage=t.untyped_storage();key=storage._cdata;self.calls+=1
  if key not in self.rows:
   self.rows[key]={'bytes':storage.nbytes(),'first_shape':list(t.shape),'dtype':str(t.dtype),'model_state':key in self.state,'first_module':self.stack[-1] if self.stack else '(root)','saves':0}
   self.refs[key]=t.detach()
  self.rows[key]['saves']+=1
  return t
 def unpack(self,t):return t
 def summary(self):
  rows=list(self.rows.values());non=[r for r in rows if not r['model_state']];return {'save_calls':self.calls,'unique_saved_storages':len(rows),'saved_nonstate_storages':len(non),'saved_nonstate_bytes':sum(r['bytes'] for r in non),'saved_state_bytes':sum(r['bytes'] for r in rows if r['model_state']),'top_nonstate_storages':sorted(non,key=lambda r:r['bytes'],reverse=True)[:30]}

def selftest(torch):
 class F(torch.autograd.Function):
  @staticmethod
  def forward(ctx,x):ctx.save_for_backward(x,x.view(-1),x.clone());return x*2
  @staticmethod
  def backward(ctx,g):return g*2
 reports=[]
 for device in ['cpu','meta']:
  x=torch.ones(2,3,device=device,requires_grad=True);meter=SavedStorage(meta_only=device=='meta')
  with torch.autograd.graph.saved_tensors_hooks(meter.pack,meter.unpack):y=F.apply(x)
  s=meter.summary();assert s['save_calls']==3 and s['unique_saved_storages']==2 and s['saved_nonstate_bytes']==48
  y.sum().backward()
  if device=='cpu':assert torch.equal(x.grad,torch.full_like(x,2))
  reports.append({'device':device,'alias_unique_bytes':48,'duplicate_views_counted_once':True,'small_backward_passed':True})
 # Forward hooks must return None; a popped module-name string would replace outputs.
 meter=SavedStorage();net=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU());handles=[]
 for name,m in net.named_modules():
  handles.append(m.register_forward_pre_hook(lambda m,args,n=name:meter.stack.append(n or '(root)')))
  handles.append(m.register_forward_hook(lambda m,args,out:(meter.stack.pop(),None)[1]))
 y=net(torch.ones(2,3));assert isinstance(y,torch.Tensor) and tuple(y.shape)==(2,4) and not meter.stack
 for h in handles:h.remove()
 reports.append({'cpu_forward_hook_output_preserved':True,'balanced_nested_hooks':True})
 return reports

def main():
 raw=(H/'contract_v2.json').read_bytes();digest=hashlib.sha256(raw).hexdigest();c=json.loads(raw);verify(c,digest)
 guard.require_idle();start=resources(10);torch=a.require_runtime();torch.set_num_threads(1)
 with contextlib.ExitStack() as stack:
  for rel in c['locks']:
   f=stack.enter_context((R/rel).open('r+b'));f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  guard.require_idle();before=protection();out=H/'attempt_v2';out.mkdir(exist_ok=False)
  save(out/'identity.json',{'owner':guard.process_identity(os.getpid()),'contract_sha256':digest,'utc':datetime.now(timezone.utc).isoformat(),'resources':start,'gpu_used':False})
  try:
   tests=selftest(torch);save(out/'counter_tests.json',{'passed':True,'checks':tests})
   torch.manual_seed(c['seed']);module=a.pgsnet_module();original=module.resnet18
   def fresh(backbone_path):
    if backbone_path is not None:raise ValueError('No external weights')
    return original(backbone_path=None,pretrained=False)
   module.resnet18=fresh
   try:model=module.PGSNet(None,None,None,None).cpu().train()
   finally:module.resnet18=original
   parameters=sum(p.numel() for p in model.parameters());assert parameters==309187249;resources(8)
   param_bytes=sum(p.numel()*p.element_size() for p in model.parameters());model.to('meta');gc.collect();resources(8)
   state={t.untyped_storage()._cdata:t.untyped_storage().nbytes() for t in model.state_dict().values()};meter=SavedStorage(state,meta_only=True)
   handles=[]
   for name,m in model.named_modules():
    handles.append(m.register_forward_pre_hook(lambda m,args,n=name:meter.stack.append(n or '(root)')))
    handles.append(m.register_forward_hook(lambda m,args,out:(meter.stack.pop(),None)[1]))
   inputs=[torch.empty(c['batch'],3,c['size'],c['size'],device='meta',dtype=torch.float32) for _ in range(3)]
   print(json.dumps({'state':'starting_metadata_forward','batch':c['batch'],'size':c['size'],'model_parameters':parameters,'actual_activation_values_allocated':False}),flush=True)
   with torch.autograd.graph.saved_tensors_hooks(meter.pack,meter.unpack):outputs=model(*inputs)
   assert len(outputs)==11 and all(tuple(t.shape)==(18,1,416,416) and t.device.type=='meta' and t.requires_grad for t in outputs)
   assert not meter.stack
   s=meter.summary();outs={t.untyped_storage()._cdata:t.untyped_storage().nbytes() for t in outputs};extra=sum(n for k,n in outs.items() if k not in state and k not in meter.rows);state_bytes=sum(state.values())
   s.update({'model_parameters':parameters,'model_state_tensors':len(model.state_dict()),'model_state_bytes':state_bytes,'fp32_parameter_bytes':param_bytes,'output_shapes':[list(t.shape) for t in outputs],'output_additional_bytes':extra,'forward_retained_logical_bytes':state_bytes+s['saved_nonstate_bytes']+extra,'all_parameter_gradient_plus_momentum_upper_allowance_bytes':2*param_bytes})
   s['forward_retained_logical_gib']=s['forward_retained_logical_bytes']/2**30
   s['forward_plus_full_gradient_momentum_allowance_gib']=(s['forward_retained_logical_bytes']+2*param_bytes)/2**30
   save(out/'saved_storage_rows.json',list(meter.rows.values()))
   for h in handles:h.remove()
   del model,outputs,inputs,meter;gc.collect();finish=resources(8);verify(c,digest);after=protection();guard.require_idle()
   report={'passed_metadata_profile':True,'contract_sha256':digest,'counter_tests_sha256':a.digest(out/'counter_tests.json'),'storage_rows_sha256':a.digest(out/'saved_storage_rows.json'),'original_batch18_size416':True,'value_forward_executed':False,'full_model_backward_executed':False,'optimizer_updates':0,'gpu_used':False,'precision_changed':False,'gradient_accumulation':False,'activation_checkpointing':False,'external_weights_loaded':False,'real_images_read':False,'profile':s,'before_protection':before,'after_protection':after,'finish_resources':finish,'interpretation':'Shape-only eager-autograd saved-storage accounting, not measured CPU/GPU peak, not device execution or numerical equivalence. Excludes loss graph, transient tensors, backward workspace, allocator and driver overhead. Gradients/momentum are an all-parameter allowance, not observed participation.','utc':datetime.now(timezone.utc).isoformat()}
   save(out/'report.json',report);print(json.dumps({'passed_metadata_profile':True,'profile':s,'finish_resources':finish}),flush=True)
  except BaseException as error:
   save(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),'automatic_retry':False,'gpu_used':False});raise
if __name__=='__main__':main()

