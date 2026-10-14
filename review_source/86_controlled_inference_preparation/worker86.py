"""Identity-gated lifecycle probe: stdlib CPU or one tiny DirectML transfer; no models."""
from pathlib import Path
import os,sys,time,traceback,contextlib
from job86 import w
H=Path(__file__).resolve().parent;R=H.parent

def main(out,nonce):
 plan=w.read(out/'plan.json');c=w.read(H/'contract.json');me=w.ident(w.K.GetCurrentProcess())
 claim={**me,'nonce':nonce,'sys_executable':os.path.normcase(os.path.abspath(sys.executable)),'worker_sha256':w.digest(__file__),'api_sha256':w.digest(R/'84_worker_supervisor_qualification/win84_v4.py'),'directory':str(out.resolve()),'torch_imported':'torch' in sys.modules,'directml_imported':'torch_directml' in sys.modules,'no_site':bool(sys.flags.no_site)}
 if plan['name']=='cpu_bad_nonce':claim['nonce']='deliberate_invalid_nonce'
 w.write(out/'identity.json',claim);deadline=time.monotonic()+15
 while not (out/'approval.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('approval_timeout')
  time.sleep(.01)
 a=w.read(out/'approval.json')
 required={**me,'nonce':nonce,'worker_sha256':w.digest(__file__),'plan_sha256':w.digest(out/'plan.json')}
 if any(a.get(k)!=v for k,v in required.items()):raise RuntimeError('invalid_approval')
 handle=w.open_process(a['supervisor_identity']['pid'])
 try:
  if w.ident(handle)!=a['supervisor_identity'] or not w.running(handle):raise RuntimeError('supervisor_not_live')
 finally:w.close(handle)
 if 'torch' in sys.modules or 'torch_directml' in sys.modules:raise RuntimeError('torch_before_approval')
 w.reserve();w.write(out/'payload_started.json',{'identity':me,'approval_sha256':w.digest(out/'approval.json'),'torch_imported':False,'monotonic_ns':time.perf_counter_ns()})
 # site and all numerical/runtime imports occur only after verified identity approval.
 import site
 site.main()
 if Path(sys.prefix).resolve()!=Path(c['venv_prefix']).resolve():raise RuntimeError('wrong_venv_after_site')
 w.write(out/'site_ready.json',{'identity':me,'prefix':sys.prefix,'site_file':site.__file__,'monotonic_ns':time.perf_counter_ns()})
 import torch
 import torch_directml as dml
 import importlib.metadata
 if str(torch.__version__)!='2.4.1+cpu' or importlib.metadata.version('torch-directml')!='0.2.5.dev240914':raise RuntimeError('runtime_version_mismatch')
 w.write(out/'torch_imported.json',{'identity':me,'torch':str(torch.__version__),'torch_directml':importlib.metadata.version('torch-directml'),'torch_file':torch.__file__,'directml_file':dml.__file__,'monotonic_ns':time.perf_counter_ns()})
 inventory=[{'index':i,'name':dml.device_name(i).rstrip(chr(0))} for i in range(dml.device_count())]
 if [r['index'] for r in inventory if r['name']==c['expected_device_name']]!=[c['device_index']]:raise RuntimeError('device_inventory_mismatch')
 device=dml.device(c['device_index'])
 if device.type!='privateuseone' or device.index!=1:raise RuntimeError('wrong_device')
 w.write(out/'device_selected.json',{'identity':me,'device':str(device),'inventory':inventory,'monotonic_ns':time.perf_counter_ns()})
 source=torch.arange(256,dtype=torch.float32).reshape(16,16)
 value=source.to(device);result=value+1;readback=result.cpu()
 if list(readback.shape)!=[16,16] or not bool(torch.isfinite(readback).all()):raise RuntimeError('probe_readback_invalid')
 w.write(out/'device_ready.json',{'identity':me,'device':str(device),'shape':list(readback.shape),'dtype':str(readback.dtype),'readback_bytes':readback.numel()*readback.element_size(),'finite':True,'model_loaded':False,'weights_or_images_read':False,'numerical_equivalence_test':False,'monotonic_ns':time.perf_counter_ns()})
 if plan['name']!='single_model_pilot':raise RuntimeError('pilot_scope_only')
 if w.digest(H/'meta_v1/complete.json')!=plan['meta_complete_sha256']:raise RuntimeError('meta_admission_changed')
 meta=w.read(H/'meta_v1/complete.json')
 if not meta['passed'] or meta['max_logical_gib']>c['meta_resource_gate_total_logical_gib']:raise RuntimeError('meta_admission_failed')
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 sys.path.insert(0,str(R/'54_pcod_safe_checkpoint_adapter'));import tensor54
 row=c['pilot_run'];cfg=row['entry']['config']
 if cfg['method']!='direct_reference' or cfg['seed']!=42:raise RuntimeError('pilot_selection_changed')
 best=tensor54.provider(cfg['run_id'],row['entry']['identity'])
 if best['epoch']!=row['selected_epoch'] or best['validation_iou']!=row['selected_validation_iou']:raise RuntimeError('selected_checkpoint_changed')
 sys.path.insert(0,str(R/'15_pcod_training'));import pcod_models as models
 sys.path.insert(0,str(R/'20_pcod_backend_diagnostics'));import resize_candidate as resize;import resize_factor2 as factor2
 resize.bilinear_half_pixel=factor2.bilinear_half_pixel
 model=resize.install_unet_candidate(models.build_model(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed'])).eval()
 model.load_state_dict(best['model_state'],strict=True)
 import hashlib,gc
 state_dir=R/'54_pcod_safe_checkpoint_adapter/conversions'/cfg['run_id']/'state'
 state_manifest=w.read(state_dir/'manifest.json');cpu_state=model.state_dict()
 for schema,entry in zip(state_manifest['schema'],state_manifest['files']):
  if hashlib.sha256(cpu_state[schema['key']].detach().numpy().tobytes(order='C')).hexdigest()!=entry['value_bytes_sha256']:raise RuntimeError('loaded_CPU_state_identity_mismatch')
 if sum(p.numel() for p in model.parameters())!=7849889:raise RuntimeError('architecture_changed')
 w.write(out/'model_cpu_ready.json',{'identity':me,'run_id':cfg['run_id'],'best_sha256':row['best_sha256'],'selected_epoch':row['selected_epoch'],'state_manifest_sha256':w.digest(state_dir/'manifest.json'),'CPU_loaded_state_bytes_verified':True,'state_tensors':len(cpu_state),'parameters':7849889,'training':model.training,'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'monotonic_ns':time.perf_counter_ns()})
 del best,cpu_state;gc.collect()
 source=(torch.arange(4*352*352,dtype=torch.float32).remainder(257)/256).reshape(1,4,352,352)
 input_sha=hashlib.sha256(source.numpy().tobytes(order='C')).hexdigest()
 model.to(device);probe=source.to(device)
 if not all(p.device==device for p in model.parameters()) or not all(not m.training for m in model.modules()):raise RuntimeError('model_device_or_eval_mismatch')
 w.write(out/'model_device_ready.json',{'identity':me,'input_shape':list(probe.shape),'input_bytes':source.numel()*source.element_size(),'input_cpu_sha256':input_sha,'device':str(device),'model_eval':True,'monotonic_ns':time.perf_counter_ns()})
 receipts=[]
 with torch.no_grad():
  for index in range(1,c['pilot_forwards']+1):
   logits=model(probe);probability=logits.sigmoid();back=probability.cpu()
   if list(back.shape)!=[1,1,352,352] or back.dtype!=torch.float32 or not bool(torch.isfinite(back).all()):raise RuntimeError('pilot_output_invalid')
   if logits.requires_grad or probability.requires_grad:raise RuntimeError('unexpected_grad_graph')
   receipt={'identity':me,'iteration':index,'output_shape':list(back.shape),'output_bytes':back.numel()*back.element_size(),'finite':True,'readback_completed':True,'gradient_enabled':torch.is_grad_enabled(),'model_training':model.training,'monotonic_ns':time.perf_counter_ns()}
   w.write(out/('pilot_forward_'+str(index)+'.json'),receipt);receipts.append(receipt)
   del logits,probability,back
   time.sleep(.15)
 if any(p.grad is not None for p in model.parameters()):raise RuntimeError('unexpected_parameter_grad')
 w.write(out/'done.json',{'identity':me,'GPU_used':True,'run_id':cfg['run_id'],'model_forwards':len(receipts),'training_steps':0,'real_images_read':False,'inference_benchmark':False,'numerical_comparison':False,'input_cpu_sha256':input_sha,'monotonic_ns':time.perf_counter_ns()})

if __name__=='__main__':
 out=Path(sys.argv[1])
 try:
  with (out/'stdout.txt').open('x',encoding='utf-8',buffering=1) as stdout,(out/'stderr.txt').open('x',encoding='utf-8',buffering=1) as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):main(out,sys.argv[2])
 except BaseException as exc:
  w.write(out/'worker_error.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()});sys.exit(71)
