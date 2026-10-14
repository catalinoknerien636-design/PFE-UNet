"""Identity-gated qualified or formal fixed-shape inference; no training or numerical comparison."""
from pathlib import Path
import os,sys,time,traceback,contextlib
from job87 import w
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
 if plan['phase'] not in ('qualification','matrix'):raise RuntimeError('invalid_phase')
 runs={r['entry']['config']['run_id']:r for r in c['runs']}
 if plan['run_id'] not in runs:raise RuntimeError('unregistered_run')
 if plan['phase']=='qualification' and plan['run_id'] not in c['qualification_order']:raise RuntimeError('wrong_qualification_population')
 if plan['phase']=='matrix':
  gate=w.read(H/'qualification_v1/complete.json')
  if not gate['passed'] or gate['qualified_run_ids']!=c['qualification_order'] or w.digest(H/'qualification_v1/complete.json')!=plan['qualification_sha256']:raise RuntimeError('qualification_required')
 row=runs[plan['run_id']];cfg=row['entry']['config']
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 sys.path.insert(0,str(R/'57_rgbp_safe_runtime'));import safe_reader57
 import hashlib,gc
 best_path=Path(cfg['output_dir'])/'best.pt'
 if w.digest(best_path)!=row['best_sha256']:raise RuntimeError('checkpoint_before_load_changed')
 reader=safe_reader57.Reader(entries=[row['entry']]);best=reader(best_path,map_location='cpu')
 if set(best)!={'identity','epoch','model_state','validation_iou'} or best['identity']!=row['entry']['identity'] or best['epoch']!=row['selected_epoch'] or best['validation_iou']!=row['selected_validation_iou']:raise RuntimeError('selected_state_identity_changed')
 if w.digest(best_path)!=row['best_sha256']:raise RuntimeError('checkpoint_after_load_changed')
 evidence=w.read(row['tensor_evidence'])['best_tensor_values']
 def verify_state(state):
  if set(state)!=set(evidence):raise RuntimeError('tensor_key_coverage')
  for key,tensor in state.items():
   array=tensor.detach().cpu().numpy();expected=evidence[key]
   if list(array.shape)!=expected['shape'] or array.dtype.str!=expected['dtype'] or hashlib.sha256(array.tobytes(order='C')).hexdigest()!=expected['sha256']:raise RuntimeError('tensor_source_identity:'+key)
 verify_state(best['model_state'])
 sys.path.insert(0,str(R/'15_pcod_training'));import pcod_models as models
 sys.path.insert(0,str(R/'20_pcod_backend_diagnostics'));import resize_candidate as resize;import resize_factor2 as factor2
 resize.bilinear_half_pixel=factor2.bilinear_half_pixel
 model=resize.install_unet_candidate(models.build_model(cfg['method'],base_channels=cfg['base_channels'],seed=cfg['seed'])).eval()
 model.load_state_dict(best['model_state'],strict=True);verify_state(model.state_dict())
 parameter_count=sum(p.numel() for p in model.parameters())
 if parameter_count!=c['parameter_counts'][cfg['method']]:raise RuntimeError('architecture_mismatch')
 w.write(out/'model_cpu_ready.json',{'identity':me,'run_id':cfg['run_id'],'method':cfg['method'],'seed':cfg['seed'],'best_sha256':row['best_sha256'],'selected_epoch':row['selected_epoch'],'tensor_evidence_sha256':w.digest(row['tensor_evidence']),'CPU_loaded_state_bytes_verified':True,'state_tensors':len(evidence),'parameters':parameter_count,'reader_calls':reader.calls,'training':model.training,'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'monotonic_ns':time.perf_counter_ns()})
 del best;gc.collect()
 inventory=[{'index':i,'name':dml.device_name(i).rstrip(chr(0))} for i in range(dml.device_count())]
 if [r['index'] for r in inventory if r['name']==c['expected_device_name']]!=[1]:raise RuntimeError('device_inventory_mismatch')
 device=dml.device(1)
 if str(device)!='privateuseone:1':raise RuntimeError('wrong_device')
 w.write(out/'device_selected.json',{'identity':me,'device':str(device),'inventory':inventory,'monotonic_ns':time.perf_counter_ns()})
 source=(torch.arange(4*352*352,dtype=torch.float32).remainder(257)/256).reshape(1,4,352,352)
 input_sha=hashlib.sha256(source.numpy().tobytes(order='C')).hexdigest()
 if input_sha!=c['input_sha256']:raise RuntimeError('synthetic_input_changed')
 model.to(device);probe=source.to(device)
 if not all(p.device==device and p.dtype==torch.float32 for p in model.parameters()) or not all(not m.training for m in model.modules()):raise RuntimeError('model_device_eval_dtype')
 w.write(out/'model_device_ready.json',{'identity':me,'input_shape':list(probe.shape),'input_bytes':source.numel()*source.element_size(),'input_sha256':input_sha,'device':str(device),'eval_mode':True,'monotonic_ns':time.perf_counter_ns()})
 from timing87 import timed_forward
 receipts=[]
 with torch.no_grad():
  for index in range(35):
   cpu,interval=timed_forward(model,probe)
   if list(cpu.shape)!=[1,1,352,352] or cpu.dtype!=torch.float32 or not bool(torch.isfinite(cpu).all()) or cpu.requires_grad:raise RuntimeError('output_validation')
   receipt={'identity':me,'iteration':index,'kind':'warmup' if index<5 else 'measured','phase':plan['phase'],'interval':interval,'output_shape':list(cpu.shape),'output_bytes':cpu.numel()*cpu.element_size(),'finite':True,'readback_completed':True,'requires_grad':cpu.requires_grad,'gradient_enabled':torch.is_grad_enabled()}
   del cpu
   w.write(out/('iteration_'+str(index).zfill(2)+'.json'),receipt);receipts.append(receipt)
   if index==4:w.write(out/'warmup_done.json',{'identity':me,'count':5,'monotonic_ns':time.perf_counter_ns()})
 w.write(out/'measurements_done.json',{'identity':me,'count':30,'monotonic_ns':time.perf_counter_ns()})
 if any(p.grad is not None for p in model.parameters()):raise RuntimeError('unexpected_parameter_grad')
 w.write(out/'done.json',{'identity':me,'run_id':cfg['run_id'],'phase':plan['phase'],'method':cfg['method'],'seed':cfg['seed'],'warmups':5,'measurements':30,'forwards':35,'training_steps':0,'test_images_read':False,'numerical_comparison':False,'input_sha256':input_sha,'monotonic_ns':time.perf_counter_ns()})

if __name__=='__main__':
 out=Path(sys.argv[1])
 try:
  with (out/'stdout.txt').open('x',encoding='utf-8',buffering=1) as stdout,(out/'stderr.txt').open('x',encoding='utf-8',buffering=1) as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):main(out,sys.argv[2])
 except BaseException as exc:
  w.write(out/'worker_error.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()});sys.exit(71)
