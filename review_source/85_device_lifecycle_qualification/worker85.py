"""Identity-gated lifecycle probe: stdlib CPU or one tiny DirectML transfer; no models."""
from pathlib import Path
import os,sys,time,traceback,contextlib
from job85 import w
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
 if plan['name'].startswith('cpu_'):
  size={'cpu_normal':16,'cpu_private_abort':32,'cpu_timeout':0}[plan['name']];data=bytearray(size*2**20)
  for i in range(0,len(data),4096):data[i]=17
  w.write(out/'allocated.json',{'bytes':len(data),'identity':me})
  if plan['name']=='cpu_normal':time.sleep(.25);w.write(out/'done.json',{'identity':me,'torch_imported':False,'GPU_used':False});return
  time.sleep(20);raise RuntimeError('cpu_abort_not_delivered')
 if plan['name'] not in ('device_normal','device_controlled_stop'):raise ValueError('unregistered_case')
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
 deadline=time.monotonic()+15
 while True:
  if plan['name']=='device_normal' and (out/'release.json').exists():
   release=w.read(out/'release.json')
   if release!={**me,'nonce':nonce}:raise RuntimeError('bad_release')
   w.write(out/'done.json',{'identity':me,'GPU_used':True,'model_loaded':False,'torch_imported':True,'monotonic_ns':time.perf_counter_ns()});return
  if time.monotonic()>deadline:raise RuntimeError('device_probe_end_timeout')
  time.sleep(.02)

if __name__=='__main__':
 out=Path(sys.argv[1])
 try:
  with (out/'stdout.txt').open('x',encoding='utf-8',buffering=1) as stdout,(out/'stderr.txt').open('x',encoding='utf-8',buffering=1) as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):main(out,sys.argv[2])
 except BaseException as exc:
  w.write(out/'worker_error.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()});sys.exit(71)
