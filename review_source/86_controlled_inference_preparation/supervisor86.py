"""Separate device-lifecycle supervisor. No scientific model execution."""
import os,time,secrets,sys,threading
from pathlib import Path
from job86 import w,LifecycleJob,reason
H=Path(__file__).resolve().parent
norm=lambda p:os.path.normcase(os.path.abspath(p))

class Supervisor:
 def __init__(self,out,runtime,expected_image,plan):
  self.out=Path(out);self.runtime=str(runtime);self.expected_image=norm(expected_image);self.plan=dict(plan)
  self.job=None;self.worker=None;self.claim=None;self.bound=None;self.samples=[];self.approved=False;self.cleanup=None
  self.nonce=secrets.token_hex(24);self.self_id=w.ident(w.K.GetCurrentProcess());self.contract=w.read(H/'contract.json');self.guard_stop=threading.Event();self.guard_ready=threading.Event();self.guard_thread=None;self.guard_reason=None
  self.plan['supervisor_identity']=self.self_id
 def verify_sources(self):
  for path,sha in self.contract['sources'].items():
   if w.digest(path)!=sha:raise ValueError('source_changed:'+path)
 def start(self):
  self.verify_sources();w.reserve();self.out.mkdir(exist_ok=False);w.write(self.out/'plan.json',self.plan)
  self.job=LifecycleJob(self.contract)
  self.launch_id=self.job.start([self.runtime,'-E','-s','-S','-B','-X','utf8',str(H/'worker86.py'),str(self.out),self.nonce],H)
  w.write(self.out/'launch.json',{'launcher':self.launch_id,'launcher_image':w.image_path(self.job.launch),'suspended_assigned_before_resume':True,'job_committed_memory_limit_gib':9,'job_active_process_limit':8,'kill_on_job_close':True,'breakaway_enabled':False,'supervisor':self.self_id})
 def bind(self,timeout=8):
  start=time.monotonic()
  while not (self.out/'identity.json').exists():
   w.reserve()
   if not w.running(self.job.launch):raise RuntimeError('worker_exited_before_identity')
   if time.monotonic()-start>timeout:raise RuntimeError('handshake_timeout')
   time.sleep(.02)
  claim=w.read(self.out/'identity.json');self.claim=claim
  if not isinstance(claim,dict):raise ValueError('identity_schema')
  if type(claim.get('pid')) is not int or type(claim.get('creation_ticks')) is not int:raise ValueError('identity_schema')
  if claim.get('nonce')!=self.nonce:raise ValueError('identity_nonce')
  if claim.get('torch_imported') or claim.get('directml_imported') or not claim.get('no_site'):raise ValueError('heavy_import_before_identity')
  expected={'worker_sha256':self.contract['sources'][str(H/'worker86.py')],'api_sha256':self.contract['sources'][str(H.parent/'84_worker_supervisor_qualification/win84_v4.py')],'directory':str(self.out.resolve()),'sys_executable':norm(self.runtime)}
  if any(claim.get(k)!=v for k,v in expected.items()):raise ValueError('identity_source_or_path')
  self.worker=w.open_process(claim['pid']);actual=w.ident(self.worker)
  if actual!={k:claim[k] for k in ('pid','creation_ticks')}:raise ValueError('identity_creation')
  if not self.job.contains(self.worker):raise ValueError('identity_not_in_owned_job')
  if not w.running(self.worker):raise ValueError('identity_dead')
  if w.image_path(self.worker)!=self.expected_image:raise ValueError('identity_image')
  # Keep the real handle. Validate the complete live parent chain back to our owned launcher.
  rows=w.snapshot();current=actual['pid'];chain=[];held=[];child_ticks=None
  try:
   for _ in range(8):
    handle=w.open_process(current);held.append(handle);entry=w.ident(handle)
    if not self.job.contains(handle) or not w.running(handle):raise ValueError('identity_chain_job_or_liveness')
    if child_ticks is not None and entry['creation_ticks']>child_ticks:raise ValueError('identity_parent_newer_than_child')
    chain.append({**entry,'image':w.image_path(handle)});child_ticks=entry['creation_ticks']
    if current==self.launch_id['pid']:
     if entry!=self.launch_id:raise ValueError('identity_launcher_reused')
     break
    if current not in rows:raise ValueError('identity_parent_missing')
    current=rows[current]['ppid']
   else:raise ValueError('identity_chain_too_long')
   if chain[-1]['pid']!=self.launch_id['pid']:raise ValueError('identity_not_descendant')
  finally:
   for handle in held:w.close(handle)
  time.sleep(.10)
  if (self.out/'payload_started.json').exists():raise ValueError('payload_before_approval')
  self.bound={'worker':actual,'worker_image':w.image_path(self.worker),'launcher':self.launch_id,'parent_chain':chain,'worker_baseline_memory':w.memory(self.worker),'launcher_baseline_memory':w.memory(self.job.launch),'no_payload_before_approval':True,'retained_actual_worker_handle':True,'source_bindings':expected}
  w.write(self.out/'binding.json',self.bound);return self.bound
 def approve(self,bad_approval=False):
  self.verify_sources()
  if not self.bound:raise RuntimeError('not_bound')
  if not w.running(self.worker) or w.ident(self.worker)!=self.bound['worker']:raise RuntimeError('identity_lost_before_approval')
  token={**self.bound['worker'],'nonce':self.nonce,'worker_sha256':w.digest(H/'worker86.py'),'supervisor_identity':self.self_id,'plan_sha256':w.digest(self.out/'plan.json')}
  if bad_approval:token['creation_ticks']+=1
  w.write(self.out/'approval.json',token);self.approved=True
 def start_guard(self):
  if not self.bound:raise RuntimeError('not_bound')
  self.guard_started=time.monotonic()
  cap=self.contract['private_process_cap_gib']*2**30
  if self.plan['name']=='cpu_private_abort':cap=self.bound['worker_baseline_memory']['private_bytes']+16*2**20
  def watch():
   try:
    while not self.guard_stop.is_set():
     if not w.running(self.worker):break
     row={'worker':self.bound['worker'],'worker_memory':w.memory(self.worker),'launcher_memory':w.memory(self.job.launch),'host':w.host(),'elapsed':time.monotonic()-self.guard_started}
     phase='before_payload'
     for label in ['payload_started','site_ready','torch_imported','device_selected','device_ready','model_cpu_ready','model_device_ready','pilot_forward_1','pilot_forward_2','pilot_forward_3','done']:
      if (self.out/(label+'.json')).exists():phase=label
     row['phase']=phase;self.samples.append(row);self.guard_ready.set()
     problem=reason(row,self.contract,cap,self.plan['timeout_seconds'])
     if problem:self.guard_reason=problem;break
     self.guard_stop.wait(.05)
   except BaseException as exc:
    if w.running(self.worker):self.guard_reason='sampling_error:'+repr(exc)
    self.guard_ready.set()
  self.guard_thread=threading.Thread(target=watch,name='actual-worker-sampler',daemon=True);self.guard_thread.start()
  if not self.guard_ready.wait(3) or self.guard_reason:raise RuntimeError('guard_not_ready:'+str(self.guard_reason))
 def monitor(self):
  ready_at=None;release_written=False
  while w.running(self.worker):
   if self.guard_reason:return self.guard_reason
   if (self.out/'device_ready.json').exists():
    if ready_at is None:ready_at=time.monotonic()
    age=time.monotonic()-ready_at
    if self.plan['name']=='device_controlled_stop' and age>=.5:return 'controlled_stop_after_device_ready'
    if self.plan['name']=='device_normal' and age>=.25 and not release_written:
     w.write(self.out/'release.json',{'nonce':self.nonce,**self.bound['worker']});release_written=True
   time.sleep(.02)
  if self.guard_reason:return self.guard_reason
  code=w.wait_exit(self.worker);return 'normal' if code==0 else 'worker_exit_'+str(code)
 def finish(self):
  errors=[]
  self.guard_stop.set()
  if self.guard_thread:
   self.guard_thread.join(5)
   if self.guard_thread.is_alive():errors.append('sampler_did_not_stop')
  try:
   if self.job:self.cleanup=self.job.stop()
  except BaseException as exc:errors.append(str(exc))
  finally:
   if self.worker:w.close(self.worker);self.worker=None
   if self.job:self.job.close()
  if self.out.exists():
   w.write(self.out/'samples.json',self.samples)
   w.write(self.out/'cleanup.json',{'result':self.cleanup,'errors':errors})
  if errors:raise RuntimeError('cleanup_failed:'+repr(errors))

