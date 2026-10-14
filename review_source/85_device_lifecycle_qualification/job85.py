"""Separate, bounded device-lifecycle Job; 84 CPU sources stay unchanged."""
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'84_worker_supervisor_qualification'))
import win84_v4 as w

class LifecycleJob(w.Job):
 def __init__(self,c):
  if c['job_commit_cap_gib']!=9 or c['job_active_process_limit']!=8:raise ValueError('frozen_lifecycle_budget')
  self.handle=w.ok(w.K.CreateJobObjectW(None,None));self.launch=None;self.launch_id=None;self.query_history=[]
  try:
   limit=w.Limits();limit.basic.flags=0x2000|0x200|0x8;limit.basic.active=8;limit.job_memory=9*2**30
   w.ok(w.K.SetInformationJobObject(self.handle,9,w.C.byref(limit),w.C.sizeof(limit)))
   check=w.Limits();w.ok(w.K.QueryInformationJobObject(self.handle,9,w.C.byref(check),w.C.sizeof(check),None))
   if (check.basic.flags,check.basic.active,check.job_memory)!=(8712,8,9*2**30):raise RuntimeError('job_limits_readback')
   self.limits={'flags':check.basic.flags,'active_process_limit':check.basic.active,'job_memory_bytes':check.job_memory}
  except BaseException:w.close(self.handle);self.handle=None;raise

def reason(row,c,private_limit_bytes,timeout):
 if row['host']['available_ram_gib']<c['abort_available_ram_gib']:return 'physical_reserve'
 if row['host']['available_commit_gib']<c['abort_available_commit_gib']:return 'system_commit_reserve'
 if row['worker_memory']['private_bytes']>private_limit_bytes:return 'actual_worker_private_limit'
 if row['worker_memory']['working_set_bytes']>c['working_set_cap_gib']*2**30:return 'actual_worker_working_set_limit'
 if row['elapsed']>timeout:return 'worker_timeout'
 return None
