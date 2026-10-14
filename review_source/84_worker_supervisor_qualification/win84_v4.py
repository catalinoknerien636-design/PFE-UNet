"""Windows CPU-only process qualification primitives. No model imports."""
import ctypes as C
from ctypes import wintypes as T
import hashlib,json,os,subprocess,time
from pathlib import Path

K=C.WinDLL('kernel32',use_last_error=True);P=C.WinDLL('psapi',use_last_error=True)
class Startup(C.Structure):
 _fields_=[('cb',T.DWORD),('reserved',T.LPWSTR),('desktop',T.LPWSTR),('title',T.LPWSTR)]+[(n,T.DWORD) for n in ['x','y','xsize','ysize','xchars','ychars','fill','flags']]+[('show',T.WORD),('reserved2size',T.WORD),('reserved2',C.c_void_p),('stdin',T.HANDLE),('stdout',T.HANDLE),('stderr',T.HANDLE)]
class ProcessInfo(C.Structure):
 _fields_=[('process',T.HANDLE),('thread',T.HANDLE),('pid',T.DWORD),('tid',T.DWORD)]
class BasicLimits(C.Structure):
 _fields_=[('process_time',C.c_int64),('job_time',C.c_int64),('flags',T.DWORD),('min_ws',C.c_size_t),('max_ws',C.c_size_t),('active',T.DWORD),('affinity',C.c_size_t),('priority',T.DWORD),('scheduling',T.DWORD)]
class IO(C.Structure):
 _fields_=[(n,C.c_uint64) for n in ['read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes']]
class Limits(C.Structure):
 _fields_=[('basic',BasicLimits),('io',IO),('process_memory',C.c_size_t),('job_memory',C.c_size_t),('peak_process',C.c_size_t),('peak_job',C.c_size_t)]
class ProcessEntry(C.Structure):
 _fields_=[('size',T.DWORD),('usage',T.DWORD),('pid',T.DWORD),('heap',C.c_size_t),('module',T.DWORD),('threads',T.DWORD),('ppid',T.DWORD),('priority',T.LONG),('flags',T.DWORD),('exe',T.WCHAR*260)]
class Memory(C.Structure):
 _fields_=[('size',T.DWORD),('faults',T.DWORD)]+[(n,C.c_size_t) for n in ['peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private']]
class HostMemory(C.Structure):
 _fields_=[('length',T.DWORD),('load',T.DWORD)]+[(n,C.c_uint64) for n in ['physical','available_physical','pagefile','available_pagefile','virtual','available_virtual','extended']]

def setup(name,args,result=T.BOOL,dll=K):
 f=getattr(dll,name);f.argtypes=args;f.restype=result;return f
setup('CloseHandle',[T.HANDLE]);setup('GetCurrentProcess',[],T.HANDLE)
setup('OpenProcess',[T.DWORD,T.BOOL,T.DWORD],T.HANDLE)
setup('GetProcessTimes',[T.HANDLE]+[C.POINTER(T.FILETIME)]*4)
setup('GetProcessId',[T.HANDLE],T.DWORD)
setup('QueryFullProcessImageNameW',[T.HANDLE,T.DWORD,T.LPWSTR,C.POINTER(T.DWORD)])
setup('WaitForSingleObject',[T.HANDLE,T.DWORD],T.DWORD);setup('GetExitCodeProcess',[T.HANDLE,C.POINTER(T.DWORD)])
setup('CreateJobObjectW',[C.c_void_p,T.LPCWSTR],T.HANDLE)
setup('SetInformationJobObject',[T.HANDLE,C.c_int,C.c_void_p,T.DWORD])
setup('QueryInformationJobObject',[T.HANDLE,C.c_int,C.c_void_p,T.DWORD,C.POINTER(T.DWORD)])
setup('AssignProcessToJobObject',[T.HANDLE,T.HANDLE]);setup('IsProcessInJob',[T.HANDLE,T.HANDLE,C.POINTER(T.BOOL)])
setup('TerminateJobObject',[T.HANDLE,T.UINT]);setup('TerminateProcess',[T.HANDLE,T.UINT])
setup('CreateProcessW',[T.LPCWSTR,T.LPWSTR,C.c_void_p,C.c_void_p,T.BOOL,T.DWORD,C.c_void_p,T.LPCWSTR,C.POINTER(Startup),C.POINTER(ProcessInfo)])
setup('ResumeThread',[T.HANDLE],T.DWORD)
setup('CreateToolhelp32Snapshot',[T.DWORD,T.DWORD],T.HANDLE)
setup('Process32FirstW',[T.HANDLE,C.POINTER(ProcessEntry)]);setup('Process32NextW',[T.HANDLE,C.POINTER(ProcessEntry)])
setup('GlobalMemoryStatusEx',[C.POINTER(HostMemory)])
setup('GetProcessMemoryInfo',[T.HANDLE,C.POINTER(Memory),T.DWORD],dll=P)
assert C.sizeof(C.c_void_p)==8 and C.sizeof(Limits)==144 and C.sizeof(ProcessEntry)==568 and C.sizeof(Startup)==104

def ok(value):
 if not value:raise C.WinError(C.get_last_error())
 return value

def close(handle):
 if handle:ok(K.CloseHandle(handle))

def open_process(pid):
 # No terminate right: destructive action is scoped to our newly created Job.
 if type(pid) is not int or pid<=0:raise ValueError('invalid_pid')
 return ok(K.OpenProcess(0x100000|0x1000|0x400|0x10,False,pid))

def ident(handle):
 ts=[T.FILETIME() for _ in range(4)];ok(K.GetProcessTimes(handle,*[C.byref(t) for t in ts]))
 pid=ok(K.GetProcessId(handle));return {'pid':pid,'creation_ticks':(ts[0].dwHighDateTime<<32)|ts[0].dwLowDateTime}

def image_path(handle):
 size=T.DWORD(32768);buf=C.create_unicode_buffer(size.value);ok(K.QueryFullProcessImageNameW(handle,0,buf,C.byref(size)));return os.path.normcase(os.path.abspath(buf.value))

def running(handle):
 value=K.WaitForSingleObject(handle,0)
 if value not in (0,258):raise C.WinError(C.get_last_error())
 return value==258

def wait_exit(handle,ms=5000):
 result=K.WaitForSingleObject(handle,ms)
 if result!=0:raise RuntimeError('exact_handle_did_not_exit:'+str(result))
 value=T.DWORD();ok(K.GetExitCodeProcess(handle,C.byref(value)));return value.value

def memory(handle):
 m=Memory();m.size=C.sizeof(m);ok(P.GetProcessMemoryInfo(handle,C.byref(m),m.size))
 return {'working_set_bytes':m.ws,'private_bytes':m.private,'os_peak_working_set_bytes':m.peak_ws,'os_peak_pagefile_bytes':m.peak_pagefile}

def host():
 m=HostMemory();m.length=C.sizeof(m);ok(K.GlobalMemoryStatusEx(C.byref(m)))
 return {'available_ram_gib':m.available_physical/2**30,'available_commit_gib':m.available_pagefile/2**30}

def reserve():
 r=host()
 if r['available_ram_gib']<8.5 or r['available_commit_gib']<8.5:raise RuntimeError('host_reserve')
 return r

def snapshot():
 handle=K.CreateToolhelp32Snapshot(2,0)
 if handle==C.c_void_p(-1).value:raise C.WinError(C.get_last_error())
 try:
  p=ProcessEntry();p.size=C.sizeof(p);ok(K.Process32FirstW(handle,C.byref(p)));rows={}
  while True:
   rows[p.pid]={'pid':p.pid,'ppid':p.ppid,'exe':p.exe}
   if not K.Process32NextW(handle,C.byref(p)):
    if C.get_last_error()!=18:raise C.WinError(C.get_last_error())
    return rows
 finally:close(handle)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,data):
 path=Path(path);temp=path.with_suffix(path.suffix+'.part')
 if path.exists():raise FileExistsError(path)
 with temp.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False)
 temp.rename(path)

class Job:
 def __init__(self):
  self.handle=ok(K.CreateJobObjectW(None,None));self.launch=None;self.launch_id=None;self.query_history=[]
  try:
   lim=Limits();lim.basic.flags=0x2000|0x200|0x8;lim.basic.active=8;lim.job_memory=256*2**20
   ok(K.SetInformationJobObject(self.handle,9,C.byref(lim),C.sizeof(lim)))
   check=Limits();ok(K.QueryInformationJobObject(self.handle,9,C.byref(check),C.sizeof(check),None))
   if check.basic.flags!=lim.basic.flags or check.basic.active!=8 or check.job_memory!=256*2**20:raise RuntimeError('job_limits_readback_mismatch')
   self.limits={'flags':check.basic.flags,'active_process_limit':check.basic.active,'job_memory_bytes':check.job_memory}
  except BaseException:close(self.handle);self.handle=None;raise
 def contains(self,handle):
  value=T.BOOL();ok(K.IsProcessInJob(handle,self.handle,C.byref(value)));return bool(value.value)
 def pids(self):
  # A complete API buffer is checked independently of the configured active-process limit.
  # Windows may report associated helper processes in this list as well.
  size=1024
  for attempt in range(4):
   b=C.create_string_buffer(size);needed=T.DWORD();result=K.QueryInformationJobObject(self.handle,3,b,size,C.byref(needed));error=C.get_last_error() if not result else 0
   assigned=C.c_uint32.from_buffer(b,0).value;count=C.c_uint32.from_buffer(b,4).value
   capacity=(size-8)//8
   self.query_history.append({'assigned':assigned,'returned':count,'needed_bytes':needed.value,'capacity':capacity,'api_error':error})
   if error not in (0,234):raise C.WinError(error)
   if count>capacity:raise RuntimeError('job_pid_buffer_bounds')
   if not error and assigned==count:return [C.c_size_t.from_buffer(b,8+8*i).value for i in range(count)]
   size=max(size*2,8+8*assigned,needed.value)
   if size>1024*1024:raise RuntimeError('unexpected_job_pid_buffer_size')
  raise RuntimeError('job_pid_list_incomplete:'+repr(self.query_history[-4:]))
 def start(self,argv,cwd):
  if self.launch:raise RuntimeError('one_launcher_only')
  si=Startup();si.cb=C.sizeof(si);pi=ProcessInfo();buf=C.create_unicode_buffer(subprocess.list2cmdline([str(x) for x in argv]))
  ok(K.CreateProcessW(str(argv[0]),buf,None,None,False,0x08000000|0x4,None,str(cwd),C.byref(si),C.byref(pi)))
  self.launch=pi.process;self.launch_id=ident(self.launch)
  try:
   ok(K.AssignProcessToJobObject(self.handle,self.launch))
   if not self.contains(self.launch):raise RuntimeError('launcher_not_in_job')
   result=K.ResumeThread(pi.thread)
   if result!=1:raise RuntimeError('unexpected_suspend_count:'+str(result))
  except BaseException:
   # This exact handle was just created suspended by this object; it is not a discovered PID.
   if running(self.launch):ok(K.TerminateProcess(self.launch,91));wait_exit(self.launch)
   raise
  finally:close(pi.thread)
  return self.launch_id
 def stop(self):
  retained=[];ids=[];images=[];already_gone=[]
  try:
   try:
    for pid in self.pids():
     # Cleanup needs identity and wait rights, not memory-inspection rights.
     handle=K.OpenProcess(0x100000|0x1000,False,pid)
     if not handle:
      error=C.get_last_error()
      if pid not in self.pids():already_gone.append({'pid':pid,'reason':'exited_before_open'});continue
      raise C.WinError(error)
     if not self.contains(handle):
      close(handle);already_gone.append({'pid':pid,'reason':'no_longer_a_job_member'});continue
     retained.append(handle);identity=ident(handle);ids.append(identity)
     try:info={**identity,'image':image_path(handle),'image_query':'ok'}
     except OSError as exc:
      info={**identity,'image':None,'image_query':'optional_cleanup_metadata_unavailable','was_running':running(handle),'winerror':exc.winerror}
     images.append(info)
   finally:
    # Optional diagnostics must never prevent terminating our own newly created job.
    ok(K.TerminateJobObject(self.handle,90))
   for handle in retained:wait_exit(handle)
   end=time.monotonic()+5
   while self.pids() and time.monotonic()<end:time.sleep(.01)
   if self.pids():raise RuntimeError('job_cleanup_incomplete')
   if self.launch:wait_exit(self.launch)
   return {'retained_member_identities':ids,'all_retained_exited':True,'job_active_processes':0,'member_images':images,'already_gone_members':already_gone,'pid_queries':self.query_history,'limits_readback':self.limits}
  finally:
   for handle in retained:close(handle)
 def close(self):
  if self.handle:
   h=self.handle;self.handle=None;close(h)
  if self.launch:
   h=self.launch;self.launch=None;close(h)
