from pathlib import Path
import contextlib,ctypes,gc,hashlib,json,msvcrt,os,shutil,subprocess,sys,threading,time,traceback
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;R=H.parent
for d in ['36_pgsnet_pretrained_contract','24_pcod_reference_acceptance','75_pgsnet_resource_profile','79_pgsnet_checkpoint_candidate']:
    sys.path.insert(0,str(R/d))
import adapter as a
import process_guard as guard
import profile75_v2 as old

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def utc():return datetime.now(timezone.utc).isoformat()
def verify(c):
    for p,h in c['sources'].items():
        if a.digest(p)!=h:raise ValueError('Bound source changed: '+p)
def resources(minimum=8):
    v={'available_ram_gib':a.memory_guard(minimum),'disk_free_gib':shutil.disk_usage(H).free/2**30}
    if v['disk_free_gib']<20:raise RuntimeError('Original disk gate')
    return v

class Counters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
def process_memory():
    k=ctypes.WinDLL('kernel32',use_last_error=True);p=ctypes.WinDLL('psapi',use_last_error=True)
    k.GetCurrentProcess.restype=ctypes.c_void_p
    p.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    p.GetProcessMemoryInfo.restype=ctypes.c_int
    v=Counters();v.cb=ctypes.sizeof(v)
    if not p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(v),v.cb):raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_gib':v.WorkingSetSize/2**30,'os_peak_working_set_gib':v.PeakWorkingSetSize/2**30,'private_usage_gib':v.PrivateUsage/2**30,'os_peak_pagefile_usage_gib':v.PeakPagefileUsage/2**30}
def violation(ram,private,elapsed,c):
    if ram<c['abort_available_ram_gib']:return 'host RAM reserve reached'
    if private>c['private_process_cap_gib']:return 'process private-memory cap reached'
    if elapsed>c['worker_timeout_seconds']:return 'bounded worker timeout'
    return None
class Monitor:
    def __init__(self,out,c):
        self.out=out;self.c=c;self.stop=threading.Event();self.start=time.perf_counter();self.rows=[];self.failure=None
    def sample(self,phase):
        m=process_memory();m.update(available_ram_gib=a.available_gib(),elapsed=time.perf_counter()-self.start,phase=phase);self.rows.append(m)
        if len(self.rows)>40000:raise RuntimeError('Sampling bound')
        reason=violation(m['available_ram_gib'],m['private_usage_gib'],m['elapsed'],self.c)
        if reason:raise RuntimeError(reason)
        return m
    def loop(self):
        try:
            while not self.stop.wait(.05):self.sample('periodic')
        except BaseException as e:
            # This watchdog exits only its own worker; it never selects another PID.
            self.failure=str(e)
            try:save(self.out/'resource_abort.json',{'error':str(e),'identity':guard.process_identity(os.getpid()),'last_samples':self.rows[-10:],'utc':utc()})
            finally:os._exit(91)
    def begin(self):
        self.sample('start');self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def finish(self):
        self.stop.set();self.thread.join();self.sample('finish')
        r={'sampling_seconds':.05,'samples':len(self.rows),'minimum_available_ram_gib':min(v['available_ram_gib'] for v in self.rows),'maximum_sampled_private_gib':max(v['private_usage_gib'] for v in self.rows),'os_peak_working_set_gib':max(v['os_peak_working_set_gib'] for v in self.rows),'os_peak_pagefile_usage_gib':max(v['os_peak_pagefile_usage_gib'] for v in self.rows),'not_gpu_memory':True,'subsampling_cannot_prove_continuous_reserve':True}
        save(self.out/'memory_samples.json',self.rows);return r

def tensor_record(t):
    if t is None:return None
    import torch
    if t.device.type!='cpu':raise RuntimeError('CPU tensors only')
    d=t.detach().contiguous();v=d.reshape(-1);h=hashlib.sha256()
    for start in range(0,v.numel(),262144):
        chunk=v[start:start+262144]
        if chunk.is_floating_point() and not bool(torch.isfinite(chunk).all()):raise RuntimeError('Nonfinite tensor')
        h.update(memoryview(chunk.numpy()).cast('B'))
    return {'shape':list(d.shape),'dtype':str(d.dtype),'sha256':h.hexdigest(),'elements':d.numel()}
def mapping(xs):return {n:tensor_record(v) for n,v in xs}

def select_region(report):
    regions={k:v for branch in report['profile']['conformer_recomputation_planning'] for k,v in branch['per_inner_baseline_saved_bytes'].items()}
    maximum=max(regions.values());ties=sorted(k for k,v in regions.items() if v==maximum)
    return {'selected':ties[0],'largest_baseline_saved_bytes':maximum,'ties':ties,'rule':'maximum saved bytes, lexicographic tie break; no metric selection'}

def selftest():
    import torch
    tests=[]
    c={'abort_available_ram_gib':8.5,'private_process_cap_gib':8.,'worker_timeout_seconds':1200}
    assert violation(18,1,1,c) is None;assert violation(8.49,1,1,c);assert violation(18,8.01,1,c);assert violation(18,1,1201,c);tests.append('resource gates and bounded deadline')
    x=torch.arange(40,dtype=torch.float32).reshape(5,8);assert tensor_record(x)==tensor_record(x.clone());assert tensor_record(x)!=tensor_record(x+1);tests.append('hash detects tensor mutation')
    assert tensor_record(x.t())==tensor_record(x.t().contiguous());tests.append('noncontiguous logical tensor hash')
    try:tensor_record(torch.tensor([float('nan')]))
    except RuntimeError:pass
    else:raise AssertionError('NaN admitted')
    tests.append('nonfinite rejection')
    assert select_region(read(R/'79_pgsnet_checkpoint_candidate/metadata_attempt_v1/report.json'))['selected']=='conformer1.conv_trans_2';tests.append('independent largest-region selection')
    m=process_memory();assert m['private_usage_gib']>0 and m['os_peak_working_set_gib']>0;tests.append('live Win32 process counters')
    save(H/'prelaunch_tests.json',{'passed':True,'checks':tests,'torch':str(torch.__version__),'utc':utc()});print(json.dumps({'tests_passed':len(tests)}),flush=True)

def worker(runtime,mode):
    c=read(H/'contract.json');verify(c);guard.require_idle();resources(c['launch_available_ram_gib'])
    out=H/(runtime+'_'+mode+'_v1');out.mkdir(exist_ok=False)
    save(out/'identity.json',{'owner':guard.process_identity(os.getpid()),'contract_sha256':a.digest(H/'contract.json'),'runtime':runtime,'mode':mode,'resources':resources(8),'utc':utc()})
    monitor=Monitor(out,c);monitor.begin()
    try:
        import torch
        expected=c['runtimes'][runtime]
        assert str(torch.__version__)==expected['torch_version'] and Path(sys.prefix).resolve()==Path(expected['prefix']).resolve()
        assert Path(torch.__file__).resolve().is_relative_to(Path(expected['prefix']).resolve()) and torch.version.cuda is None
        if runtime=='cpu214':a.require_runtime()
        assert 'torch_directml' not in sys.modules
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        import checkpoint_candidate as cp
        module=a.pgsnet_module();torch.manual_seed(c['model_seed'])
        model=module.ConvTransBlock(**c['constructor']).cpu().train()
        # Apply the same initializer that the containing upstream Conformer applies.
        model.apply(lambda m:module.Conformer._init_weights(None,m))
        before=mapping(model.state_dict().items())
        torch.manual_seed(c['input_seed'])
        inputs=[torch.randn(*shape,dtype=torch.float32,requires_grad=True) for shape in c['input_shapes']]
        input_before=mapping((str(i),x) for i,x in enumerate(inputs));rng_before=tensor_record(torch.get_rng_state())
        opt=torch.optim.SGD(model.parameters(),**c['optimizer'])
        restore=cp.install(model,['']) if mode=='candidate' else []
        bn_ids={n:id(v) for n,v in model.named_buffers()}
        monitor.sample('before_forward');start=time.perf_counter()
        print(json.dumps({'runtime':runtime,'mode':mode,'phase':'forward','pid':os.getpid()}),flush=True)
        output=model(*inputs);forward_seconds=time.perf_counter()-start
        assert [list(v.shape) for v in output]==c['input_shapes']
        output_records=mapping((str(i),v) for i,v in enumerate(output))
        loss=sum(v.square().mean() for v in output)
        loss_record=tensor_record(loss)
        monitor.sample('before_backward');start=time.perf_counter()
        print(json.dumps({'runtime':runtime,'mode':mode,'phase':'backward'}),flush=True)
        loss.backward();backward_seconds=time.perf_counter()-start
        assert bn_ids=={n:id(v) for n,v in model.named_buffers()}
        grads=mapping((n,p.grad) for n,p in model.named_parameters())
        assert all(p.grad is not None for p in model.parameters())
        input_grads=mapping((str(i),x.grad) for i,x in enumerate(inputs))
        buffers=mapping(model.named_buffers())
        counters={n:int(m.num_batches_tracked) for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
        assert counters and all(v==1 for v in counters.values())
        opt.step();parameters_after=mapping(model.named_parameters())
        momentum=mapping((n,opt.state[p]['momentum_buffer']) for n,p in model.named_parameters())
        input_after=mapping((str(i),x) for i,x in enumerate(inputs));assert input_before==input_after
        rng_after=tensor_record(torch.get_rng_state());future_draw=tensor_record(torch.randn(17))
        cp.uninstall(restore);monitor.sample('after_update')
        evidence={'initial_state':before,'input_before':input_before,'output':output_records,'loss':loss_record,'parameter_gradients':grads,'input_gradients':input_grads,'BN_buffers_after_backward':buffers,'BN_counters':counters,'parameters_after_update':parameters_after,'momentum':momentum,'rng_before':rng_before,'rng_after':rng_after,'future_random_draw':future_draw}
        save(out/'tensor_evidence.json',evidence)
        parameter_count=sum(p.numel() for p in model.parameters());del model,opt,inputs,output,loss,restore;gc.collect()
        memory=monitor.finish();verify(c)
        report={'completed':True,'runtime':runtime,'torch':str(torch.__version__),'mode':mode,'parameter_count':parameter_count,'gradient_tensors':len(grads),'BN_layers':len(counters),'actual_SGD_steps':1,'batch':18,'input_shapes':c['input_shapes'],'forward_seconds':forward_seconds,'backward_seconds':backward_seconds,'memory':memory,'tensor_evidence_sha256':a.digest(out/'tensor_evidence.json'),'full_PGSNet_executed':False,'GPU_used':False,'real_data_or_external_weights_used':False,'utc':utc()}
        save(out/'report.json',report);print(json.dumps(report),flush=True)
    except BaseException as e:
        monitor.stop.set()
        save(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'utc':utc(),'automatic_retry':False});raise

def compare(runtime):
    baseline=read(H/(runtime+'_baseline_v1/tensor_evidence.json'));candidate=read(H/(runtime+'_candidate_v1/tensor_evidence.json'))
    keys=list(baseline);assert keys==list(candidate)
    matches={k:baseline[k]==candidate[k] for k in keys}
    return {'passed':all(matches.values()),'runtime':runtime,'comparison':'exact shape/dtype/SHA256 of every logical tensor and exact scalar/counter equality, within runtime only','categories':matches,'baseline_report_sha256':a.digest(H/(runtime+'_baseline_v1/report.json')),'candidate_report_sha256':a.digest(H/(runtime+'_candidate_v1/report.json'))}

def run():
    c=read(H/'contract.json');verify(c);guard.require_idle();resources(c['launch_available_ram_gib'])
    out=H/'serial_execution_v1';out.mkdir(exist_ok=False)
    with contextlib.ExitStack() as stack:
        for rel in c['locks']:
            f=stack.enter_context((R/rel).open('r+b'));f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        guard.require_idle();before=old.protection();save(out/'identity.json',{'owner':guard.process_identity(os.getpid()),'before_protection':before,'contract_sha256':a.digest(H/'contract.json'),'utc':utc()})
        try:
            results=[]
            for runtime,spec in c['runtimes'].items():
                for mode in ['baseline','candidate']:
                    resources(c['launch_available_ram_gib']);verify(c)
                    logfile=stack.enter_context((out/(runtime+'_'+mode+'.log')).open('x',encoding='utf-8'))
                    proc=subprocess.Popen([str(Path(spec['prefix'])/'Scripts/python.exe'),'-X','utf8','-B',str(H/'qualify80.py'),'worker',runtime,mode],stdout=logfile,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    save(out/(runtime+'_'+mode+'_child.json'),{'owner':guard.process_identity(proc.pid),'utc':utc()})
                    try:code=proc.wait(timeout=c['worker_timeout_seconds']+90)
                    except subprocess.TimeoutExpired:
                        # Popen retains the exact Windows process handle; no lookup by reused PID.
                        proc.terminate();proc.wait(timeout=20);raise RuntimeError('Exact child timeout')
                    if code:raise RuntimeError(f'{runtime} {mode} exited {code}; no automatic retry')
                result=compare(runtime);save(out/(runtime+'_comparison.json'),result);results.append(result)
                if not result['passed']:raise RuntimeError('Component equivalence failed: '+runtime)
            after=old.protection();verify(c);guard.require_idle()
            save(out/'complete.json',{'passed':True,'comparisons':results,'before_protection':before,'after_protection':after,'resources':resources(8),'utc':utc(),'no_cross_runtime_equivalence_claim':True,'full_model_and_GPU_qualification':False})
            print(json.dumps({'completed':True,'runtimes':list(c['runtimes'])}),flush=True)
        except BaseException as e:
            save(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'utc':utc(),'automatic_retry':False});raise
if __name__=='__main__':
    if sys.argv[1]=='selftest':selftest()
    elif sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=='run':run()
    else:raise ValueError('Unknown action')
