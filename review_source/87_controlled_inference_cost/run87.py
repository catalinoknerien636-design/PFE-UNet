from pathlib import Path
import sys,contextlib,msvcrt,traceback,hashlib
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'80_pgsnet_module_qualification'));import qualify80 as q
from job87 import w
from supervisor87 import Supervisor

def verify_cpu(c):
 test=q.read(H/'cpu_tests.json')
 if not test['passed'] or test['timing_source_sha256']!=q.a.digest(H/'timing87.py'):raise RuntimeError('timing_CPU_tests_required')

def validate_run(out,expected,phase):
 done=q.read(out/'done.json');assert done['run_id']==expected and done['phase']==phase and done['forwards']==35 and done['warmups']==5 and done['measurements']==30
 assert not done['numerical_comparison'] and not done['test_images_read'] and done['training_steps']==0
 iteration_files=sorted(out.glob('iteration_*.json'));assert len(iteration_files)==35
 last=0
 for i,p in enumerate(iteration_files):
  row=q.read(p);assert row['iteration']==i and row['kind']==('warmup' if i<5 else 'measured') and row['phase']==phase
  v=row['interval'];assert v['end_ns']>v['start_ns']>=last and v['end_ns']-v['start_ns']==v['elapsed_ns'];last=v['end_ns']
  assert row['finite'] and row['readback_completed'] and row['output_shape']==[1,1,352,352] and row['output_bytes']==495616 and not row['gradient_enabled'] and not row['requires_grad']
 assert done['monotonic_ns']>=last
 cleanup=q.read(out/'cleanup.json');assert not cleanup['errors'] and cleanup['result']['job_active_processes']==0
 s=q.read(out/'samples.json');assert s and s[0]['phase']=='before_payload'
 return {'run_id':expected,'phase':phase,'passed':True,'forwards':35,'worker':q.read(out/'binding.json')['worker'],'min_available_ram_gib':min(x['host']['available_ram_gib'] for x in s),'max_private_gib':max(x['worker_memory']['private_bytes'] for x in s)/2**30,'max_working_set_gib':max(x['worker_memory']['working_set_bytes'] for x in s)/2**30,'done_sha256':q.a.digest(out/'done.json'),'samples_sha256':q.a.digest(out/'samples.json')}

def main(phase):
 if phase not in ('qualification','matrix'):raise ValueError('phase')
 c=q.read(H/'contract.json');sealed=q.a.digest(H/'contract.json');q.verify(c);verify_cpu(c);q.guard.require_idle();q.resources(17)
 gate=None
 if phase=='matrix':
  gate=q.read(H/'qualification_v1/complete.json');assert gate['passed'] and gate['qualified_run_ids']==c['qualification_order'] and gate['contract_sha256']==sealed
  audit=q.read(H/'qualification_assessment.json');assert audit['passed'] and audit['complete_sha256']==q.a.digest(H/'qualification_v1/complete.json') and audit['contract_sha256']==sealed
  assert c['allow_matrix_only_after_all_five_qualified']
 order=c['qualification_order'] if phase=='qualification' else c['protocol']['run_order'];out=H/(phase+'_v1');out.mkdir(exist_ok=False);finished=[];current=None
 try:
  with contextlib.ExitStack() as locks:
   for rel in c['locks']:
    f=locks.enter_context((R/rel).open('r+b'));f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   before=q.old.protection();q.save(out/'protection_before.json',before)
   for index,runid in enumerate(order):
    current=runid
    if q.a.digest(H/'contract.json')!=sealed:raise RuntimeError('contract_changed')
    q.verify(c);q.resources(17)
    plan={'name':'inference_cost','run_id':runid,'phase':phase,'timeout_seconds':180}
    if phase=='matrix':plan['qualification_sha256']=q.a.digest(H/'qualification_v1/complete.json')
    target=out/runid;s=Supervisor(target,c['runtime'],c['expected_worker_image'],plan)
    try:s.start();s.bind();s.start_guard();s.approve();reason=s.monitor()
    finally:s.finish()
    if reason!='normal':
     q.save(target/'controller_failure.json',{'reason':reason,'run_id':runid,'phase':phase,'automatic_retry':False});raise RuntimeError('worker_'+reason)
    result=validate_run(target,runid,phase);q.save(target/'result.json',result);finished.append(result)
    print(str(index+1)+'/'+str(len(order))+' '+phase+' '+runid+' private='+str(round(result['max_private_gib'],3))+'GiB',flush=True)
   after=q.old.protection();assert after==before;q.verify(c);q.save(out/'protection_after.json',after)
   q.save(out/'complete.json',{'passed':True,'phase':phase,'contract_sha256':sealed,'qualified_run_ids':order if phase=='qualification' else c['qualification_order'],'completed_run_ids':order,'results':finished,'total_forwards':35*len(finished),'paper_timing_results':phase=='matrix','original_results_unchanged':True,'resources':q.resources(),'utc':q.utc()})
  q.guard.require_idle();print('COMPLETE '+phase,flush=True)
 except BaseException as exc:
  q.save(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'run_id':current,'completed_results':finished,'automatic_retry':False,'utc':q.utc()});raise
if __name__=='__main__':main(sys.argv[1])
