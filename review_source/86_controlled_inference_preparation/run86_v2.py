from pathlib import Path
import sys,contextlib,msvcrt,traceback
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'80_pgsnet_module_qualification'));import qualify80 as q
from job86 import w
from supervisor86_v2 import Supervisor

def main():
 c=q.read(H/'contract_v2.json');q.verify(c);q.guard.require_idle();q.resources(17)
 admission=q.read(H/'pilot_admission_v2.json')
 if not admission['passed'] or q.a.digest(H/'meta_v2/complete.json')!=admission['meta_complete_sha256'] or q.a.digest(H/'contract_v2.json')!=admission['contract_sha256']:raise RuntimeError('independent_meta_admission_required')
 out=H/'pilot_v2';out.mkdir(exist_ok=False)
 try:
  with contextlib.ExitStack() as locks:
   for rel in c['locks']:
    f=locks.enter_context((R/rel).open('r+b'));f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   before=q.old.protection();q.save(out/'protection_before.json',before)
   plan={'name':'single_model_pilot','timeout_seconds':c['pilot_timeout_seconds'],'meta_complete_sha256':admission['meta_complete_sha256'],'run_id':c['pilot_run']['entry']['config']['run_id']}
   supervisor=Supervisor(out/'worker',c['runtime'],c['expected_worker_image'],plan)
   try:supervisor.start();supervisor.bind();supervisor.start_guard();supervisor.approve();reason=supervisor.monitor()
   finally:supervisor.finish()
   if reason!='normal':raise RuntimeError('pilot_'+reason)
   done=q.read(out/'worker/done.json');assert done['model_forwards']==3 and not done['inference_benchmark'] and not done['numerical_comparison']
   cleanup=q.read(out/'worker/cleanup.json');assert not cleanup['errors'] and cleanup['result']['job_active_processes']==0
   q.verify(c);after=q.old.protection();assert before==after
   q.save(out/'protection_after.json',after)
   q.save(out/'complete.json',{'passed':True,'reason':reason,'contract_sha256':q.a.digest(H/'contract_v2.json'),'single_pilot_only':True,'benchmark_matrix_executed':False,'true_worker':supervisor.bound['worker'],'launcher':supervisor.launch_id,'done':done,'resources':q.resources(),'utc':q.utc()})
  q.guard.require_idle();print(q.read(out/'complete.json'))
 except BaseException as exc:q.save(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'utc':q.utc(),'automatic_retry':False});raise
if __name__=='__main__':main()

