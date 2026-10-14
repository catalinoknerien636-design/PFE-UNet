"""Companion entry for unchanged frozen41 training with safe owned-file reads."""
import sys,json,pathlib,importlib.util,datetime
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'41_rgbp_portable_runner'))
import rgbp_core as c
import rgbp_protocol as p
import safe_reader57 as safe
spec=importlib.util.spec_from_file_location('frozen_rgbp_queue41',c.HERE/'run_queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

def validate():
 contract=safe.read_json(HERE/'contract.json')
 for path,digest in contract['sources'].items():
  if safe.digest(path)!=digest:raise ValueError('Bound companion source changed: '+path)
 tests=safe.read_json(HERE/'cpu_attempt_v1/report.json')
 if not tests['passed'] or tests['gpu_started'] or tests['source_sha256']!=contract['tested_sources']:raise ValueError('Bound CPU tests required')
 frozen=p.validate(p.read(p.PROTOCOL))
 pcod=safe.read_json(ROOT/'25_pcod_portable_runner/queue_status.json')
 if pcod['state']!='completed' or len(pcod['completed'])!=28:raise ValueError('PCOD28 must be complete')
 accepted=[]
 for stage in ('55_pcod_checkpoint_evidence','56_pcod_checkpoint_evidence_remaining'):
  actual=ROOT/stage/'actual_checks';report=safe.read_json(actual/'report.json')
  if not report['passed']:raise ValueError('PCOD internal evidence failed')
  for row in report['runs']:
   if safe.digest(actual/(row['run_id']+'.json'))!=row['sha256']:raise ValueError('PCOD internal evidence changed')
   accepted.append(row['run_id'])
 if set(accepted)!=set(pcod['completed']) or len(accepted)!=28:raise ValueError('PCOD internal coverage incomplete')
 return frozen,contract

def main():
 frozen,contract=validate();q.require_resources();q.guard.require_idle()
 for name in ('queue_status.json','queue.lock','runs'):
  if (c.HERE/name).exists():raise ValueError('First launch only; existing41 runtime requires separate inspected recovery')
 launch=HERE/'launch_v1';launch.mkdir(exist_ok=False)
 owner=q.guard.process_identity(__import__('os').getpid())
 c.candidate.save_new(launch/'identity.json',dict(owner=owner,contract_sha256=safe.digest(HERE/'contract.json'),protocol_sha256=c.t.sha(p.PROTOCOL),utc=c.t.utc_now(),training_implementation='immutable41',checkpoint_reader='bounded_nonexecuting57',resume=False))
 reader=safe.Reader(entries=frozen['configs'])
 try:
  with safe.installed(reader):q.run(allow_resume=False)
  c.candidate.save_new(launch/'completed.json',dict(utc=c.t.utc_now(),safe_reads=reader.calls,test_evaluated=False))
 except BaseException as error:
  c.candidate.save_new(launch/'failure.json',dict(utc=c.t.utc_now(),error=type(error).__name__+': '+str(error),automatic_retry=False,safe_reads=reader.calls));raise
if __name__=='__main__':main()
