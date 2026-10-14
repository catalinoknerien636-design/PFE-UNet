"""Bounded CPU tests with preserved attempt evidence; no scientific approval."""
import argparse
import io
import json
import re
import shutil
import unittest
import rgbp_core as c
import rgbp_protocol as p
import test_runner41 as tests


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',default='cpu_attempt_v1')
    args=parser.parse_args()
    if not re.fullmatch(r'cpu_attempt_v[1-9][0-9]*',args.attempt):
        raise ValueError('Attempt must be a local version label')
    if p.TESTS.exists() or p.PROTOCOL.exists() or p.PROPOSAL.exists():
        raise FileExistsError('Release evidence already exists; do not overwrite')
    tests.ATTEMPT=c.HERE/args.attempt
    tests.ATTEMPT.mkdir()
    sources={str(path.relative_to(c.ROOT)):c.t.sha(path) for path in (*c.RUNTIME_FILES,*p.TEST_FILES)}
    snapshots=tests.ATTEMPT/'source_snapshot'
    snapshots.mkdir()
    for path in (*c.RUNTIME_FILES,*p.TEST_FILES):
        shutil.copy2(path,snapshots/path.name)
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    report={'utc':c.t.utc_now(),'passed':result.wasSuccessful(),'tests':result.testsRun,
        'log':log.getvalue(),'gpu_started':False,'formal_runs_started':0,
        'source_sha256':sources,'attempt':args.attempt,
        'synthetic_cpu_methods':list(tests.FIXTURES),'synthetic_size':32,'synthetic_width':4,
        'synthetic_epochs':2,'scientific_admission_granted':False,
        'real_windows_lock_test':True,'actual_model_device_acceptance_repeated':False}
    c.candidate.save_new(tests.ATTEMPT/'test_results.json',report)
    if not result.wasSuccessful():raise RuntimeError(log.getvalue())
    assert all(c.t.sha(c.ROOT/name)==digest for name,digest in sources.items())
    c.candidate.save_new(p.TESTS,report)
    proposal=p.specification(False)
    c.candidate.save_new(p.PROPOSAL,proposal)
    print(json.dumps({'passed':True,'tests':result.testsRun,'synthetic_methods':len(tests.FIXTURES),
        'disabled_configs':len(proposal['configs']),'gpu_started':False,
        'proposal_sha256':c.t.sha(p.PROPOSAL),'formal_protocol_exists':p.PROTOCOL.exists()},indent=2))


if __name__=='__main__':main()
