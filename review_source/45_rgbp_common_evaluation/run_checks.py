"""Exclusive CPU checks and source binding; no dataset pixels or checkpoints."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import unittest
import offline_evaluate as o
import test_evaluation
import test_offline

HERE=Path(__file__).resolve().parent


def sources():
    paths=list(HERE.glob('*.py'))
    paths += [o.ROOT/p for p in ('06_experiment_preparation/boundary_metrics.py',
        '06_experiment_preparation/rgbp_glass_dataset.py','12_training/train_rgbp.py')]
    return {str(p.relative_to(o.ROOT)):o.sha(p) for p in paths}


def main():
    out=HERE/'cpu_attempt_v1'
    out.mkdir(exist_ok=False)
    bound=sources()
    with (out/'source_snapshot.json').open('x',encoding='utf-8') as f:json.dump(bound,f,indent=2)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in HERE.glob('*.py'):
        with (snapshot/p.name).open('xb') as f:f.write(p.read_bytes())
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m)
                             for m in (test_evaluation,test_offline))
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    with (out/'tests.log').open('x',encoding='utf-8') as f:f.write(log.getvalue())
    assert sources()==bound
    report={'utc':datetime.now(timezone.utc).isoformat(),'passed':result.wasSuccessful(),
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'source_sha256':bound,'synthetic_only':True,'test_images_read':False,
        'model_created':False,'gpu_started':False,'checkpoint_loaded':False,
        'formal_dataset_evaluated':False,'PGSNet_admitted':False}
    with (out/'results.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(log.getvalue());print(json.dumps(report,indent=2))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
