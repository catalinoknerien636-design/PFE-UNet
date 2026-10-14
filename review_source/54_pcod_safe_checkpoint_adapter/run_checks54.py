"""Exclusive synthetic CPU test attempt; no real checkpoint loading."""
from datetime import datetime, timezone
import io
import shutil
import unittest
import metadata54 as m
import convert54 as c


def main():
    plan = m.read(m.HERE/'plan.json'); c.validate_plan(plan)
    torch = c.runtime(); available = c.memory_guard()
    target = m.HERE/'cpu_attempt_v1'; target.mkdir(exist_ok=False)
    (target/'fixtures').mkdir(); (target/'source_snapshot').mkdir()
    for path in m.HERE.glob('*.py'): shutil.copyfile(path, target/'source_snapshot'/path.name)
    before = list(torch.serialization.get_safe_globals())
    import test54
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test54))
    (target/'unittest.log').write_text(stream.getvalue(), encoding='utf-8')
    stable = before == list(torch.serialization.get_safe_globals())
    report = {'passed': result.wasSuccessful() and stable, 'tests': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
        'default_serialization_registry_unchanged': stable, 'torch_version': str(torch.__version__),
        'utc': datetime.now(timezone.utc).isoformat(), 'initial_available_memory_bytes': available,
        'plan_sha256': m.sha(m.HERE/'plan.json'), 'model_created': False, 'real_images_or_checkpoints': False,
        'GPU': False, 'synthetic_archive_layout': 'observed old members; newer writer hints omitted in fixtures only'}
    m.write(target/'cpu_test_results.json', report)
    print(report, flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
