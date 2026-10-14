"""Exclusive CPU verification; no real test-data inference."""
import argparse
import json
from pathlib import Path
import unittest
import torch
import export_core as e
import test_export as tests


def sources():
    return {str(p.relative_to(e.ROOT)): e.sha(p) for p in sorted(e.HERE.glob('*.py'))}


def main(attempt):
    torch.set_num_threads(1)
    output = e.HERE / attempt
    tests.WORK = output / 'fixtures'
    output.mkdir(exist_ok=False)
    before = sources()
    source_copy = output / 'source_snapshot'
    source_copy.mkdir()
    for path in e.HERE.glob('*.py'):
        with (source_copy / path.name).open('xb') as stream: stream.write(path.read_bytes())
    with (output / 'unittest.log').open('x', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    report = {'passed': result.wasSuccessful(), 'tests': result.testsRun, 'errors': len(result.errors),
        'failures': len(result.failures), 'method_checks': tests.METHOD_RESULTS, 'source_sha256': before,
        'test_pixels_read': False, 'external_weights_loaded': False, 'gpu_started': False,
        'formal_inference_executed': False, 'full_width_device_export_verified': False}
    if sources() != before: raise ValueError('Test source changed')
    e.save_new(output / 'results.json', report)
    print(json.dumps(report))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True, choices=['cpu_attempt_v2', 'cpu_attempt_v3'])
    main(parser.parse_args().attempt)
