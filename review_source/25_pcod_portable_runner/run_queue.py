"""Single-process serial formal runner; unavailable until the release is frozen."""
from contextlib import contextmanager, ExitStack
import gc
import json
import msvcrt
import os
from pathlib import Path
import shutil

import execution_protocol as protocol
import portable_core as core

guard = core.acceptance.guard
STATE = core.HERE / 'queue_status.json'


@contextmanager
def byte_lock(path, create=False):
    with path.open('a+b' if create else 'r+b') as stream:
        if create and stream.seek(0, 2) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def completed_run(cfg, expected_identity, protocol_sha):
    out = Path(cfg['output_dir'])
    marker = out / 'portable_complete.json'
    if not marker.is_file():
        return False
    evidence = protocol.read(marker)
    if (evidence['protocol_sha256'] != protocol_sha or evidence['run_id'] != cfg['run_id']
            or evidence['identity'] != expected_identity):
        raise ValueError('Completed-run provenance differs')
    required = {'latest.pt', 'best.pt', 'history.csv', 'best_validation_per_image.json', 'status.json'}
    if set(evidence['artifact_sha256']) != required:
        raise ValueError('Completed-run artifact set differs')
    if any(core.t.sha(out / name) != value for name, value in evidence['artifact_sha256'].items()):
        raise ValueError('Completed-run artifact changed')
    state = core.inspect_run(cfg, expected_identity)
    if state['epoch'] != cfg['epochs']:
        raise ValueError('Incomplete run has completion marker')
    return True


def mark_completed(cfg, identity, protocol_sha):
    out = Path(cfg['output_dir'])
    record = protocol.read(out / 'status.json')
    if (record['state'] != 'completed' or record['completed_epoch'] != cfg['epochs']
            or record['test_evaluated'] is not False):
        raise ValueError('Training did not complete the full registered budget')
    if core.inspect_run(cfg, identity)['epoch'] != cfg['epochs']:
        raise ValueError('Checkpoint did not complete the budget')
    if record['checkpoint_sha256'] != core.t.sha(out / 'latest.pt'):
        raise ValueError('Completion status refers to a different checkpoint')
    latest = core.backend.torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    best = core.backend.torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    if (best.get('identity') != identity or best.get('epoch') != latest['best_epoch']
            or best.get('validation_iou') != latest['best_iou']):
        raise ValueError('Exported best checkpoint differs from validation selection')
    core.backend.h.compare_tree(best['model_state'], latest['best_model_state'], {'atol': 0, 'rtol': 0})
    del latest, best
    core.t.write_json(out / 'portable_complete.json', {
        'run_id': cfg['run_id'], 'identity': identity, 'protocol_sha256': protocol_sha,
        'completed_utc': core.t.utc_now(), 'test_evaluated': False,
        'artifact_sha256': {name: core.t.sha(out / name) for name in (
            'latest.pt', 'best.pt', 'history.csv', 'best_validation_per_image.json', 'status.json')}})


def main():
    # No device selection, training-state write, or resume before full admission.
    frozen = protocol.validate(protocol.read(protocol.PROTOCOL))
    frozen_sha = core.t.sha(protocol.PROTOCOL)
    with ExitStack() as locks:
        locks.enter_context(byte_lock(core.HERE / 'queue.lock', create=True))
        locks.enter_context(byte_lock(core.ROOT / '21_portable_rgbp_development/queue.lock'))
        locks.enter_context(byte_lock(core.ROOT / '24_pcod_reference_acceptance/dependency.lock'))
        guard.require_idle()
        owner = guard.process_identity(os.getpid())
        completed = []
        try:
            for entry in frozen['configs']:
                if core.t.sha(protocol.PROTOCOL) != frozen_sha:
                    raise ValueError('Protocol changed while running')
                cfg, expected_identity = entry['config'], entry['identity']
                if core.identity(cfg) != expected_identity:
                    raise ValueError('Runtime/input identity changed')
                if completed_run(cfg, expected_identity, frozen_sha):
                    completed.append(cfg['run_id'])
                    continue
                if shutil.disk_usage(core.ROOT).free < 20 * 1024**3:
                    raise RuntimeError('Less than20GiB free; do not start the next run')
                core.t.write_json(STATE, {'state': 'running', 'owner': owner,
                    'current_run': cfg['run_id'], 'completed': completed, 'utc': core.t.utc_now(),
                    'protocol_sha256': frozen_sha, 'test_evaluated': False})
                core.execute(cfg, expected_identity)
                mark_completed(cfg, expected_identity, frozen_sha)
                completed.append(cfg['run_id'])
                gc.collect()
            core.t.write_json(STATE, {'state': 'completed', 'owner': owner, 'completed': completed,
                'utc': core.t.utc_now(), 'protocol_sha256': frozen_sha, 'test_evaluated': False})
        except BaseException as error:
            core.t.write_json(STATE, {'state': 'stopped_with_error', 'owner': owner,
                'completed': completed, 'utc': core.t.utc_now(), 'error': str(error),
                'automatic_retry': False, 'protocol_sha256': frozen_sha})
            raise


if __name__ == '__main__':
    main()
