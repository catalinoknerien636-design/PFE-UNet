"""Bind the live v3 controller; hardware checks only after full completion."""
import argparse
from contextlib import contextmanager
import json
import msvcrt
import os
from pathlib import Path

import check_references as check
import process_guard as guard

STATE = check.HERE / 'dependency_status.json'


@contextmanager
def locked(path, create=True):
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


def run(pid, creation, protocol_sha):
    if STATE.exists():
        raise FileExistsError('A dependency attempt exists; inspect before any explicit new version')
    owner = guard.process_identity(os.getpid())
    if check.t.sha(check.PROTOCOL) != protocol_sha:
        raise ValueError('Launch protocol hash differs')
    protocol = check.read_json(check.PROTOCOL)
    check.validate_protocol(protocol)
    guard.verify_dependency_command(guard.python_processes(), pid)
    try:
        def update(dependency):
            check.t.write_json(STATE, {'state': 'waiting_on_exact_v3_development_controller',
                                       'owner': owner, 'dependency': dependency,
                                       'protocol_sha256': protocol_sha, 'utc': check.t.utc_now(),
                                       'formal_training_started': False})
        guard.wait_for_exact_process(pid, creation, update)
        if check.t.sha(check.PROTOCOL) != protocol_sha:
            raise ValueError('Protocol changed during wait')
        # The same byte lock prevents this upstream controller from being
        # resumed while the supplemental device checks own the GPU stage.
        with locked(check.ROOT / '21_portable_rgbp_development/queue.lock', create=False):
            check.validate_protocol(protocol)
            check.verify_development(protocol)
            guard.require_idle()
            check.t.write_json(STATE, {'state': 'supplemental_hardware_checks_running',
                                       'owner': owner, 'protocol_sha256': protocol_sha,
                                       'utc': check.t.utc_now(), 'formal_training_started': False})
            check.run(protocol)
        result = check.read_json(check.REPORT)
        if (not result['passed'] or result['protocol_sha256'] != protocol_sha or
                {r['method'] for r in result['gpu_resume']} != set(check.METHODS) or
                not all(r['passed'] for r in result['gpu_resume'])):
            raise ValueError('Incomplete supplemental hardware report')
        check.t.write_json(STATE, {'state': 'completed', 'owner': owner,
                                   'protocol_sha256': protocol_sha, 'utc': check.t.utc_now(),
                                   'formal_training_started': False})
    except BaseException as error:
        check.t.write_json(STATE, {'state': 'stopped_with_error', 'owner': owner,
                                   'error': str(error), 'automatic_retry': False,
                                   'utc': check.t.utc_now(), 'formal_training_started': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dependency-pid', type=int, required=True)
    parser.add_argument('--dependency-creation', type=int, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    with locked(check.HERE / 'dependency.lock'):
        run(args.dependency_pid, args.dependency_creation, args.protocol_sha256)
