"""One-process serial RGBP runner with no waiting or automatic retries."""
import argparse
from contextlib import contextmanager, ExitStack
import ctypes
from ctypes import wintypes
import gc
import msvcrt
import os
import shutil
import sys
import rgbp_core as core
import rgbp_protocol as protocol

sys.path.insert(0,str(core.ROOT / '24_pcod_reference_acceptance'))
import process_guard as guard
STATE = core.HERE / 'queue_status.json'
SHARED_LOCKS = (core.ROOT / '21_portable_rgbp_development/queue.lock',
                core.ROOT / '24_pcod_reference_acceptance/dependency.lock',
                core.ROOT / '25_pcod_portable_runner/queue.lock')


@contextmanager
def byte_lock(path, create=False):
    with path.open('a+b' if create else 'r+b') as stream:
        if create and stream.seek(0,2) == 0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def available_memory():
    class Memory(ctypes.Structure):
        _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)] + [
            (name,ctypes.c_ulonglong) for name in ('total','available','total_page','available_page',
                                                  'total_virtual','available_virtual','extended')]
    value = Memory();value.length = ctypes.sizeof(value)
    api = ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
    api.argtypes = [ctypes.POINTER(Memory)];api.restype = wintypes.BOOL
    if not api(ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return value.available


def require_resources():
    if shutil.disk_usage(core.ROOT).free < 20 * 1024**3:
        raise RuntimeError('Less than20GiB free disk; inspect before continuing')
    if available_memory() < 8 * 1024**3:
        raise RuntimeError('Less than8GiB available physical RAM; do not lower the guard automatically')


def run(allow_resume=False):
    # Admission precedes even queue-state creation or device selection.
    frozen = protocol.validate(protocol.read(protocol.PROTOCOL))
    frozen_sha = core.t.sha(protocol.PROTOCOL)
    with ExitStack() as locks:
        locks.enter_context(byte_lock(core.HERE / 'queue.lock',create=True))
        for path in SHARED_LOCKS:
            locks.enter_context(byte_lock(path))
        guard.require_idle()
        owner = guard.process_identity(os.getpid())
        completed = []
        try:
            for item in frozen['configs']:
                if core.t.sha(protocol.PROTOCOL) != frozen_sha:
                    raise ValueError('Formal protocol changed during execution')
                protocol.review_evidence()
                protocol.hardware_evidence()
                cfg, expected_identity = item['config'],item['identity']
                if core.identity(cfg) != expected_identity:
                    raise ValueError('Run identity changed')
                require_resources()
                if core.completed_run(cfg,expected_identity,frozen_sha):
                    completed.append(cfg['run_id']);continue
                state = core.inspect_run(cfg,expected_identity)
                if state['resume'] and not allow_resume:
                    raise ValueError('Checkpoint present; inspect then explicitly select --resume')
                core.t.write_json(STATE,{'state':'running','owner':owner,'completed':completed,
                    'current_run':cfg['run_id'],'utc':core.t.utc_now(),
                    'protocol_sha256':frozen_sha,'test_evaluated':False})
                core.execute(cfg,expected_identity,allow_resume=allow_resume)
                core.mark_completed(cfg,expected_identity,frozen_sha)
                completed.append(cfg['run_id']);gc.collect()
            core.t.write_json(STATE,{'state':'completed','owner':owner,'completed':completed,
                'utc':core.t.utc_now(),'protocol_sha256':frozen_sha,'test_evaluated':False})
        except BaseException as error:
            core.t.write_json(STATE,{'state':'stopped_with_error','owner':owner,'completed':completed,
                'utc':core.t.utc_now(),'error':str(error),'automatic_retry':False,
                'protocol_sha256':frozen_sha})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args().resume)
