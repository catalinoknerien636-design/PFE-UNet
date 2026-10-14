"""Supplemental hardware acceptance for two corrected PCOD references only."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / '20_pcod_backend_diagnostics'))
import check_corrected_hardware as corrected
import torch

h, t = corrected.h, corrected.t
METHODS = ('direct_reference', 'learned_fusion_reference')
PROTOCOL = HERE / 'protocol.json'
REPORT = HERE / 'hardware_verification.json'
STATUS = HERE / 'hardware_status.json'
SOURCES = (Path(__file__), HERE / 'run_after_development.py', HERE / 'process_guard.py')
sys.path.insert(0, str(HERE))
import process_guard as guard


def rgbp_module():
    path = ROOT / '21_portable_rgbp_development/train_portable.py'
    spec = importlib.util.spec_from_file_location('rgbp_development_dependency', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variants(method):
    if method not in METHODS:
        raise ValueError('Unregistered supplemental method')
    configs = h.variants(method)
    for cfg in configs:
        cfg['run_id'] = cfg['run_id'].replace('acceptance_v1', 'acceptance_v3_half_pixel_reference')
        cfg['output_dir'] = str(t.HERE / 'runs' / cfg['run_id'])
        cfg['unet_resampling'] = corrected.SEMANTICS
    return configs


def identity(cfg):
    if cfg not in [item for method in METHODS for item in variants(method)]:
        raise ValueError('Supplemental acceptance config changed')
    result = corrected.corrected_identity(cfg)
    result['source_sha256'].update({str(p.relative_to(ROOT)): t.sha(p) for p in SOURCES})
    return result


def specification():
    previous = json.loads(corrected.PROTOCOL.read_text())
    if previous != corrected.specification():
        raise ValueError('Prior corrected acceptance sources changed')
    report = json.loads(corrected.REPORT.read_text())
    if not report['passed'] or report['protocol_sha256'] != t.sha(corrected.PROTOCOL):
        raise ValueError('Prior corrected acceptance is not valid')
    development = rgbp_module()
    development.validate_queue(json.loads(development.QUEUE.read_text()))
    return {
        'purpose': 'supplemental_reference_hardware_acceptance_only', 'methods': list(METHODS),
        'configs': [{'config': cfg, 'identity': identity(cfg)} for method in METHODS for cfg in variants(method)],
        'source_sha256': {str(p.relative_to(ROOT)): t.sha(p) for p in SOURCES},
        'prior_protocol_sha256': t.sha(corrected.PROTOCOL), 'prior_report_sha256': t.sha(corrected.REPORT),
        'development_queue_sha256': t.sha(development.QUEUE),
        'backend_tolerances': previous['backend_tolerances'],
        'resume_tolerances': previous['resume_tolerances'],
        'test_images_read': False, 'formal_runs_allowed': False,
    }


def validate_protocol(protocol):
    if protocol != specification():
        raise ValueError('Supplemental protocol/config/source identity changed')


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_completed_queue(queue, status, reader=read_json, hasher=t.sha):
    if status.get('state') != 'completed' or status.get('formal_runs') != 0:
        raise ValueError('Development controller has not completed successfully')
    run_ids = []
    for item in queue['configs']:
        if hasher(item['path']) != item['sha256']:
            raise ValueError('Development config changed')
        cfg = reader(item['path'])
        run_ids.append(cfg['run_id'])
        out = Path(cfg['output_dir'])
        record = reader(out / 'status.json')
        if (record.get('state') != 'completed' or record.get('completed_epoch') != cfg['epochs']
                or cfg['epochs'] != 50 or cfg['seed'] != 17):
            raise ValueError('A full development run is incomplete')
        if reader(out / 'identity.json') != item['identity']:
            raise ValueError('Development run identity changed')
        if hasher(out / 'latest.pt') != record['checkpoint_sha256']:
            raise ValueError('Development checkpoint changed')
    if len(run_ids) != 5 or len(set(run_ids)) != 5 or status.get('completed') != run_ids:
        raise ValueError('Incomplete development matrix')


def verify_development(protocol):
    module = rgbp_module()
    if t.sha(module.QUEUE) != protocol['development_queue_sha256']:
        raise ValueError('Upstream queue changed')
    queue = read_json(module.QUEUE)
    module.validate_queue(queue)
    verify_completed_queue(queue, read_json(module.HERE / 'queue_status.json'))


def run(protocol):
    validate_protocol(protocol)
    verify_development(protocol)
    guard.require_idle()
    if REPORT.exists() or STATUS.exists():
        raise FileExistsError('Prior attempt exists; inspect rather than overwrite or retry')
    for method in METHODS:
        for cfg in variants(method):
            if Path(cfg['output_dir']).exists():
                raise FileExistsError(cfg['output_dir'])
    report = {'passed': False, 'protocol_sha256': t.sha(PROTOCOL), 'backend_parity': [],
              'updates': [], 'gpu_resume': [], 'formal_experiments_completed': 0, 'test_evaluated': False}
    old_build, old_validate = t.build_model, t.validate_config
    corrected.resize.bilinear_half_pixel = corrected.factor2.bilinear_half_pixel
    t.build_model, t.validate_config = corrected.corrected_model, identity
    try:
        torch.set_num_threads(1)
        device, name = t.choose_device(variants(METHODS[0])[0])
        report['device'] = name
        for method in METHODS:
            t.write_json(STATUS, {'state': 'running', 'method': method, 'phase': 'backend_parity',
                                 'owner': guard.process_identity(os.getpid()), 'utc': t.utc_now()})
            report['backend_parity'].append(h.backend_parity(method, device, protocol['backend_tolerances']))
        for method in METHODS:
            t.write_json(STATUS, {'state': 'running', 'method': method, 'phase': 'updates_and_resume',
                                 'owner': guard.process_identity(os.getpid()), 'utc': t.utc_now()})
            update, full_cfg, split_cfg = variants(method)
            t.run_config(update)
            report['updates'].append(h.audit_update(update))
            t.run_config(full_cfg)
            t.run_config(split_cfg, stop_after_epoch=1)
            t.run_config(split_cfg, resume=True)
            full = torch.load(Path(full_cfg['output_dir']) / 'latest.pt', map_location='cpu', weights_only=False)
            split = torch.load(Path(split_cfg['output_dir']) / 'latest.pt', map_location='cpu', weights_only=False)
            tol = protocol['resume_tolerances']
            model_difference = h.compare_tree(full['model_state'], split['model_state'], tol)
            optimizer_difference = h.compare_tree(full['optimizer_state'], split['optimizer_state'], tol)
            if not torch.equal(full['rng']['torch_cpu'], split['rng']['torch_cpu']):
                raise AssertionError('Resumed CPU RNG differs')
            report['gpu_resume'].append({'method': method, 'passed': True,
                                          'model_max_abs_difference': model_difference,
                                          'optimizer_max_abs_difference': optimizer_difference, **tol})
            del full, split
            gc.collect()
        report['passed'] = True
        report['completed_utc'] = t.utc_now()
        t.write_json(REPORT, report)
        t.write_json(STATUS, {'state': 'completed', 'protocol_sha256': t.sha(PROTOCOL), 'utc': t.utc_now()})
    except BaseException as error:
        report.update(error=str(error), failed_utc=t.utc_now())
        t.write_json(HERE / 'hardware_failure.json', report)
        t.write_json(STATUS, {'state': 'failed', 'error': str(error), 'automatic_retry': False, 'utc': t.utc_now()})
        raise
    finally:
        t.build_model, t.validate_config = old_build, old_validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'validate'))
    args = parser.parse_args()
    if args.action == 'prepare':
        value = specification()
        with PROTOCOL.open('x') as stream:
            json.dump(value, stream, indent=2)
    else:
        validate_protocol(read_json(PROTOCOL))
    print(json.dumps({'protocol_sha256': t.sha(PROTOCOL), 'gpu_started': False, 'formal_runs': 0}))
