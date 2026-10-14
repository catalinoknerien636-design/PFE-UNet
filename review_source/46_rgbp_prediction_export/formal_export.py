"""Explicit one-run inference, gated on completed queues and held shared locks."""
import argparse
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import sys
import torch
import export_core as e

HERE, ROOT = e.HERE, e.ROOT
sys.path.insert(0, str(ROOT / '41_rgbp_portable_runner'))
import rgbp_protocol as protocol
core = protocol.core
queue = e.load_module('rgbp41_queue_for_export46', core.HERE / 'run_queue.py')
CONTRACT = HERE / 'export_contract.json'


def verify_contract():
    contract = e.read(CONTRACT)
    for path, digest in contract['source_sha256'].items():
        if e.sha(ROOT / path) != digest:
            raise ValueError('Bound exporter source changed: ' + path)
    evaluation = e.read(e.offline.HERE / 'evaluation_contract.json')
    e.offline.verify_sources(evaluation)
    if [x['sample_id'] for x in contract['members']] != [x['sample_id'] for x in evaluation['members']]:
        raise ValueError('Export/evaluation population differs')
    return contract, evaluation


def require_finished_queues(formal):
    sys.path.insert(0, str(ROOT / '25_pcod_portable_runner'))
    import execution_protocol as pcod
    pcod_queue = e.load_module('pcod25_queue_for_export46', pcod.HERE / 'run_queue.py')
    for p, module in ((pcod, pcod_queue), (protocol, core)):
        value = formal if p is protocol else p.validate(p.read(p.PROTOCOL))
        state = p.read(p.HERE / 'queue_status.json')
        ids = [x['config']['run_id'] for x in value['configs']]
        if state.get('state') != 'completed' or state.get('completed') != ids:
            raise ValueError('All formal training must finish before test inference')
        for item in value['configs']:
            if not module.completed_run(item['config'], item['identity'], e.sha(p.PROTOCOL)):
                raise ValueError('Full-budget queue completion unverified')
        gc.collect()


def selected_item(formal, run_id):
    items = [x for x in formal['configs'] if x['config']['run_id'] == run_id]
    if len(items) != 1:
        raise ValueError('Only an exact formal RGBP run is eligible')
    return items[0]


def device_parity(cpu, gpu, cfg, device, tolerance):
    channels = 3 if cfg['method'] == 'rgb_only' else 6
    generator = torch.Generator().manual_seed(46027)
    image = torch.rand(2, channels, 352, 352, generator=generator)
    cpu.eval(); gpu.eval()
    with torch.no_grad():
        a, b = cpu(image).float().cpu(), gpu(image.to(device)).float().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite synthetic device parity')
    torch.testing.assert_close(a, b, **tolerance)
    pa, pb = e.probabilities(a, 2, [352, 352]), e.probabilities(b, 2, [352, 352])
    return {'raw_grid': [352, 352], 'samples': 2, 'seed': 46027,
            'logit_max_abs_difference': float((a - b).abs().max()),
            'probability_max_abs_difference': float(abs(pa - pb).max()),
            'tolerance': tolerance, 'passed': True, 'test_pixels_read': False}


def run(run_id):
    contract, _ = verify_contract()
    formal = protocol.validate(protocol.read(protocol.PROTOCOL))
    item = selected_item(formal, run_id)
    cfg, identity = item['config'], item['identity']
    output = Path(cfg['output_dir']) / 'common_grid45_exports'
    attempt = HERE / 'device_checks' / run_id
    if output.exists() or attempt.exists():
        raise FileExistsError('Exclusive attempt; inspect prior output, never auto-retry')
    # No checkpoint, input pixel or device is accessed until locks and idle checks.
    with ExitStack() as locks:
        for path in (core.HERE / 'queue.lock', *queue.SHARED_LOCKS):
            locks.enter_context(queue.byte_lock(path))
        queue.guard.require_idle()
        queue.require_resources()
        require_finished_queues(formal)
        owner = queue.guard.process_identity(os.getpid())
        owner['creation_ticks'] = str(owner['creation_ticks'])
        attempt.mkdir(parents=True, exist_ok=False)
        try:
            best_path = Path(cfg['output_dir']) / 'best.pt'
            marker = Path(cfg['output_dir']) / 'portable_complete.json'
            best_sha, marker_sha = e.sha(best_path), e.sha(marker)
            # completed_run above verifies all local transaction exports. No external weights.
            best = torch.load(best_path, map_location='cpu', weights_only=False)
            if best['identity'] != identity or e.sha(best_path) != best_sha:
                raise ValueError('Selected local checkpoint identity changed')
            cpu = core.backend.ORIGINAL_BUILD(cfg['method'], base_channels=cfg['base_channels'], seed=cfg['seed'])
            cpu.load_state_dict(best['model_state'], strict=True)
            core.backend.compare_tree(cpu.state_dict(), best['model_state'], {'atol': 0, 'rtol': 0})
            with core.installed_runtime():
                gpu = core.t.build_model(cfg['method'], base_channels=cfg['base_channels'], seed=cfg['seed'])
                gpu.load_state_dict(best['model_state'], strict=True)
                core.backend.compare_tree(gpu.state_dict(), best['model_state'], {'atol': 0, 'rtol': 0})
                device, name = core.t.choose_device(cfg)
                gpu.to(device)
                check = device_parity(cpu, gpu, cfg, device, contract['device_parity_tolerance'])
                check.update(owner=owner, device=name, best_checkpoint_sha256=best_sha,
                             export_contract_sha256=e.sha(CONTRACT), utc=core.t.utc_now())
                e.save_new(attempt / 'parity.json', check)
                del cpu, best
                gc.collect()
                dataset = e.BoundInputs(cfg['dataset_root'], contract['members'],
                                        'rgb' if cfg['method'] == 'rgb_only' else 'rgb_dolp', 352)
                loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                    drop_last=False, num_workers=0, generator=torch.Generator().manual_seed(46027))
                provenance = {'run_id': run_id, 'source_stage': 41, 'exporter_stage': 46,
                    'evaluation_contract_sha256': e.sha(e.offline.HERE / 'evaluation_contract.json'),
                    'export_contract_sha256': e.sha(CONTRACT), 'best_checkpoint_sha256': best_sha,
                    'completion_sha256': marker_sha, 'device_check_sha256': e.sha(attempt / 'parity.json'),
                    'owner': owner, 'selected_epoch': check_epoch(cfg), 'utc': core.t.utc_now()}

                def final_check():
                    verify_contract()
                    protocol.validate(protocol.read(protocol.PROTOCOL))
                    if e.sha(best_path) != best_sha or e.sha(marker) != marker_sha:
                        raise ValueError('Training provenance changed during export')

                return e.export_batches(gpu, loader, device, output,
                    [x['sample_id'] for x in contract['members']], [352, 352], provenance, final_check)
        except BaseException as error:
            e.save_new(attempt / 'failure.json', {'error': type(error).__name__ + ': ' + str(error),
                                                'automatic_retry': False, 'owner': owner})
            raise


def check_epoch(cfg):
    return e.read(Path(cfg['output_dir']) / 'status.json')['best_epoch']


def verified_evaluate(bundle_path):
    contract, evaluation = verify_contract()
    bundle = e.verify_receipt(bundle_path, e.sha(CONTRACT), [x['sample_id'] for x in contract['members']])
    cfg = e.offline.admit_stage41(bundle, bundle_path, evaluation)
    check = HERE / 'device_checks' / cfg['run_id'] / 'parity.json'
    if e.sha(check) != bundle.get('device_check_sha256'):
        raise ValueError('Device-to-export evidence differs')
    record = e.read(check)
    if (record.get('passed') is not True or record.get('export_contract_sha256') != e.sha(CONTRACT)
            or record.get('best_checkpoint_sha256') != bundle['best_checkpoint_sha256']
            or record.get('tolerance') != contract['device_parity_tolerance']
            or record.get('raw_grid') != [352, 352] or record.get('samples') != 2
            or record.get('seed') != 46027 or record.get('test_pixels_read') is not False
            or (check.parent / 'failure.json').exists()):
        raise ValueError('Device acceptance unbound')
    return e.offline.evaluate(bundle_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest='action', required=True)
    command.add_parser('export').add_argument('--run-id', required=True)
    command.add_parser('evaluate').add_argument('--bundle', required=True)
    args = parser.parse_args()
    print(run(args.run_id) if args.action == 'export' else verified_evaluate(args.bundle))
