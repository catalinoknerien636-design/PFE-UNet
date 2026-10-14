"""Explicit future PCOD inference/evaluation. No waiting, training or auto-retry."""
import argparse
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import sys
import torch
import pcod_eval as e

HERE, ROOT = e.HERE, e.ROOT
CONTRACT = HERE / 'evaluation_contract.json'
sys.path.insert(0, str(ROOT / '25_pcod_portable_runner'))
import execution_protocol as protocol
core = protocol.core
queue = e.load('pcod53_queue25', core.HERE / 'run_queue.py')
sys.path.insert(0, str(ROOT / '41_rgbp_portable_runner'))
import rgbp_protocol
rgbp_queue = e.load('pcod53_queue41', ROOT / '41_rgbp_portable_runner/run_queue.py')


def verify_contract():
    value = e.read(CONTRACT)
    for path, digest in value['source_sha256'].items():
        if e.sha(ROOT / path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    if value['sigmoid_order'] != e.ENCODING or value['grid'] != e.GRID:
        raise ValueError('PCOD evaluation semantics changed')
    return value


def selected(formal, run_id):
    items = [x for x in formal['configs'] if x['config']['run_id'] == run_id]
    if len(items) != 1:
        raise ValueError('Only exact corrected PCOD formal runs admitted')
    return items[0]


def require_finished_metadata():
    values = []
    for p in (protocol, rgbp_protocol):
        formal = p.validate(p.read(p.PROTOCOL))
        state = p.read(p.HERE / 'queue_status.json') if (p.HERE / 'queue_status.json').is_file() else {}
        if (state.get('state') != 'completed' or state.get('completed') !=
                [x['config']['run_id'] for x in formal['configs']] or
                state.get('protocol_sha256') != e.sha(p.PROTOCOL)):
            raise ValueError('Both approved training queues must finish before test inference')
        values.append(formal)
    return values


def verify_all_completed(values):
    for p, module, formal in ((protocol, queue, values[0]), (rgbp_protocol, rgbp_protocol.core, values[1])):
        for item in formal['configs']:
            if not module.completed_run(item['config'], item['identity'], e.sha(p.PROTOCOL)):
                raise ValueError('Full-budget real checkpoint completion not verified')
            gc.collect()


def load_selected_best(path, identity, status, digest):
    if e.sha(path) != digest:
        raise ValueError('Selected checkpoint hash changed before load')
    # Local stage25 artifacts only. Unsupported pickle metadata fails closed;
    # no unsafe fallback, external weight route or added serialization allowlist.
    best = torch.load(path, map_location='cpu', weights_only=True)
    if (set(best) != {'identity', 'epoch', 'model_state', 'validation_iou'}
            or best['identity'] != identity or best['epoch'] != status['best_epoch']
            or best['validation_iou'] != status['best_validation_iou']
            or e.sha(path) != digest):
        raise ValueError('Validation-selected checkpoint identity/epoch differs')
    if not isinstance(best['model_state'], dict) or any(
            not isinstance(x, torch.Tensor) or not torch.isfinite(x).all()
            for x in best['model_state'].values()):
        raise ValueError('Invalid finite state dictionary')
    return best


def parity(cpu, gpu, device, tolerance):
    inputs = torch.rand(2, 4, 352, 352, generator=torch.Generator().manual_seed(53019))
    cpu.eval(); gpu.eval()
    with torch.no_grad():
        a, b = cpu(inputs), gpu(inputs.to(device))
        pa, pb = e.probabilities(a, 2), e.probabilities(b, 2)
    torch.testing.assert_close(a.float().cpu(), b.float().cpu(), **tolerance)
    torch.testing.assert_close(torch.from_numpy(pa), torch.from_numpy(pb), **tolerance)
    return {'passed': True, 'samples': 2, 'seed': 53019, 'grid': e.GRID,
            'tolerance': tolerance, 'test_pixels_read': False, 'sigmoid_order': e.ENCODING,
            'logit_max_abs_difference': float((a.float().cpu() - b.float().cpu()).abs().max()),
            'probability_max_abs_difference': float(abs(pa - pb).max())}


def export(run_id):
    contract = verify_contract()
    formal = protocol.validate(protocol.read(protocol.PROTOCOL))
    item = selected(formal, run_id)
    values = require_finished_metadata()
    if contract['checkpoint_metadata_audit']['restricted_compatibility_verified'] is not True:
        raise RuntimeError('A versioned local-checkpoint metadata adapter must be verified before inference')
    cfg, identity = item['config'], item['identity']
    output = Path(cfg['output_dir']) / 'pcod53_exports'
    attempt = HERE / 'device_checks' / run_id
    if output.exists() or attempt.exists():
        raise FileExistsError('Exclusive attempt; inspect failure before explicit new version')
    with ExitStack() as locks:
        for path in (rgbp_protocol.HERE / 'queue.lock', *rgbp_queue.SHARED_LOCKS):
            locks.enter_context(rgbp_queue.byte_lock(path))
        rgbp_queue.guard.require_idle()
        rgbp_queue.require_resources()
        verify_all_completed(values)
        owner = rgbp_queue.guard.process_identity(os.getpid())
        owner['creation_ticks'] = str(owner['creation_ticks'])
        attempt.mkdir(parents=True, exist_ok=False)
        try:
            out = Path(cfg['output_dir'])
            best_sha, completion_sha = e.sha(out / 'best.pt'), e.sha(out / 'portable_complete.json')
            status = e.read(out / 'status.json')
            best = load_selected_best(out / 'best.pt', identity, status, best_sha)
            cpu = core.backend.ORIGINAL_BUILD(cfg['method'], base_channels=cfg['base_channels'], seed=cfg['seed'])
            cpu.load_state_dict(best['model_state'], strict=True)
            core.backend.h.compare_tree(cpu.state_dict(), best['model_state'], {'atol': 0, 'rtol': 0})
            with core.installed_runtime():
                gpu = core.t.build_model(cfg['method'], base_channels=cfg['base_channels'], seed=cfg['seed'])
                gpu.load_state_dict(best['model_state'], strict=True)
                core.backend.h.compare_tree(gpu.state_dict(), best['model_state'], {'atol': 0, 'rtol': 0})
                device, name = core.t.choose_device(cfg)
                gpu.to(device)
                record = parity(cpu, gpu, device, contract['device_tolerance'])
                record.update(owner=owner, device=name, best_checkpoint_sha256=best_sha,
                              contract_sha256=e.sha(CONTRACT), utc=core.t.utc_now())
                e.save_new(attempt / 'parity.json', record)
                del cpu, best
                gc.collect()
                dataset = e.BoundInputs(cfg['dataset_root'], contract['members'])
                loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False,
                    num_workers=0, generator=torch.Generator().manual_seed(53019))
                provenance = {'run_id': run_id, 'source_stage': 25, 'exporter_stage': 53,
                    'contract_sha256': e.sha(CONTRACT), 'best_checkpoint_sha256': best_sha,
                    'completion_sha256': completion_sha, 'selected_epoch': status['best_epoch'],
                    'device_check_sha256': e.sha(attempt / 'parity.json'), 'owner': owner, 'utc': core.t.utc_now()}
                def final_check():
                    verify_contract()
                    protocol.validate(protocol.read(protocol.PROTOCOL))
                    if e.sha(out / 'best.pt') != best_sha or e.sha(out / 'portable_complete.json') != completion_sha:
                        raise ValueError('Training provenance changed during inference')
                return e.export_batches(gpu, loader, device, output,
                    [m['sample_id'] for m in contract['members']], provenance, final_check)
        except BaseException as error:
            e.save_new(attempt / 'failure.json', {'error': type(error).__name__ + ': ' + str(error),
                                                'automatic_retry': False, 'owner': owner})
            raise


def admitted_bundle(bundle_path, contract):
    values = require_finished_metadata()
    path = Path(bundle_path).resolve(strict=True)
    candidate = e.read(path)
    item = selected(values[0], candidate.get('run_id'))
    cfg = item['config']
    out = Path(cfg['output_dir'])
    if path != (out / 'pcod53_exports/bundle.json').resolve():
        raise ValueError('Bundle must belong to its exact formal run')
    if not queue.completed_run(cfg, item['identity'], e.sha(protocol.PROTOCOL)):
        raise ValueError('Incomplete formal training')
    bundle = e.verify_receipt(path, e.sha(CONTRACT), [m['sample_id'] for m in contract['members']])
    check_path = HERE / 'device_checks' / cfg['run_id'] / 'parity.json'
    if (bundle.get('source_stage') != 25 or bundle.get('exporter_stage') != 53
            or bundle.get('best_checkpoint_sha256') != e.sha(out / 'best.pt')
            or bundle.get('completion_sha256') != e.sha(out / 'portable_complete.json')
            or bundle.get('device_check_sha256') != e.sha(check_path)
            or bundle.get('selected_epoch') != e.read(out / 'status.json')['best_epoch']
            or (check_path.parent / 'failure.json').exists()):
        raise ValueError('Unbound prediction provenance')
    check = e.read(check_path)
    expected = {'passed': True, 'samples': 2, 'seed': 53019, 'grid': e.GRID,
        'tolerance': contract['device_tolerance'], 'test_pixels_read': False, 'sigmoid_order': e.ENCODING,
        'best_checkpoint_sha256': bundle['best_checkpoint_sha256'], 'contract_sha256': e.sha(CONTRACT)}
    if any(check.get(k) != v for k, v in expected.items()) or check.get('owner') != bundle.get('owner'):
        raise ValueError('Checkpoint-specific device acceptance differs')
    return cfg, bundle


def evaluate(bundle_path):
    contract = verify_contract()
    cfg, bundle = admitted_bundle(bundle_path, contract)
    output = HERE / 'evaluations' / cfg['run_id']
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for member, item in zip(contract['members'], bundle['items']):
            probability = e.read_probability(Path(bundle_path).parent / item['path'], item['sha256'], e.GRID)
            mask = e.mask_grid(e.bound_payload(cfg['dataset_root'], member, 'mask'))
            rows.append(e.image_metrics(member['sample_id'], probability, mask))
        if [r['sample_id'] for r in rows] != [m['sample_id'] for m in contract['members']]:
            raise ValueError('Incomplete evaluation population')
        report = {'run_id': cfg['run_id'], 'method': cfg['method'], 'seed': cfg['seed'],
            'all_test': e.aggregate(rows), 'contract_sha256': e.sha(CONTRACT),
            'prediction_bundle_sha256': e.sha(bundle_path), 'best_checkpoint_sha256': bundle['best_checkpoint_sha256'],
            'physical_scene_independence_established': False}
        verify_contract()
        e.save_new(output / 'per_image.json', rows)
        e.save_new(output / 'summary.json', report)
        e.save_new(output / 'complete.json', {name: e.sha(output / name) for name in ('per_image.json', 'summary.json')})
        return report
    except BaseException as error:
        e.save_new(output / 'failure.json', {'error': type(error).__name__ + ': ' + str(error), 'rows': len(rows)})
        raise


def summarize():
    contract = verify_contract()
    formal = require_finished_metadata()[0]
    reports, evidence = [], {}
    for item in formal['configs']:
        cfg = item['config']
        folder = HERE / 'evaluations' / cfg['run_id']
        receipt = e.read(folder / 'complete.json')
        if (folder / 'failure.json').exists() or receipt != {n: e.sha(folder / n) for n in ('per_image.json', 'summary.json')}:
            raise ValueError('Failed/incomplete evaluation')
        report, rows = e.read(folder / 'summary.json'), e.read(folder / 'per_image.json')
        bundle_path = Path(cfg['output_dir']) / 'pcod53_exports/bundle.json'
        admitted_bundle(bundle_path, contract)
        if (any(report.get(k) != cfg[k] for k in ('run_id', 'method', 'seed'))
                or report.get('contract_sha256') != e.sha(CONTRACT)
                or report.get('prediction_bundle_sha256') != e.sha(bundle_path)
                or report.get('best_checkpoint_sha256') != e.sha(Path(cfg['output_dir']) / 'best.pt')
                or [r['sample_id'] for r in rows] != [m['sample_id'] for m in contract['members']]
                or report['all_test'] != e.aggregate(rows)):
            raise ValueError('Metric source or aggregation differs')
        reports.append(report)
        evidence[cfg['run_id']] = e.sha(folder / 'complete.json')
    value = e.statistics(reports)
    value.update(contract_sha256=e.sha(CONTRACT), evaluations=evidence)
    e.save_new(HERE / 'formal_statistics.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('export').add_argument('--run-id', required=True)
    commands.add_parser('evaluate').add_argument('--bundle', required=True)
    commands.add_parser('summarize')
    args = parser.parse_args()
    print(export(args.run_id) if args.action == 'export' else evaluate(args.bundle) if args.action == 'evaluate' else summarize())
