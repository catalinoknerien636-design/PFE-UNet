"""Metadata-only source/input binding; no model, test pixels, locks or GPU."""
import csv
import json
from pathlib import Path
import export_core as e
run_checks = e.load_module('export46_checks_for_freeze', e.HERE / 'run_checks.py')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))


def main():
    target = e.HERE / 'export_contract.json'
    if target.exists(): raise FileExistsError('Export definition already frozen')
    report = e.read(e.HERE / 'cpu_attempt_v3/results.json')
    if not report['passed'] or len(report['method_checks']) != 5 or report['source_sha256'] != run_checks.sources():
        raise ValueError('Complete source-bound synthetic checks required')
    evaluation = e.read(e.offline.HERE / 'evaluation_contract.json')
    e.offline.verify_sources(evaluation)
    inputs = e.read(e.ROOT / '40_rgbp_formal_preparation/input_contract.json')
    manifest = inputs['manifests']['official_dateblock_reference']['test']
    if e.sha(manifest['path']) != manifest['sha256']: raise ValueError('Test manifest changed')
    rows = csv_rows(manifest['path'])
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in evaluation['members']]:
        raise ValueError('Test membership/order differs')
    integrity_path = e.ROOT / '10_rgbp_glass_preparation/audit/file_integrity.csv'
    integrity = {r['path']: r for r in csv_rows(integrity_path)}
    members = []
    for row in rows:
        member = {'sample_id': row['sample_id']}
        for name in ('image', 'dolp'):
            record = integrity[row[name]]
            if record['split'] != 'test' or record['modality'] != name or record['channels'] != '3':
                raise ValueError('Input integrity metadata mismatch')
            member.update({name: row[name], name + '_sha256': record['sha256']})
        members.append(member)
    sources = run_checks.sources()
    for path in (e.HERE / 'README_ZH.md', e.HERE / 'cpu_attempt_v3/results.json',
                 e.offline.HERE / 'evaluation_contract.json', integrity_path,
                 e.ROOT / '41_rgbp_portable_runner/formal_protocol.json',
                 e.ROOT / '25_pcod_portable_runner/formal_protocol.json', Path(manifest['path'])):
        sources[str(path.relative_to(e.ROOT))] = e.sha(path)
    contract = {'definition_frozen': True, 'inference_started': False, 'device_export_checked': False,
        'external_weights_allowed': False, 'PGSNet_admitted': False, 'raw_grid': [352, 352],
        'sigmoid': 'CPU FP32 after detached FP32 logits, exactly as stage12 Metrics',
        'mask_inputs': False, 'augmentation': 'identity', 'batch_size': 'unchanged per stage41 config',
        'training_gate': 'all 28 stage25 and 30 stage41 runs completed with checkpoint evidence',
        'device_parity_tolerance': {'atol': 1e-5, 'rtol': 1e-4},
        'device_parity': 'each selected checkpoint, two synthetic352 inputs, CPU native vs corrected device; no tolerance relaxation',
        'locks': ['41/queue.lock', '21/queue.lock', '24/dependency.lock', '25/queue.lock'],
        'resources': {'disk_GiB': 20, 'available_physical_RAM_GiB': 8},
        'test_read_before_gates': False, 'automatic_retry': False, 'source_sha256': sources, 'members': members}
    e.save_new(target, contract)
    print(json.dumps({'sha256': e.sha(target), 'members': len(members), 'input_hashes': len(members) * 2,
                      'test_pixels_read': False, 'gpu_started': False}))


if __name__ == '__main__': main()
