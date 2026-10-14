"""Plain NPY/JSON state provider; never invokes pickle or torch.load."""
from collections import OrderedDict
import os
from pathlib import Path
import shutil
import numpy as np
import metadata54 as m

FORMAT = 'pcod54-validation-best-npy-v1'


def owned_path(path):
    path = Path(path).resolve()
    m.need(path.is_relative_to(m.HERE) and path != m.HERE, 'New-stage derived output only')
    return path


def write_state(path, state, metadata, binding):
    path = owned_path(path)
    arrays = [(key, value.detach().cpu().numpy()) for key, value in state.items()]
    schema = [m.store.schema_row(k, list(a.shape), a.dtype.str) for k, a in arrays]
    total = m.store.validate_schema(schema)
    m.need(shutil.disk_usage(path.parent).free >= 20 * 1024**3 + total, 'Keep20GiB disk reserve plus new state')
    path.mkdir(exist_ok=False)
    files = []
    try:
        for i, ((key, value), row) in enumerate(zip(arrays, schema)):
            m.store.validate_array(value, row)
            name = f'{i:05d}.npy'; destination = path / name
            with destination.open('xb') as stream:
                np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            digest = m.sha(destination)
            reread = m.store.read_array(destination, row, digest)
            m.need(np.array_equal(reread, value), 'Tensor roundtrip mismatch')
            files.append({'file': name, 'sha256': digest,
                          'value_bytes_sha256': m.hashlib.sha256(value.tobytes(order='C')).hexdigest()})
            del reread
        manifest = {'format': FORMAT, 'binding': binding, 'metadata': metadata,
                    'schema': schema, 'files': files, 'total_tensor_bytes': total}
        m.write(path / 'manifest.json', manifest)
        m.write(path / 'complete.json', {'format': FORMAT, 'manifest_sha256': m.sha(path / 'manifest.json'),
                                       'binding_sha256': m.store.object_sha(binding)})
        return m.sha(path / 'complete.json')
    except BaseException as error:
        m.write(path / 'failure.json', {'error': type(error).__name__ + ': ' + str(error)})
        raise


def verify_state(path, receipt_sha, binding):
    path = owned_path(path)
    m.need(m.sha(path / 'complete.json') == receipt_sha, 'State completion changed')
    receipt, manifest = m.read(path / 'complete.json'), m.read(path / 'manifest.json')
    m.need(receipt == {'format': FORMAT, 'manifest_sha256': m.sha(path / 'manifest.json'),
                      'binding_sha256': m.store.object_sha(binding)}, 'State manifest or identity changed')
    m.need(manifest['format'] == FORMAT and manifest['binding'] == binding, 'Wrong state source')
    m.need(m.store.validate_schema(manifest['schema']) == manifest['total_tensor_bytes'], 'State schema total')
    m.need(len(manifest['files']) == len(manifest['schema']), 'State file count')
    expected = {'manifest.json', 'complete.json'}
    for i, entry in enumerate(manifest['files']):
        m.need(set(entry) == {'file', 'sha256', 'value_bytes_sha256'} and entry['file'] == f'{i:05d}.npy', 'Canonical tensor entries')
        expected.add(entry['file'])
    m.need({p.name for p in path.iterdir()} == expected, 'Foreign, failed or incomplete state files')
    m.need(all(p.is_file() and not p.is_symlink() and p.resolve().parent == path for p in path.iterdir()), 'No external linked state')
    for row, entry in zip(manifest['schema'], manifest['files']):
        value = m.store.read_array(path / entry['file'], row, entry['sha256'])
        m.need(m.hashlib.sha256(value.tobytes(order='C')).hexdigest() == entry['value_bytes_sha256'], 'State value-byte hash')
    return manifest


def load_state(path, receipt_sha, binding):
    import torch
    manifest = verify_state(path, receipt_sha, binding)
    result = OrderedDict()
    for row, entry in zip(manifest['schema'], manifest['files']):
        array = m.store.read_array(Path(path) / entry['file'], row, entry['sha256'])
        result[row['key']] = torch.from_numpy(array)
    metadata = manifest['metadata']
    if metadata['state_dict_metadata'] is not None:
        result._metadata = metadata['state_dict_metadata']
    return {'identity': metadata['identity'], 'epoch': metadata['epoch'], 'validation_iou': metadata['validation_iou'],
            'model_state': result}


def provider(run_id, expected_identity):
    plan = m.read(m.HERE / 'plan.json')
    for path, digest in plan['source_sha256'].items(): m.need(m.sha(path) == digest, 'Adapter source changed')
    selected = [x for x in plan['checkpoints'] if x['run_id'] == run_id]
    m.need(len(selected) == 1 and selected[0]['identity'] == expected_identity, 'Exact prepared formal run required')
    item = selected[0]
    for name, digest in item['artifacts_sha256'].items():
        m.need(m.sha(Path(item['output_dir']) / name) == digest, 'Original completed-run artifact changed')
    m.need(m.sha(Path(item['output_dir']) / 'portable_complete.json') == item['completion_sha256'], 'Original completion changed')
    folder = m.HERE / 'conversions' / run_id
    m.need(not (folder / 'failure.json').exists(), 'Failed conversion rejected')
    completion = m.read(folder / 'conversion_complete.json')
    report = m.read(folder / 'conversion_report.json')
    m.need(completion == {'report_sha256': m.sha(folder / 'conversion_report.json'), 'plan_sha256': m.sha(m.HERE / 'plan.json')},
           'Unbound conversion proof')
    binding = {'run_id': run_id, 'original_best_sha256': item['artifacts_sha256']['best.pt'],
               'original_completion_sha256': item['completion_sha256'], 'formal_protocol_sha256': plan['formal_protocol_sha256']}
    m.need(report['binding'] == binding and report['passed'] is True, 'Conversion source differs')
    m.need(m.sha(folder / 'normalized_best.pt') == report['normalized_sha256'], 'Normalized archive changed')
    m.need(m.sha(folder / 'normalization.json') == report['normalization_sha256'], 'Normalization proof changed')
    proof = m.read(folder / 'normalization.json')
    m.need(proof['source_sha256'] == binding['original_best_sha256']
           and proof['normalized_sha256'] == report['normalized_sha256']
           and proof['storage_and_tensor_recipe_bytes_unchanged'] is True,
           'Normalization provenance differs')
    best = load_state(folder / 'state', report['state_receipt_sha256'], binding)
    m.need(best['identity'] == expected_identity and best['epoch'] == item['best_epoch']
           and best['validation_iou'] == item['best_validation_iou'], 'Selected validation metadata differs')
    return best
