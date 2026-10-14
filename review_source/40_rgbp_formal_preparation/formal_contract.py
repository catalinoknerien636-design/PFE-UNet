"""Disabled RGBP candidates and checkpoint contracts; no training entry point."""
from contextlib import contextmanager
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / '21_portable_rgbp_development'))
import train_portable as development
backend, t = development.backend, development.t
METHODS = tuple(development.METHODS)
SEEDS = (42, 2026, 3407)
VARIANTS = ('official_dateblock_reference', 'association_quarantine_sensitivity_v2')
COUNTS = {VARIANTS[0]: {'train': 2589, 'validation': 617},
          VARIANTS[1]: {'train': 1857, 'validation': 456}}
SOURCES = (Path(__file__), HERE / 'prepare_candidates.py', HERE / 'test_formal40.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate_members(official, projected):
    if set(official) != set(projected):
        raise ValueError('Projection must cover every original ID exactly')
    for sid, part in official.items():
        if projected[sid] not in (part, 'quarantine'):
            raise ValueError('Partition reassignment is forbidden')
    if {s for s,p in official.items() if p == 'test'} != {s for s,p in projected.items() if p == 'test'}:
        raise ValueError('Full test set must remain unchanged')


def candidates():
    inputs = read(HERE / 'input_contract.json')
    base = read(ROOT / '21_portable_rgbp_development/configs/rgbp_dev_v3_half_pixel_direct_seed17.json')
    result = []
    for variant in VARIANTS:
        for method in METHODS:
            for seed in SEEDS:
                cfg = dict(base)
                run_id = f'rgbp_formal_v1_{variant}_{method}_seed{seed}'
                cfg.update(run_id=run_id, purpose='formal', method=method, seed=seed,
                    epochs=50, protocol_variant=variant, candidate_only=True,
                    training_authorized=False, protocol_frozen=False,
                    initialization='from_scratch_no_development_or_historical_warm_start',
                    output_dir=str(HERE / 'runs' / run_id),
                    manifest_sha256={p:inputs['manifests'][variant][p]['sha256'] for p in ('train','validation')},
                    sample_counts=COUNTS[variant], execution_engine='not_implemented_no_training_entry')
                for part in ('train','validation'):
                    cfg[part + '_manifest'] = inputs['manifests'][variant][part]['path']
                result.append(cfg)
    return result


def candidate_identity(cfg):
    expected = {(c['protocol_variant'],c['method'],c['seed']): c for c in candidates()}
    if cfg != expected.get((cfg.get('protocol_variant'),cfg.get('method'),cfg.get('seed'))):
        raise ValueError('Candidate differs from complete disabled design')
    inputs = read(HERE / 'input_contract.json')
    for name, digest in inputs['source_sha256'].items():
        if t.sha(ROOT / name) != digest:
            raise ValueError('Input evidence changed: ' + name)
    parts = {}
    for part in ('train','validation'):
        path = Path(cfg[part + '_manifest'])
        if t.sha(path) != cfg['manifest_sha256'][part]:
            raise ValueError('Candidate manifest changed')
        rows = csv_rows(path)
        if len(rows) != cfg['sample_counts'][part] or any(r['split'] != 'train' for r in rows):
            raise ValueError('Input count or training-pool origin differs')
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate input ID')
        parts[part] = rows
    if ({r['sample_id'] for r in parts['train']} & {r['sample_id'] for r in parts['validation']}
            or {r['sample_id'][:8] for r in parts['train']} & {r['sample_id'][:8] for r in parts['validation']}):
        raise ValueError('Date-block split overlap')
    test_manifest = inputs['manifests'][cfg['protocol_variant']]['test']
    if t.sha(test_manifest['path']) != test_manifest['sha256']:
        raise ValueError('Retained test manifest changed')
    source_paths = [*SOURCES, Path(development.__file__), ROOT / '12_training/train_rgbp.py',
        ROOT / '06_experiment_preparation/rgbp_glass_dataset.py',
        ROOT / '11_rgbp_model_preflight/model_adapter.py', *backend.SOURCES,
        *sorted((ROOT / '11_rgbp_model_preflight/reference_models').glob('*.py'))]
    return {'candidate_config_sha256': t.object_sha(cfg), 'input_contract_sha256': t.sha(HERE / 'input_contract.json'),
        'manifest_sha256': cfg['manifest_sha256'],
        'source_sha256': {str(p.relative_to(ROOT)):t.sha(p) for p in source_paths},
        'versions': {'torch':t.torch.__version__, 'numpy':t.np.__version__, 'opencv':t.cv2.__version__,
            'python':sys.version, 'torch_directml':importlib.metadata.version('torch-directml')},
        'resampling_semantics':backend.SEMANTICS, 'executable_formal_identity':False}


def validate_matrix(configs):
    if configs != candidates():
        raise ValueError('Full ordered 30-run disabled matrix required')
    return [candidate_identity(cfg) for cfg in configs]


@contextmanager
def corrected_cpu_test_runtime():
    """Install the frozen operator only for bounded CPU synthetic tests."""
    old_build, old_resize = t.build_model, backend.resize.bilinear_half_pixel
    backend.resize.bilinear_half_pixel = backend.fast.bilinear_half_pixel
    t.build_model = backend.corrected_model
    try:
        yield
    finally:
        t.build_model, backend.resize.bilinear_half_pixel = old_build, old_resize


def validate_checkpoint(checkpoint, cfg, expected_identity, validation_ids):
    if checkpoint.get('identity') != expected_identity:
        raise ValueError('Foreign, development or changed-source checkpoint')
    epoch = checkpoint.get('epoch')
    if type(epoch) is not int or not 1 <= epoch <= cfg['epochs']:
        raise ValueError('Checkpoint epoch outside budget')
    history = checkpoint.get('history', [])
    if [r.get('epoch') for r in history] != list(range(1, epoch + 1)):
        raise ValueError('Noncontiguous history')
    for row in history:
        if (row.get('train_images') != cfg['sample_counts']['train']
                or row.get('val_images') != cfg['sample_counts']['validation']):
            raise ValueError('Partial sample coverage')
        for key in ('train_loss','val_loss','val_iou','val_f1','val_image_mae'):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Nonfinite or negative training/validation value')
        if any(row[key] > 1 for key in ('val_iou','val_f1','val_image_mae')):
            raise ValueError('Metric outside its range')
    best = max(history, key=lambda r:r['val_iou'])
    if checkpoint.get('best_epoch') != best['epoch'] or checkpoint.get('best_iou') != best['val_iou']:
        raise ValueError('Selection is not earliest maximum validation IoU')
    for key in ('model_state','best_model_state','optimizer_state','rng','best_validation_rows'):
        if key not in checkpoint or not checkpoint[key]:
            raise ValueError('Missing checkpoint payload')
    if set(checkpoint['rng']) != {'torch_cpu','python','numpy'}:
        raise ValueError('Incomplete RNG state')
    rows = checkpoint['best_validation_rows']
    if len(rows) != len(validation_ids) or {r['stem'] for r in rows} != set(validation_ids):
        raise ValueError('Missing, foreign or duplicate validation IDs')
    if len(rows) != cfg['sample_counts']['validation']:
        raise ValueError('Wrong validation population')
    for row in rows:
        counts = [row[k] for k in ('tp','fp','fn','tn')]
        if any(type(v) is not int or v < 0 for v in counts) or sum(counts) != cfg['image_size'] ** 2:
            raise ValueError('Invalid per-image confusion counts')
        if not math.isfinite(row['mae']) or not 0 <= row['mae'] <= 1:
            raise ValueError('Invalid per-image MAE')
    sums = {k:sum(r[k] for r in rows) for k in ('tp','fp','fn','tn')}
    if any(best['val_' + k] != v for k,v in sums.items()):
        raise ValueError('Best-image and selected-history confusion totals differ')
    expected_iou = sums['tp'] / (sums['tp'] + sums['fp'] + sums['fn'] + 1e-7)
    expected_f1 = 2 * sums['tp'] / (2 * sums['tp'] + sums['fp'] + sums['fn'] + 1e-7)
    expected_mae = sum(r['mae'] for r in rows) / len(rows)
    if (not math.isclose(best['val_iou'],expected_iou,abs_tol=1e-12,rel_tol=0)
            or not math.isclose(best['val_f1'],expected_f1,abs_tol=1e-12,rel_tol=0)
            or not math.isclose(best['val_image_mae'],expected_mae,abs_tol=1e-12,rel_tol=0)):
        raise ValueError('Selected validation metric aggregation differs')
    return epoch


def inspect_resume(cfg, expected_identity, validation_ids, loader=None):
    out = Path(cfg['output_dir']).resolve()
    if not out.is_relative_to(HERE.resolve()):
        raise ValueError('Only this stage local run records are accepted')
    if not out.exists() or not any(out.iterdir()):
        return {'resume':False,'epoch':0}
    # Reject foreign metadata before deserializing a known locally produced checkpoint.
    for filename, value in (('config.json',cfg),('identity.json',expected_identity)):
        path = out / filename
        if not path.is_file() or read(path) != value:
            raise ValueError('Run sidecar identity changed')
    path = out / 'latest.pt'
    if not path.is_file():
        raise ValueError('Nonempty output without latest checkpoint')
    if loader is None:
        loader = lambda path: t.torch.load(path,map_location='cpu',weights_only=False)
    checkpoint = loader(path)
    return {'resume':True,'epoch':validate_checkpoint(checkpoint,cfg,expected_identity,validation_ids)}
