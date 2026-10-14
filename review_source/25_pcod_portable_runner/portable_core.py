"""Corrected PCOD operator and strict checkpoint identities; no device CLI."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / '23_pcod_common_protocol'))
import prepare_common as candidate
sys.path.insert(0, str(ROOT / '24_pcod_reference_acceptance'))
import check_references as acceptance

backend = acceptance.corrected
t = backend.t
RUNTIME_FILES = (Path(__file__), HERE / 'execution_protocol.py', HERE / 'run_queue.py')


def configurations(enabled=False):
    result = candidate.matrix()
    for cfg in result:
        cfg['training_authorized'] = enabled
        cfg['protocol_frozen'] = enabled
        cfg['candidate_only'] = not enabled
        cfg['execution_engine'] = 'pcod_portable25_v1'
    return result


def identity(cfg):
    expected = {(c['method'], c['seed']): c for c in configurations(True)}
    if cfg != expected.get((cfg.get('method'), cfg.get('seed'))):
        raise ValueError('Configuration differs from the complete portable main matrix')
    result = backend.ORIGINAL_VALIDATE(cfg)
    sources = (*RUNTIME_FILES, *backend.SOURCES, Path(candidate.__file__),
               ROOT / '15_pcod_training/prepare_configs.py')
    result['source_sha256'].update({str(p.relative_to(ROOT)): t.sha(p) for p in sources})
    result['resampling_semantics'] = backend.SEMANTICS
    result['execution_engine'] = 'pcod_portable25_v1'
    return result


@contextmanager
def installed_runtime():
    old_build, old_validate = t.build_model, t.validate_config
    old_resize = backend.resize.bilinear_half_pixel
    backend.resize.bilinear_half_pixel = backend.factor2.bilinear_half_pixel
    t.build_model, t.validate_config = backend.corrected_model, identity
    try:
        yield
    finally:
        t.build_model, t.validate_config = old_build, old_validate
        backend.resize.bilinear_half_pixel = old_resize


def validate_checkpoint(checkpoint, cfg, expected_identity):
    if checkpoint.get('identity') != expected_identity:
        raise ValueError('Historical, changed-source or foreign checkpoint rejected before device access')
    epoch = checkpoint.get('epoch')
    if type(epoch) is not int or not 1 <= epoch <= cfg['epochs']:
        raise ValueError('Invalid checkpoint epoch')
    history = checkpoint.get('history', [])
    if [row.get('epoch') for row in history] != list(range(1, epoch + 1)):
        raise ValueError('Checkpoint history is not contiguous')
    counts = {'train_images': cfg['sample_limits']['train'] or 3104,
              'val_images': cfg['sample_limits']['validation'] or 776}
    for row in history:
        if any(row.get(key) != count for key, count in counts.items()):
            raise ValueError('Checkpoint history has incomplete input coverage')
        if not all(math.isfinite(row[key]) for key in ('train_loss', 'val_loss', 'val_image_mae')):
            raise ValueError('Nonfinite checkpoint history')
    scores = [row['val_iou'] for row in history]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise ValueError('Invalid validation selection values')
    best = max(range(len(scores)), key=scores.__getitem__) + 1
    if checkpoint.get('best_epoch') != best or checkpoint.get('best_iou') != scores[best - 1]:
        raise ValueError('Selection must be earliest maximum validation IoU')
    required = ('model_state', 'optimizer_state', 'best_model_state', 'best_validation_rows', 'rng')
    if not all(key in checkpoint for key in required):
        raise ValueError('Incomplete resumable checkpoint')
    if set(checkpoint['rng']) != {'torch_cpu', 'python', 'numpy'}:
        raise ValueError('Incomplete RNG state')
    rows = checkpoint['best_validation_rows']
    if len(rows) != counts['val_images'] or len({r['stem'] for r in rows}) != len(rows):
        raise ValueError('Best-validation records missing or duplicated')
    return epoch


def inspect_run(cfg, expected_identity, loader=None):
    """Use checkpoint as transaction source; CSV/status may lag after interruption."""
    out = Path(cfg['output_dir'])
    if not out.exists() or not any(out.iterdir()):
        return {'resume': False, 'epoch': 0}
    path = out / 'latest.pt'
    if not path.is_file():
        raise ValueError('Nonempty run has no latest checkpoint; explicit investigation required')
    if loader is None:
        loader = lambda p: backend.torch.load(p, map_location='cpu', weights_only=False)
    checkpoint = loader(path)
    epoch = validate_checkpoint(checkpoint, cfg, expected_identity)
    for name, expected in (('config.json', cfg), ('identity.json', expected_identity)):
        saved = out / name
        if not saved.is_file() or json.loads(saved.read_text()) != expected:
            raise ValueError(f'Run metadata differs: {name}')
    return {'resume': True, 'epoch': epoch}


def execute(cfg, expected_identity):
    if identity(cfg) != expected_identity:
        raise ValueError('Source, environment or input identity changed')
    state = inspect_run(cfg, expected_identity)
    with installed_runtime():
        return t.run_config(cfg, resume=state['resume'])
