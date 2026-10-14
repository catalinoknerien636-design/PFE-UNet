"""RGBP formal identities and local checkpoint transactions; no device CLI."""
from contextlib import contextmanager
import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / '40_rgbp_formal_preparation'))
import formal_contract as candidate

t, backend = candidate.t, candidate.backend
read = candidate.read
RUNTIME_FILES = (Path(__file__), HERE / 'rgbp_protocol.py', HERE / 'run_queue.py',
                 ROOT / '24_pcod_reference_acceptance/process_guard.py')
ARTIFACTS = ('config.json', 'identity.json', 'latest.pt', 'best.pt', 'history.csv',
             'best_validation_per_image.json', 'status.json')


def configurations(enabled=False):
    result = candidate.candidates()
    for cfg in result:
        cfg.update(training_authorized=enabled, protocol_frozen=enabled,
                   candidate_only=not enabled, execution_engine='rgbp_portable41_v1')
        cfg['run_id'] = cfg['run_id'].replace('rgbp_formal_v1_', 'rgbp_formal_v1_half_pixel_', 1)
        cfg['output_dir'] = str(HERE / 'runs' / cfg['run_id'])
    return result


def verify_preparation():
    report = read(candidate.HERE / 'cpu_verification.json')
    if not report['passed'] or report['tests'] != 27 or report['gpu_started']:
        raise ValueError('Candidate CPU evidence missing')
    for name, digest in report['source_sha256'].items():
        if t.sha(ROOT / name) != digest:
            raise ValueError('Candidate test source changed')
    for name, key in (('input_contract.json','input_contract_sha256'),
                      ('disabled_budget50_candidates.json','proposal_sha256')):
        if t.sha(candidate.HERE / name) != report[key]:
            raise ValueError('Candidate release changed')
    proposal = read(candidate.HERE / 'disabled_budget50_candidates.json')
    if any(proposal[k] for k in ('execution_enabled','formal_protocol_frozen','scientific_admission')):
        raise ValueError('Original candidate release must stay disabled')
    configs = [r['config'] for r in proposal['configs']]
    if candidate.validate_matrix(configs) != [r['candidate_identity'] for r in proposal['configs']]:
        raise ValueError('Candidate identities changed')
    return {str((candidate.HERE / n).relative_to(ROOT)):t.sha(candidate.HERE / n)
            for n in ('input_contract.json','disabled_budget50_candidates.json','cpu_verification.json')}


def identity(cfg):
    expected = {(c['protocol_variant'],c['method'],c['seed']):c for c in configurations(True)}
    key = (cfg.get('protocol_variant'),cfg.get('method'),cfg.get('seed'))
    if cfg != expected.get(key):
        raise ValueError('Not an exact member of the admitted full RGBP design')
    originals = {(c['protocol_variant'],c['method'],c['seed']):c for c in candidate.candidates()}
    original = candidate.candidate_identity(originals[key])
    bound = read(candidate.HERE / 'disabled_budget50_candidates.json')['configs']
    matching = [r for r in bound if (r['config']['protocol_variant'],r['config']['method'],r['config']['seed']) == key]
    if len(matching) != 1 or matching[0]['candidate_identity'] != original:
        raise ValueError('Underlying prepared candidate changed')
    return {'config_sha256':t.object_sha(cfg), 'prepared_candidate_identity':original,
            'source_sha256':{str(p.relative_to(ROOT)):t.sha(p) for p in RUNTIME_FILES},
            'resampling_semantics':backend.SEMANTICS, 'execution_engine':'rgbp_portable41_v1'}


@contextmanager
def installed_runtime():
    old_build, old_validate = t.build_model, t.validate_config
    old_resize = backend.resize.bilinear_half_pixel
    backend.resize.bilinear_half_pixel = backend.fast.bilinear_half_pixel
    t.build_model, t.validate_config = backend.corrected_model, identity
    try:
        yield
    finally:
        t.build_model, t.validate_config = old_build, old_validate
        backend.resize.bilinear_half_pixel = old_resize


def finite_tree(value):
    if isinstance(value, t.torch.Tensor):
        if value.is_floating_point() and not t.torch.isfinite(value).all().item():
            raise ValueError('Nonfinite model or optimizer tensor')
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite_tree(item)


def validation_ids(cfg):
    return [r['sample_id'] for r in candidate.csv_rows(cfg['validation_manifest'])]


def load_checkpoint(cfg, expected_identity, ids=None):
    out = Path(cfg['output_dir']).resolve()
    if not out.is_relative_to(HERE.resolve()) or out == HERE.resolve():
        raise ValueError('Run output outside the independent formal directory')
    for name, value in (('config.json',cfg), ('identity.json',expected_identity)):
        if not (out / name).is_file() or read(out / name) != value:
            raise ValueError('Foreign sidecar rejected before checkpoint deserialization')
    # Only this project's own local checkpoint; never an external pretrained file.
    checkpoint = t.torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    candidate.validate_checkpoint(checkpoint, cfg, expected_identity,
                                  validation_ids(cfg) if ids is None else ids)
    for key in ('model_state','best_model_state','optimizer_state'):
        finite_tree(checkpoint[key])
    return checkpoint


def inspect_run(cfg, expected_identity, ids=None):
    out = Path(cfg['output_dir']).resolve()
    if not out.is_relative_to(HERE.resolve()) or out == HERE.resolve():
        raise ValueError('Invalid output boundary')
    if not out.exists() or not any(out.iterdir()):
        return {'resume':False,'epoch':0}
    checkpoint = load_checkpoint(cfg, expected_identity, ids)
    return {'resume':True,'epoch':checkpoint['epoch']}


def verify_exports(cfg, expected_identity, ids=None):
    out = Path(cfg['output_dir'])
    latest = load_checkpoint(cfg, expected_identity, ids)
    if latest['epoch'] != cfg['epochs']:
        raise ValueError('Incomplete checkpoint budget')
    status = read(out / 'status.json')
    if (status.get('state') != 'completed' or status.get('completed_epoch') != cfg['epochs']
            or status.get('run_id') != cfg['run_id'] or status.get('purpose') != cfg['purpose']
            or status.get('test_evaluated') is not False
            or status.get('checkpoint_sha256') != t.sha(out / 'latest.pt')
            or status.get('best_epoch') != latest['best_epoch']
            or status.get('best_validation_iou') != latest['best_iou']):
        raise ValueError('Completion status inconsistent with checkpoint')
    best = t.torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    if (best.get('identity') != expected_identity or best.get('epoch') != latest['best_epoch']
            or best.get('validation_iou') != latest['best_iou']):
        raise ValueError('Best checkpoint provenance changed')
    backend.compare_tree(best['model_state'], latest['best_model_state'], {'atol':0,'rtol':0})
    if read(out / 'best_validation_per_image.json') != latest['best_validation_rows']:
        raise ValueError('Best-image export differs from transaction source')
    with (out / 'history.csv').open(encoding='utf-8', newline='') as stream:
        history = list(csv.DictReader(stream))
    if history != [{k:str(v) for k,v in row.items()} for row in latest['history']]:
        raise ValueError('History export differs from transaction source')
    return {name:t.sha(out / name) for name in ARTIFACTS}


def completed_run(cfg, expected_identity, protocol_sha, ids=None):
    path = Path(cfg['output_dir']) / 'portable_complete.json'
    if not path.exists():
        return False
    record = read(path)
    if (record.get('run_id') != cfg['run_id'] or record.get('identity') != expected_identity
            or record.get('protocol_sha256') != protocol_sha or record.get('test_evaluated') is not False):
        raise ValueError('Completion marker provenance differs')
    if record.get('artifact_sha256') != verify_exports(cfg, expected_identity, ids):
        raise ValueError('Completed artifacts changed')
    return True


def mark_completed(cfg, expected_identity, protocol_sha, ids=None):
    hashes = verify_exports(cfg, expected_identity, ids)
    candidate.save_new(Path(cfg['output_dir']) / 'portable_complete.json', {
        'run_id':cfg['run_id'], 'identity':expected_identity, 'protocol_sha256':protocol_sha,
        'completed_utc':t.utc_now(), 'test_evaluated':False, 'artifact_sha256':hashes})


def execute(cfg, expected_identity, allow_resume=False):
    if identity(cfg) != expected_identity:
        raise ValueError('Formal identity changed')
    state = inspect_run(cfg, expected_identity)
    if state['resume'] and not allow_resume:
        raise ValueError('Existing checkpoint requires explicit inspected --resume')
    with installed_runtime():
        return t.run_config(cfg, resume=state['resume'])
