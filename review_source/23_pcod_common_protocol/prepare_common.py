"""Prepare disabled PCOD candidates; no training or GPU entry point exists here."""
import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ORIGIN = WORK / '15_pcod_training'
sys.path.insert(0, str(ORIGIN))
spec = importlib.util.spec_from_file_location('pcod_config_origin', ORIGIN / 'prepare_configs.py')
origin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(origin)
t = origin.t
SEEDS = (42, 2026, 3407, 7819, 12011)
METHODS = ('direct_reference', 'pfe_reference', 'learned_fusion_reference',
           'global_film_style', 'wide_direct', t.STANDARD_METHOD)
RESAMPLING = 'pytorch_half_pixel_bilinear_factor2_v1'


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def expected_config(method, seed):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError('Unregistered method/seed')
    if method == t.STANDARD_METHOD and seed not in SEEDS[:3]:
        raise ValueError('Standard backbone has three registered seeds')
    cfg = origin.base_config(method, seed, 'formal')
    cfg['run_id'] = f'pcod_formal_v2_half_pixel_{method}_seed{seed}'
    cfg['output_dir'] = str(t.HERE / 'runs' / cfg['run_id'])
    cfg['training_authorized'] = False
    cfg['protocol_frozen'] = False
    cfg['unet_resampling'] = RESAMPLING
    cfg['unet_resampling_applicable'] = method != t.STANDARD_METHOD
    cfg['initialization'] = 'from_scratch_no_historical_checkpoint_warm_start'
    cfg['comparison_family'] = 'pcod_corrected_backend_full_reference_v2'
    cfg['candidate_only'] = True
    return cfg


def validate_disabled(cfg):
    if cfg != expected_config(cfg['method'], cfg['seed']):
        raise ValueError('Candidate differs from complete common specification')
    # Validate the old schema on an isolated copy. The on-disk candidate stays
    # disabled; this function never calls a device, a data loader, or fit().
    prospective = copy.deepcopy(cfg)
    prospective['training_authorized'] = True
    prospective['protocol_frozen'] = True
    identity = t.validate_config(prospective)
    sources = [Path(__file__), ORIGIN / 'prepare_configs.py',
               WORK / '20_pcod_backend_diagnostics/resize_candidate.py',
               WORK / '20_pcod_backend_diagnostics/resize_factor2.py']
    return {'candidate_sha256': t.object_sha(cfg),
            'prospective_schema_config_sha256': identity['config_sha256'],
            'source_sha256': {**identity['source_sha256'],
                              **{str(p.relative_to(WORK)): t.sha(p) for p in sources}},
            'versions': identity['versions'], 'manifest_sha256': identity['manifest_sha256'],
            'launch_authorized': False, 'test_evaluated': False}


def matrix():
    return [expected_config(method, seed) for seed in SEEDS for method in METHODS
            if method != t.STANDARD_METHOD or seed in SEEDS[:3]]


def main():
    configs = matrix()
    items = []
    for cfg in configs:
        identity = validate_disabled(cfg)
        path = HERE / 'candidates' / f"{cfg['run_id']}.json"
        write_once(path, cfg)
        items.append({'config': str(path.relative_to(WORK)), 'sha256': t.sha(path),
                      'method': cfg['method'], 'seed': cfg['seed'],
                      'validation': identity})
    hardware = WORK / '20_pcod_backend_diagnostics/corrected_hardware_verification.json'
    verification = json.loads(hardware.read_text())
    if verification['passed'] is not True:
        raise ValueError('Prior corrected hardware verification failed')
    hardware_methods = {row['method'] for row in verification['gpu_resume']}
    report = {
        'purpose': 'Disabled full PCOD common-backend comparison preparation',
        'created_utc': t.utc_now(), 'formal_launch_allowed': False,
        'candidate_count': len(configs), 'counts': dict(Counter(c['method'] for c in configs)),
        'candidates': items,
        'technical_gate': {'prior_report_sha256': t.sha(hardware),
                           'prior_verified_methods': sorted(hardware_methods),
                           'remaining_full_device_acceptance': sorted(set(METHODS) - hardware_methods)},
        'scope': '28 main-comparison runs; registration/scene-disjoint and task-specific runs are separate, not removed',
        'added_reference_repeats_due_to_resampling_correction': 15,
        'historical_runs_reused_as_new_replicates': 0,
        'checkpoint_selection': 'Maximum pooled validation IoU, earliest epoch on exact tie; no test selection',
        'same_seed_limit': 'Historical global Torch RNG sampler retained: model construction consumes RNG differently. '
                           'Seed labels match, but batch order and augmentation trajectories are not asserted identical.',
        'budgets': {'epochs_each': 50, 'batch_size': 4, 'image_size': 352,
                    'learning_rate': .0001, 'weight_decay': .0001},
        'convergence_limit': 'Matched fixed budget, not a claim of fully optimized strongest baselines; inspect all validation curves',
        'statistical_unit': 'Five paired seed labels for primary methods; three matched seeds for standard backbone',
        'remaining_gates': [
            '21 development queue must finish; exact PID and creation-time verification before any GPU phase',
            'Direct and learned-fusion original-tolerance device parity, full-resolution updates and resume acceptance',
            'Review PCOD image-association limitations; existing original_stem groups are not physical scenes',
            'New corrected runner, frozen protocol, explicit hardware identities and checkpoint resume guards',
            'Validate source/config/report hashes, disk budget, and single-GPU controller before enabling formal configs',
        ],
        'do_not': ['Do not run these configs through the old unpatched 15 runner',
                   'Do not only flip the candidate authorization flags',
                   'Do not restart the stopped old 18 continuation chain',
                   'Do not infer training speed or success from hardware smoke tests'],
    }
    write_once(HERE / 'candidate_matrix.json', report)
    print(json.dumps({key: report[key] for key in ('candidate_count', 'counts', 'technical_gate', 'formal_launch_allowed')}, indent=2))


if __name__ == '__main__':
    main()
