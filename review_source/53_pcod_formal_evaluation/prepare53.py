"""Freeze definitions and run exclusive synthetic checks, without dataset decoding."""
import argparse
import ctypes
from datetime import datetime, timezone
import io
from pathlib import Path
import pickletools
import shutil
import time
import unittest
import zipfile
import pcod_eval as e
import formal53 as f

HERE, ROOT = e.HERE, e.ROOT


def sources():
    files = list(HERE.glob('*.py')) + [HERE / 'README.md']
    files += [ROOT / name for name in (
        '06_experiment_preparation/EXPERIMENT_QUEUE.json', '06_experiment_preparation/METRIC_CONTRACT.md',
        '06_experiment_preparation/boundary_metrics.py', '13_registration_evaluation/protocol.json',
        '15_pcod_training/train_pcod.py', '15_pcod_training/pcod_models.py',
        '13_registration_evaluation/reference/04_multiscale_ablation/code/scripts/extension_dataset.py',
        '13_registration_evaluation/reference/04_multiscale_ablation/code/scripts/pcod_dataset.py',
        '25_pcod_portable_runner/formal_protocol.json', '25_pcod_portable_runner/run_queue.py',
        '25_pcod_portable_runner/portable_core.py', '25_pcod_portable_runner/execution_protocol.py',
        '37_pcod_scientific_review/scientific_review_pending_hardware.json',
        '37_pcod_scientific_review/admission_record.json', '41_rgbp_portable_runner/formal_protocol.json',
        '41_rgbp_portable_runner/run_queue.py', '45_rgbp_common_evaluation/evaluation.py',
        '45_rgbp_common_evaluation/offline_evaluate.py')]
    return {p.relative_to(ROOT).as_posix(): e.sha(p) for p in files}


def definitions():
    formal = f.protocol.validate(f.protocol.read(f.protocol.PROTOCOL))
    archived = e.read(ROOT / '13_registration_evaluation/protocol.json')
    ids = archived['stems']; e.validate_ids(ids)
    if len(ids) != 230 or len(archived['input_files']) != 690 or archived['sigmoid_device'] != 'model device':
        raise ValueError('Archived population definition changed')
    root = Path(formal['configs'][0]['config']['dataset_root']).resolve(strict=True)
    index = {(x['stem'], x['modality']): x for x in archived['input_files']}
    if len(index) != 690:
        raise ValueError('Duplicate archived modality members')
    members = []
    for stem in ids:
        item = {'sample_id': stem}
        for modality in ('rgb', 'dop', 'mask'):
            source = index[stem, modality]
            path = Path(source['path']).resolve(strict=True)
            if not path.is_relative_to(root) or path.stem != stem:
                raise ValueError('Archived source path differs')
            item[modality] = path.relative_to(root).as_posix()
            item[modality + '_sha256'] = source['sha256']
        members.append(item)
    from scripts.pcod_dataset import build_samples
    actual = build_samples(root, 'test')
    if [x.stem for x in actual] != ids:
        raise ValueError('Current directory metadata does not match frozen test order')
    for sample, member in zip(actual, members):
        for modality in ('rgb', 'dop', 'mask'):
            if getattr(sample, modality + '_path').resolve() != (root / member[modality]).resolve():
                raise ValueError('Dataset input membership changed')
    # Inspect only the pickle instruction stream of a completed, hash-bound local artifact.
    # No tensor payload read, unpickling, model creation or unsafe deserialization.
    first = formal['configs'][0]['config']
    best = Path(first['output_dir']) / 'best.pt'
    marker = e.read(Path(first['output_dir']) / 'portable_complete.json')
    if marker['artifact_sha256']['best.pt'] != e.sha(best):
        raise ValueError('Local completed best checksum differs')
    with zipfile.ZipFile(best) as archive:
        entries = [x for x in archive.infolist() if x.filename.endswith('/data.pkl')]
        if len(entries) != 1 or entries[0].file_size > 4 * 1024**2:
            raise ValueError('Unexpected checkpoint metadata format')
        operations = list(pickletools.genops(archive.read(entries[0])))
        globals_used = sorted({arg for op, arg, _ in operations if op.name == 'GLOBAL'})
    return {'stage': 53, 'utc': datetime.now(timezone.utc).isoformat(), 'members': members,
        'source_sha256': sources(), 'grid': e.GRID, 'sigmoid_order': e.ENCODING,
        'input': 'PIL RGB/L bilinear, divide by 255; mask-free export, no augmentation',
        'mask': 'PIL convert L, nearest resize 352, then >127, float32',
        'threshold': '>0.5', 'dense_epsilon': 1e-7, 'mae': 'per-image CPU torch FP32 reduction; equal image mean',
        'boundary': {'primary_tolerance_px': 2, 'sensitivity_tolerances_px': [1, 4],
                     'band_ratio': .02, 'band_width_on_352': 10, 'both_empty_score': 1,
                     'dense_empty_union_iou': 0, 'image_mean_and_pooled_boundary_iou': True},
        'primary': 'PFE vs Direct pooled IoU five seeds; full differences/mean/sample SD/range/exploratory exact sign flip',
        'secondary': 'fusion/FiLM/wide five pairs; DeepLab first three matched PFE pairs; descriptive only',
        'scientific_limits': ['record-groups are not physical scenes', 'seed labels do not imply identical sample order',
                              'images/pixels/conditions do not increase independent training replication',
                              'fixed budget is not optimized hyperparameters or demonstrated convergence'],
        'device_tolerance': {'atol': 1e-5, 'rtol': 1e-4},
        'execution': 'explicit only after all25/all41 checkpoints, four locks, idle, 20GiB disk/8GiB RAM and per-checkpoint parity',
        'checkpoint_metadata_audit': {'run_id': first['run_id'], 'best_sha256': e.sha(best),
            'pickle_globals': globals_used, 'pickle_instructions_only_no_unpickle': True,
            'restricted_compatibility_verified': False,
            'pending': 'Verify a versioned safe adapter for any unsupported local metadata before real inference; no unsafe fallback'},
        'formal_inference_performed': False, 'real_test_pixels_or_predictions_read': False,
        'formal_results': False, 'training_or_waiter_created': False,
        'historical_model_outputs_or_weights_reused': False}


def verify_sources(plan):
    for name, digest in plan['source_sha256'].items():
        if e.sha(ROOT / name) != digest: raise ValueError('Bound source changed: ' + name)


def checks(attempt):
    plan = e.read(HERE / 'plan.json'); verify_sources(plan)
    folder = HERE / attempt; folder.mkdir(exist_ok=False)
    for p in HERE.glob('*.py'): shutil.copy2(p, folder / p.name)
    try:
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000): raise ctypes.WinError()
        f.rgbp_queue.require_resources()
        e.torch.set_num_threads(1); e.torch.set_num_interop_threads(1)
        import cv2
        cv2.setNumThreads(1)
        import test53
        test53.FIXTURES = folder / 'fixtures'; test53.FIXTURES.mkdir()
        registry = list(e.torch.serialization.get_safe_globals())
        log = io.StringIO(); started = time.perf_counter()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test53))
        (folder / 'unittest.log').write_text(log.getvalue(), encoding='utf-8')
        verify_sources(plan)
        if registry != list(e.torch.serialization.get_safe_globals()): raise ValueError('Serialization allowlist changed')
        report = {'passed': result.wasSuccessful(), 'tests': result.testsRun, 'failures': len(result.failures),
            'errors': len(result.errors), 'seconds': time.perf_counter() - started, 'plan_sha256': e.sha(HERE / 'plan.json'),
            'source_sha256': plan['source_sha256'], 'scope': 'Synthetic images and five 352/base4 UNets; no full DeepLab or formal checkpoint loading',
            'GPU': False, 'test_pixels': False, 'external_weights': False, 'allowlist_changed': False,
            'real_formal_load_or_device_export_verified': False, 'utc': datetime.now(timezone.utc).isoformat()}
        e.save_new(folder / 'cpu_test_results.json', report)
        print({k: report[k] for k in ('passed', 'tests', 'errors', 'failures', 'seconds')})
        if not result.wasSuccessful(): raise RuntimeError('CPU checks failed; retain attempt')
    except BaseException as error:
        e.save_new(folder / 'failure.json', {'error': type(error).__name__ + ': ' + str(error)})
        raise


def freeze(attempt):
    plan = e.read(HERE / 'plan.json'); verify_sources(plan)
    report = e.read(HERE / attempt / 'cpu_test_results.json')
    if (report.get('passed') is not True or report['plan_sha256'] != e.sha(HERE / 'plan.json')
            or report['source_sha256'] != plan['source_sha256'] or (HERE / attempt / 'failure.json').exists()):
        raise ValueError('CPU evidence not bound')
    value = dict(plan, cpu_report_sha256=e.sha(HERE / attempt / 'cpu_test_results.json'),
                 cpu_report=(HERE / attempt / 'cpu_test_results.json').relative_to(ROOT).as_posix(),
                 definition_frozen=True, full_inference_path_accepted=False)
    e.save_new(f.CONTRACT, value)
    print({'contract_sha256': e.sha(f.CONTRACT), 'definition_only': True, 'real_checkpoint_adapter_pending': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'checks', 'freeze'))
    parser.add_argument('--attempt', default='cpu_attempt_v1'); args = parser.parse_args()
    if args.action == 'prepare': e.save_new(HERE / 'plan.json', definitions())
    elif args.action == 'checks': checks(args.attempt)
    else: freeze(args.attempt)
