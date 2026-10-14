"""CPU-only protocol, operator, transaction and serial-lock regression checks."""
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

import portable_core as core
import execution_protocol as protocol
import run_queue as queue


@contextmanager
def temporary_directory():
    with tempfile.TemporaryDirectory(dir=core.HERE) as name:
        root = Path(name).resolve()
        assert root.is_relative_to(core.HERE.resolve())
        yield root


class Toy(Dataset):
    def __init__(self):
        self.images = torch.rand(4, 4, 32, 32, generator=torch.Generator().manual_seed(117))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {'image': self.images[index], 'mask': (self.images[index, :1] > .5).float(),
                'stem': str(index)}


def fake_checkpoint(identity):
    scores = [.4, .5, .5]
    return {'identity': identity, 'epoch': 3,
            'history': [{'epoch': i + 1, 'val_iou': score, 'train_images': 3104, 'val_images': 776,
                         'train_loss': .7, 'val_loss': .8, 'val_image_mae': .1}
                        for i, score in enumerate(scores)],
            'best_epoch': 2, 'best_iou': .5, 'model_state': {}, 'optimizer_state': {},
            'best_model_state': {}, 'best_validation_rows': [{'stem': str(i)} for i in range(776)],
            'rng': {'torch_cpu': [], 'python': [], 'numpy': []}}


def fake_hardware(methods):
    return {'passed': True, 'protocol_sha256': 'frozen', 'formal_experiments_completed': 0,
            'test_evaluated': False,
            'backend_parity': [{'method': m, 'atol': 1e-5, 'rtol': 1e-4} for m in methods],
            'gpu_resume': [{'method': m, 'passed': True, 'atol': 1e-6, 'rtol': 1e-4} for m in methods],
            'updates': [{'method': m, 'optimizer_steps': 16, 'finite': True,
                         'dolp_input_gradient_l1_after_updates_cpu_check': .1} for m in methods]}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matrix_and_disabled_state(self):
        configs = core.configurations()
        self.assertEqual(len(configs), 28)
        self.assertEqual(Counter(c['method'] for c in configs),
                         {m: 3 if m == core.t.STANDARD_METHOD else 5 for m in core.candidate.METHODS})
        self.assertTrue(all(c['epochs'] == 50 and c['batch_size'] == 4 and not c['training_authorized']
                            and not c['protocol_frozen'] and c['candidate_only'] for c in configs))
        self.assertEqual(len({c['output_dir'] for c in configs}), 28)

    def test_complete_config_identity_rejects_changes(self):
        cfg = core.configurations(True)[0]
        for key, value in [('seed', 17), ('epochs', 10), ('candidate_only', True),
                           ('unet_resampling', 'native'), ('initialization', 'old_checkpoint'),
                           ('output_dir', str(core.HERE)), ('batch_size', 2)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                core.identity(dict(cfg, **{key: value}))
        with self.assertRaises(ValueError):
            core.identity(core.configurations(False)[0])

    def test_disabled_protocol_never_checks_device_gate(self):
        with patch.object(protocol, 'hardware_evidence') as hardware:
            with self.assertRaises(ValueError):
                protocol.validate({'execution_enabled': False, 'formal_protocol_frozen': False})
            hardware.assert_not_called()

    def test_missing_formal_release_never_calls_device_or_locks(self):
        with patch.object(protocol, 'read', side_effect=FileNotFoundError('formal protocol missing')):
            with patch.object(core.t, 'choose_device') as device, patch.object(queue, 'byte_lock') as lock:
                with self.assertRaises(FileNotFoundError):
                    queue.main()
                device.assert_not_called()
                lock.assert_not_called()

    def test_missing_supplemental_hardware_cannot_pass(self):
        with temporary_directory() as root, patch.object(core.acceptance, 'REPORT', root / 'missing.json'):
            with self.assertRaisesRegex(ValueError, 'pending'):
                protocol.hardware_evidence()

    def test_failed_freeze_creates_no_release(self):
        with temporary_directory() as root, patch.object(protocol, 'PROTOCOL', root / 'formal_protocol.json'):
            with patch.object(protocol, 'hardware_evidence', side_effect=ValueError('pending')):
                with self.assertRaises(ValueError):
                    protocol.freeze()
            self.assertFalse(protocol.PROTOCOL.exists())

    def test_full_hardware_membership_steps_and_tolerances(self):
        methods = core.acceptance.METHODS
        report = fake_hardware(methods)
        state = {'state': 'completed', 'protocol_sha256': 'frozen'}
        protocol.hardware_report(report, state, methods, 'frozen')
        for key in ('backend_parity', 'updates', 'gpu_resume'):
            changed = copy.deepcopy(report)
            changed[key].pop()
            with self.assertRaises(ValueError):
                protocol.hardware_report(changed, state, methods, 'frozen')
        for key, field, value in [('updates', 'optimizer_steps', 8), ('gpu_resume', 'atol', .1),
                                  ('backend_parity', 'rtol', .1), ('updates', 'finite', False)]:
            changed = copy.deepcopy(report)
            changed[key][0][field] = value
            with self.assertRaises(ValueError):
                protocol.hardware_report(changed, state, methods, 'frozen')

    def test_review_not_implicitly_approved(self):
        with patch.object(protocol, 'read', return_value={'decision': 'pending'}):
            with self.assertRaises(ValueError):
                protocol.review_evidence()

    def test_frozen_scope_and_matrix_cannot_change(self):
        expected = {'execution_enabled': True, 'formal_protocol_frozen': True,
                    'rules': dict(protocol.RULES), 'configs': ['synthetic28'],
                    'hardware_evidence': {'h': 'sha'}, 'review_evidence': {'r': 'sha'}}
        base = {k: v for k, v in expected.items() if k not in ('hardware_evidence', 'review_evidence')}
        with patch.object(protocol, 'specification', return_value=base), \
             patch.object(protocol, 'hardware_evidence', return_value={'h': 'sha'}), \
             patch.object(protocol, 'review_evidence', return_value={'r': 'sha'}):
            protocol.validate(expected)
            changed = copy.deepcopy(expected)
            changed['rules']['test_evaluation'] = True
            with self.assertRaises(ValueError):
                protocol.validate(changed)

    def test_corrected_cpu_forward_gradient_and_state_layout(self):
        for method in core.candidate.METHODS:
            if method == core.t.STANDARD_METHOD:
                continue
            with self.subTest(method=method):
                original = core.backend.ORIGINAL_BUILD(method, base_channels=4, seed=19).eval()
                with core.installed_runtime():
                    corrected = core.t.build_model(method, base_channels=4, seed=19).eval()
                    self.assertEqual(list(original.state_dict()), list(corrected.state_dict()))
                    for key, value in original.state_dict().items():
                        self.assertTrue(torch.equal(value, corrected.state_dict()[key]))
                    x = torch.rand(2, 4, 64, 64, generator=torch.Generator().manual_seed(42)).requires_grad_(True)
                    y = x.detach().clone().requires_grad_(True)
                    a, b = original(x), corrected(y)
                    a.sum().backward()
                    b.sum().backward()
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-4)

    def test_context_restores_hooks_even_on_failure(self):
        original = (core.t.build_model, core.t.validate_config, core.backend.resize.bilinear_half_pixel)
        with self.assertRaises(RuntimeError):
            with core.installed_runtime():
                self.assertIs(core.t.build_model, core.backend.corrected_model)
                raise RuntimeError('synthetic')
        self.assertEqual(original, (core.t.build_model, core.t.validate_config,
                                    core.backend.resize.bilinear_half_pixel))

    def test_checkpoint_selection_contiguity_coverage_and_foreign_identity(self):
        cfg = core.configurations(True)[0]
        checkpoint = fake_checkpoint({'portable': True})
        self.assertEqual(core.validate_checkpoint(checkpoint, cfg, {'portable': True}), 3)
        for field, value in [('identity', {'historical': True}), ('best_epoch', 3), ('best_iou', .6),
                             ('epoch', 51), ('rng', {}), ('best_validation_rows', [])]:
            with self.assertRaises(ValueError):
                core.validate_checkpoint(dict(checkpoint, **{field: value}), cfg, {'portable': True})
        for key, value in [('epoch', 99), ('train_images', 3103), ('val_loss', float('nan'))]:
            changed = copy.deepcopy(checkpoint)
            changed['history'][0][key] = value
            with self.assertRaises(ValueError):
                core.validate_checkpoint(changed, cfg, {'portable': True})

    def test_nonempty_without_checkpoint_rejected(self):
        with temporary_directory() as root:
            cfg = dict(core.configurations(True)[0], output_dir=str(root / 'run'))
            self.assertFalse(core.inspect_run(cfg, {})['resume'])
            core.t.write_json(Path(cfg['output_dir']) / 'status.json', {'state': 'running', 'pid': 123})
            with self.assertRaises(ValueError):
                core.inspect_run(cfg, {})

    def test_cpu_resume_model_optimizer_rng_and_metric_replay(self):
        for method in ('direct_reference', 'pfe_reference', 'learned_fusion_reference', 'global_film_style'):
            with self.subTest(method=method), temporary_directory() as root:
                cfg = {'run_id': 'synthetic_only', 'purpose': 'hardware_smoke', 'method': method,
                       'seed': 42, 'base_channels': 4, 'batch_size': 2, 'epochs': 2,
                       'learning_rate': 1e-4, 'weight_decay': 1e-4, 'horizontal_flip_probability': 0,
                       'device': 'cpu', 'sample_limits': {'train': 4, 'validation': 4}}
                a, b = (dict(cfg, output_dir=str(root / name)) for name in ('full', 'resume'))
                identity = {'synthetic_cpu_test': True, 'resampling': core.backend.SEMANTICS}
                with core.installed_runtime(), redirect_stdout(io.StringIO()):
                    core.t.fit(a, Toy(), Toy(), identity)
                    core.t.fit(b, Toy(), Toy(), identity, stop_after_epoch=1)
                    core.t.fit(b, Toy(), Toy(), identity, resume=True)
                first, second = (torch.load(root / name / 'latest.pt', map_location='cpu', weights_only=False)
                                 for name in ('full', 'resume'))
                self.assertEqual(core.validate_checkpoint(second, b, identity), 2)
                for key in ('model_state', 'optimizer_state'):
                    self.assertEqual(core.backend.h.compare_tree(first[key], second[key], {'atol': 0, 'rtol': 0}), 0)
                self.assertTrue(torch.equal(first['rng']['torch_cpu'], second['rng']['torch_cpu']))
                self.assertEqual(first['rng']['python'], second['rng']['python'])
                for x, y in zip(first['rng']['numpy'], second['rng']['numpy']):
                    np.testing.assert_equal(x, y)
                for x, y in zip(first['history'], second['history']):
                    self.assertEqual({k: v for k, v in x.items() if not k.endswith('seconds')},
                                     {k: v for k, v in y.items() if not k.endswith('seconds')})

    def test_same_queue_lock_cannot_be_owned_twice(self):
        with temporary_directory() as root:
            path = root / 'serial.lock'
            with queue.byte_lock(path, create=True):
                with self.assertRaises(OSError):
                    with queue.byte_lock(path):
                        self.fail('Second lock unexpectedly acquired')
            with queue.byte_lock(path):
                pass

    def test_completed_artifact_tamper_not_silently_skipped(self):
        with temporary_directory() as root:
            cfg = dict(core.configurations(True)[0], output_dir=str(root))
            for name in ('latest.pt', 'best.pt', 'history.csv', 'best_validation_per_image.json'):
                (root / name).write_text('synthetic fixture', encoding='utf-8')
            core.t.write_json(root / 'status.json', {'state': 'completed', 'completed_epoch': 50,
                'test_evaluated': False, 'checkpoint_sha256': core.t.sha(root / 'latest.pt')})
            latest = {'best_epoch': 12, 'best_iou': .5, 'best_model_state': {'a': torch.tensor([1.])}}
            best = {'identity': {'synthetic': True}, 'epoch': 12, 'validation_iou': .5,
                    'model_state': {'a': torch.tensor([1.])}}
            with patch.object(core, 'inspect_run', return_value={'resume': True, 'epoch': 50}), \
                 patch.object(core.backend.torch, 'load', side_effect=[latest, best]):
                queue.mark_completed(cfg, {'synthetic': True}, 'frozen')
                self.assertTrue(queue.completed_run(cfg, {'synthetic': True}, 'frozen'))
                (root / 'best.pt').write_text('tampered fixture', encoding='utf-8')
                with self.assertRaises(ValueError):
                    queue.completed_run(cfg, {'synthetic': True}, 'frozen')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    core.t.write_json(protocol.TESTS, {'tests': result.testsRun, 'passed': result.wasSuccessful(),
        'failures': len(result.failures), 'errors': len(result.errors), 'gpu_started': False,
        'test_images_read': False, 'checked_utc': core.t.utc_now(),
        'source_sha256': {str(p.relative_to(core.ROOT)): core.t.sha(p)
                         for p in (*core.RUNTIME_FILES, Path(__file__))}})
    raise SystemExit(0 if result.wasSuccessful() else 1)
