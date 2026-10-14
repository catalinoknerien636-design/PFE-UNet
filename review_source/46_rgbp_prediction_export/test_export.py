"""Synthetic tests only; this module never opens the real RGBP test pixels."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
import export_core as e
import formal_export as f

WORK = e.HERE / 'cpu_attempt_v1' / 'fixtures'
METHOD_RESULTS = []
IDS = ['20990101000001', '20990101000002', '20990101000003']


class Echo(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.path = WORK / self._testMethodName
        self.path.mkdir(parents=True, exist_ok=False)
        self.images = torch.linspace(-3, 3, 3 * 352 * 352).reshape(3, 1, 352, 352).repeat(1, 3, 1, 1)
        self.loader = [{'stem': IDS[:2], 'image': self.images[:2]}, {'stem': IDS[2:], 'image': self.images[2:]}]
        self.provenance = {'export_contract_sha256': 'synthetic-only', 'synthetic_fixture': True}

    def export(self, loader=None, callback=lambda: None):
        return e.export_batches(Echo(), self.loader if loader is None else loader, 'cpu',
                                self.path / 'export', IDS, [352, 352], self.provenance, callback)

    def test_full_float_roundtrip(self):
        bundle = self.export()
        loaded = e.verify_receipt(self.path / 'export/bundle.json', 'synthetic-only', IDS)
        self.assertEqual(bundle, loaded)
        for i, item in enumerate(bundle['items']):
            actual = e.offline.read_probability(self.path / 'export' / item['path'], item['sha256'], [352, 352])
            np.testing.assert_array_equal(actual, torch.sigmoid(self.images[i, 0]).numpy())
        self.assertFalse((self.path / 'export/failure.json').exists())

    def test_dense_metrics_match_validation(self):
        bundle = self.export()
        masks = (self.images[:, :1] > .3).float()
        metrics = f.core.t.Metrics()
        metrics.update(self.images[:2, :1], masks[:2], IDS[:2])
        metrics.update(self.images[2:, :1], masks[2:], IDS[2:])
        rows = []
        for i, item in enumerate(bundle['items']):
            value = e.offline.read_probability(self.path / 'export' / item['path'], item['sha256'], [352, 352])
            rows.append(e.offline.e.image_metrics(IDS[i], value, masks[i, 0].numpy().astype(np.uint8), raw_grid=[352, 352]))
        actual = e.offline.e.aggregate(rows)
        for key, value in metrics.finalize().items():
            self.assertEqual(actual[key], value)

    def test_order_rejected(self):
        self.loader[0]['stem'] = IDS[1::-1]
        with self.assertRaises(ValueError): self.export()

    def test_extra_rejected(self):
        self.loader.append(self.loader[-1])
        with self.assertRaises(ValueError): self.export()

    def test_missing_rejected(self):
        with self.assertRaises(ValueError): self.export(self.loader[:1])
        self.assertFalse((self.path / 'export/export_complete.json').exists())

    def test_mask_in_batch_rejected(self):
        self.loader[0]['mask'] = torch.zeros(2, 1, 352, 352)
        with self.assertRaises(ValueError): self.export()

    def test_nonfinite_input(self):
        self.loader[0]['image'][0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): self.export()

    def test_bad_input_type(self):
        self.loader[0]['image'] = self.loader[0]['image'].double()
        with self.assertRaises(ValueError): self.export()

    def test_final_provenance_change(self):
        def fail(): raise ValueError('Changed source')
        with self.assertRaises(ValueError): self.export(callback=fail)
        self.assertTrue((self.path / 'export/failure.json').is_file())
        self.assertFalse((self.path / 'export/bundle.json').exists())

    def test_exclusive_no_overwrite(self):
        self.export()
        old = e.sha(self.path / 'export/bundle.json')
        with self.assertRaises(FileExistsError): self.export()
        self.assertEqual(e.sha(self.path / 'export/bundle.json'), old)

    def test_probability_tamper(self):
        self.export()
        path = self.path / 'export/probabilities' / (IDS[0] + '.npy')
        with path.open('ab') as stream: stream.write(b'corrupt')
        with self.assertRaises(ValueError): e.verify_receipt(self.path / 'export/bundle.json', 'synthetic-only', IDS)

    def test_bundle_tamper(self):
        self.export()
        with (self.path / 'export/bundle.json').open('a') as stream: stream.write(' ')
        with self.assertRaises(ValueError): e.verify_receipt(self.path / 'export/bundle.json', 'synthetic-only', IDS)

    def test_receipt_contract(self):
        self.export()
        with self.assertRaises(ValueError): e.verify_receipt(self.path / 'export/bundle.json', 'foreign', IDS)

    def test_failed_receipt(self):
        self.export()
        e.save_new(self.path / 'export/failure.json', {'error': 'synthetic'})
        with self.assertRaises(ValueError): e.verify_receipt(self.path / 'export/bundle.json', 'synthetic-only', IDS)

    def test_id_safety(self):
        for ids in ([], [IDS[0]] * 2, ['../foreign'], ['a' * 14]):
            with self.assertRaises(ValueError): e.validate_ids(ids)

    def test_logit_contract(self):
        for value in ([torch.zeros(1)], torch.zeros(1, 2, 352, 352), torch.full((1, 1, 352, 352), float('inf'))):
            with self.assertRaises(ValueError): e.probabilities(value, 1, [352, 352])

    def test_probability_contract(self):
        for value in (np.zeros((352, 352), np.float64), np.full((352, 352), 1.1, np.float32), np.full((352, 352), np.nan, np.float32)):
            with self.assertRaises(ValueError): e.write_probability(self.path / 'bad.npy', value)

    def test_reserved_provenance(self):
        self.provenance['items'] = []
        with self.assertRaises(ValueError): self.export()

    def test_inputs_equal_frozen_adapter(self):
        adapter = e.load_module('rgbp06_input_test46', e.ROOT / '06_experiment_preparation/rgbp_glass_dataset.py')
        rgb = np.arange(13 * 17 * 3, dtype=np.uint16).reshape(13, 17, 3) * 97
        dolp = (rgb % 256).astype(np.uint8)
        mask = np.zeros((13, 17), np.float32)
        a, b, _ = adapter.transform_arrays(rgb[..., ::-1].astype(np.float32) / 65535,
                                           dolp.astype(np.float32) / 255, mask, 352, False)
        for mode in ('rgb', 'rgb_dolp'):
            expected = a if mode == 'rgb' else np.concatenate((a, b), axis=0)
            np.testing.assert_array_equal(e.input_tensor(rgb, dolp, mode, 352).numpy(), expected)

    def test_input_type_shape_guards(self):
        rgb = np.zeros((9, 10, 3), np.uint16)
        dolp = np.zeros((9, 10, 3), np.uint8)
        for a, b, mode in ((rgb.astype(np.uint8), dolp, 'rgb'), (rgb, dolp[:8], 'rgb'), (rgb, dolp, 'aolp')):
            with self.assertRaises(ValueError): e.input_tensor(a, b, mode, 352)

    def make_input(self):
        member = {'sample_id': IDS[0]}
        for name, suffix, dtype in (('image', 'rgb', np.uint16), ('dolp', 'dolp', np.uint8)):
            path = self.path / f'test/{name}/{IDS[0]}_{suffix}.tiff'
            path.parent.mkdir(parents=True, exist_ok=True)
            success, encoded = cv2.imencode('.tiff', np.ones((10, 11, 3), dtype=dtype))
            self.assertTrue(success)
            with path.open('xb') as stream: stream.write(encoded.tobytes())
            member[name] = path.relative_to(self.path).as_posix()
            member[name + '_sha256'] = e.sha(path)
        return member

    def test_dataset_never_needs_mask(self):
        member = self.make_input()
        item = e.BoundInputs(self.path, [member], 'rgb_dolp', 352)[0]
        self.assertEqual(set(item), {'image', 'stem'})
        self.assertEqual(tuple(item['image'].shape), (6, 352, 352))
        self.assertFalse((self.path / 'test/mask').exists())

    def test_changed_input_hash(self):
        member = self.make_input()
        member['image_sha256'] = '0' * 64
        with self.assertRaises(ValueError): e.BoundInputs(self.path, [member], 'rgb', 352)[0]

    def test_input_path_escape(self):
        member = self.make_input()
        member['image'] = '../foreign.tiff'
        with self.assertRaises(ValueError): e.BoundInputs(self.path, [member], 'rgb', 352)[0]

    def test_foreign_run(self):
        with self.assertRaises(ValueError): f.selected_item({'configs': []}, 'development_seed17')

    def test_lock_failure_precedes_pixels_model_and_checkpoint(self):
        cfg = {'run_id': 'fixture', 'output_dir': str(self.path / 'run')}
        formal = {'configs': [{'config': cfg, 'identity': {}}]}
        with patch.object(f, 'verify_contract', return_value=({}, {})), patch.object(f.protocol, 'read', return_value=formal), \
                patch.object(f.protocol, 'validate', return_value=formal), \
                patch.object(f.queue, 'byte_lock', side_effect=PermissionError('busy')), \
                patch.object(torch, 'load', side_effect=AssertionError('must not load')), \
                patch.object(e, 'BoundInputs', side_effect=AssertionError('must not read pixels')), \
                patch.object(f.core.t, 'choose_device', side_effect=AssertionError('must not select GPU')):
            with self.assertRaises(PermissionError): f.run('fixture')
        self.assertFalse((self.path / 'run').exists())

    def test_resource_guards(self):
        with patch.object(f.queue.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(1, 1, 0)):
            with self.assertRaises(RuntimeError): f.queue.require_resources()

    def test_running_actual_queue_blocks_before_load(self):
        with patch.object(torch, 'load', side_effect=AssertionError('must not deserialize')):
            with self.assertRaisesRegex(ValueError, 'All formal training must finish'):
                f.require_finished_queues({'configs': []})

    def test_parity_cpu_plumbing_only(self):
        record = f.device_parity(Echo(), Echo(), {'method': 'rgb_only'}, 'cpu', {'atol': 1e-5, 'rtol': 1e-4})
        self.assertEqual(record['logit_max_abs_difference'], 0)
        self.assertEqual(record['probability_max_abs_difference'], 0)
        self.assertEqual(record['samples'], 2)
        self.assertFalse(record['test_pixels_read'])

    def test_parity_refuses_error(self):
        class Offset(torch.nn.Module):
            def forward(self, x): return x[:, :1] + .01
        with self.assertRaises(AssertionError):
            f.device_parity(Echo(), Offset(), {'method': 'rgb_only'}, 'cpu', {'atol': 1e-5, 'rtol': 1e-4})

    def test_memory_guard(self):
        with patch.object(f.queue.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(30*1024**3, 0, 30*1024**3)), \
                patch.object(f.queue, 'available_memory', return_value=0):
            with self.assertRaises(RuntimeError): f.queue.require_resources()


def make_model_test(method):
    def test(self):
        channels = 3 if method == 'rgb_only' else 6
        generator = torch.Generator().manual_seed(46)
        image = torch.rand(1, channels, 352, 352, generator=generator)
        with f.core.installed_runtime():
            model = f.core.t.build_model(method, base_channels=4, seed=46).eval()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            local_checkpoint = self.path / 'self_generated_state.pt'
            torch.save(before, local_checkpoint)
            restored = f.core.t.build_model(method, base_channels=4, seed=99)
            restored.load_state_dict(torch.load(local_checkpoint, map_location='cpu', weights_only=True), strict=True)
            restored.eval()
            with torch.no_grad(): expected = torch.sigmoid(model(image).float().cpu()).numpy()[0, 0]
            bundle = e.export_batches(restored, [{'stem': [IDS[0]], 'image': image}], 'cpu',
                self.path / 'export', IDS[:1], [352, 352], self.provenance, lambda: None)
            item = bundle['items'][0]
            actual = e.offline.read_probability(self.path / 'export' / item['path'], item['sha256'], [352, 352])
            np.testing.assert_array_equal(actual, expected)
            f.core.backend.compare_tree(before, restored.state_dict(), {'atol': 0, 'rtol': 0})
            METHOD_RESULTS.append({'method': method, 'grid': [352, 352], 'base_channels': 4,
                'samples': 1, 'device': 'CPU', 'checkpoint_reload_and_export_exact': True,
                'model_and_BN_unchanged': True, 'probability_min': float(actual.min()), 'probability_max': float(actual.max())})
    return test


for method in f.core.candidate.METHODS:
    setattr(ExportTests, 'test_model_' + method, make_model_test(method))
