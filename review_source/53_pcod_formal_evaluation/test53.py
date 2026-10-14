"""Synthetic CPU checks; real admission tests may inspect metadata, never pixels."""
import io
import os
from pathlib import Path
import pickle
import unittest
from unittest import mock
import numpy as np
from PIL import Image
import torch
import pcod_eval as e
import formal53 as f

FIXTURES = None


def picture(array, mode=None):
    stream = io.BytesIO()
    Image.fromarray(array, mode=mode).save(stream, format='PNG')
    return stream.getvalue()


class Checks(unittest.TestCase):
    def setUp(self):
        self.folder = FIXTURES / self._testMethodName
        self.folder.mkdir()

    def test_rgb_dop_matches_frozen_loader(self):
        from scripts.extension_dataset import ExtensionPCODDataset
        from types import SimpleNamespace
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, (13, 19, 3), dtype=np.uint8)
        dop = rng.integers(0, 256, (13, 19), dtype=np.uint8)
        a, b = picture(rgb), picture(dop)
        (self.folder / 'rgb.png').write_bytes(a); (self.folder / 'dop.png').write_bytes(b)
        dataset = object.__new__(ExtensionPCODDataset)
        dataset.image_size = 352; dataset.aux_name = 'dop'
        dataset.registration_aug = False; dataset.fixed_shift = None; dataset.fixed_rotation = None
        sample = SimpleNamespace(rgb_path=self.folder / 'rgb.png', dop_path=self.folder / 'dop.png')
        _, old_rgb = dataset._load_rgb(sample.rgb_path)
        old_dop = np.asarray(dataset._load_aux(sample, old_rgb), dtype=np.float32) / 255.0
        expected = np.concatenate((old_rgb.transpose(2, 0, 1), old_dop[None]), axis=0).astype(np.float32)
        np.testing.assert_array_equal(e.input_tensor(a, b).numpy(), expected)

    def test_mask_matches_pil_resize_then_threshold(self):
        from scripts.pcod_dataset import load_mask
        value = np.resize(np.array([0, 1, 127, 128, 254, 255], np.uint8), (11, 17))
        payload = picture(value)
        path = self.folder / 'mask.png'; path.write_bytes(payload)
        np.testing.assert_array_equal(e.mask_grid(payload), load_mask(path, 352)[0])

    def test_input_does_not_open_mask(self):
        a = picture(np.zeros((3, 4, 3), np.uint8)); b = picture(np.full((3, 4), 255, np.uint8))
        (self.folder / 'rgb.png').write_bytes(a); (self.folder / 'dop.png').write_bytes(b)
        member = {'sample_id': 'sample_0', 'rgb': 'rgb.png', 'dop': 'dop.png', 'mask': 'DO_NOT_OPEN',
                  'rgb_sha256': e.sha(self.folder / 'rgb.png'), 'dop_sha256': e.sha(self.folder / 'dop.png')}
        batch = e.BoundInputs(self.folder, [member])[0]
        self.assertEqual(set(batch), {'image', 'stem'})
        self.assertTrue(torch.all(batch['image'][:3] == 0) and torch.all(batch['image'][3] == 1))

    def test_input_hash_refusal(self):
        path = self.folder / 'a.png'; path.write_bytes(picture(np.zeros((2, 2), np.uint8)))
        with self.assertRaises(ValueError):
            e.bound_payload(self.folder, {'rgb': 'a.png', 'rgb_sha256': '0' * 64}, 'rgb')

    def test_input_path_refusal(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            e.bound_payload(self.folder, {'rgb': '../outside.png', 'rgb_sha256': '0' * 64}, 'rgb')

    def test_sigmoid_precedes_float_conversion(self):
        x = torch.linspace(-10, 10, 352 * 352, dtype=torch.float64).reshape(1, 1, 352, 352)
        result = e.probabilities(x, 1)
        np.testing.assert_array_equal(result, torch.sigmoid(x).float().numpy()[:, 0])
        self.assertTrue(np.any(result != torch.sigmoid(x.float()).numpy()[:, 0]))

    def test_nonfinite_logits_refused(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError): e.probabilities(torch.full((1, 1, 352, 352), value), 1)

    def test_logits_shape_refused(self):
        for shape in ((1, 352, 352), (1, 2, 352, 352), (1, 1, 416, 416)):
            with self.assertRaises(ValueError): e.probabilities(torch.zeros(shape), 1)

    def test_metrics_equal_original_single_and_partitioned(self):
        generator = torch.Generator().manual_seed(19)
        logits = torch.randn(5, 1, 352, 352, generator=generator)
        masks = (torch.rand(5, 1, 352, 352, generator=generator) > .63).float()
        logits[0, :, :100] = 0
        ids = ['sample_' + str(i) for i in range(5)]
        a, b = f.core.t.Metrics(), f.core.t.Metrics()
        a.update(logits, masks, ids)
        b.update(logits[:4], masks[:4], ids[:4]); b.update(logits[4:], masks[4:], ids[4:])
        rows = [e.image_metrics(s, p, m.numpy()[0]) for s, p, m in zip(ids, e.probabilities(logits, 5), masks)]
        result = e.aggregate(rows)
        for key, value in a.finalize().items():
            self.assertEqual(result[key], value); self.assertEqual(b.finalize()[key], value)
        for row, old in zip(rows, a.rows):
            for key in ('tp', 'fp', 'fn', 'tn', 'mae'): self.assertEqual(row[key], old[key])

    def test_empty_and_threshold_boundary_conventions(self):
        y = np.zeros((352, 352), np.float32)
        row = e.image_metrics('empty', np.full_like(y, .5), y)
        result = e.aggregate([row])
        self.assertEqual(result['iou'], 0); self.assertEqual(row['boundary_f1_2px'], 1)
        self.assertEqual(result['both_empty_images'], 1); self.assertEqual(row['tp'] + row['fp'], 0)

    def test_bad_probability_and_mask_refused(self):
        p = np.zeros((352, 352), np.float32)
        for a, y in ((p.astype('float64'), p), (p + 2, p), (p, p + .5), (p, p.astype('uint8'))):
            with self.assertRaises(ValueError): e.image_metrics('bad', a, y)

    def test_duplicate_images_refused(self):
        p = np.zeros((352, 352), np.float32)
        row = e.image_metrics('dup', p, p)
        with self.assertRaises(ValueError): e.aggregate([row, row])

    def test_seed_pair_minimum_n5(self):
        a, b = ({s: .7 for s in e.SEEDS}, {s: .5 for s in e.SEEDS})
        self.assertEqual(e.paired(a, b, 'direct_reference', sign_flip=True)['two_sided_non_mid_p'], .0625)

    def test_seed_pair_minimum_n3(self):
        a, b = ({s: .7 for s in e.SEEDS[:3]}, {s: .5 for s in e.SEEDS[:3]})
        self.assertEqual(e.paired(a, b, e.METHODS[-1], sign_flip=True)['two_sided_non_mid_p'], .25)

    def test_zero_differences_p_one(self):
        a = {s: .5 for s in e.SEEDS}
        result = e.paired(a, a, 'direct_reference', sign_flip=True)
        self.assertEqual(result['two_sided_non_mid_p'], 1); self.assertEqual(result['sample_sd'], 0)

    def test_five_vs_three_rejected(self):
        with self.assertRaises(ValueError):
            e.paired({s: .5 for s in e.SEEDS}, {s: .5 for s in e.SEEDS[:3]}, e.METHODS[-1])

    def test_wrong_seed_and_nonfinite_rejected(self):
        for a in ({0: .2}, {s: float('nan') for s in e.SEEDS}, {str(s): .4 for s in e.SEEDS}):
            with self.assertRaises(ValueError): e.paired(a, a, 'direct_reference')

    def test_complete_matrix_statistics(self):
        keys = ('iou', 'f1', 'image_mae', 'boundary_f1_2px_image_mean', 'boundary_iou_pooled')
        reports = [{'method': m, 'seed': s, 'all_test': {k: .4 + .01 * e.METHODS.index(m) for k in keys}}
                   for m in e.METHODS for s in (e.SEEDS[:3] if m == e.METHODS[-1] else e.SEEDS)]
        value = e.statistics(reports)
        self.assertEqual(len(value['contrasts']), 5)
        self.assertEqual(value['contrasts'][e.METHODS[-1]]['iou']['n'], 3)
        self.assertNotIn('two_sided_non_mid_p', value['contrasts']['wide_direct']['iou'])
        for wrong in (reports[:-1], reports + [reports[0]], reports[:-1] + [reports[0]]):
            with self.assertRaises(ValueError): e.statistics(wrong)

    def test_npy_roundtrip_and_trailing_refusal(self):
        p = np.linspace(0, 1, 352 * 352, dtype=np.float32).reshape(352, 352)
        path = self.folder / 'p.npy'; digest = e.write_probability(path, p)
        np.testing.assert_array_equal(e.read_probability(path, digest, e.GRID), p)
        with path.open('ab') as stream: stream.write(b'x')
        with self.assertRaises(ValueError): e.read_probability(path, e.sha(path), e.GRID)

    def test_npy_dtype_version_refused(self):
        for i, dtype in enumerate(('float64', 'uint8')):
            path = self.folder / (str(i) + '.npy'); np.save(path, np.zeros((352, 352), dtype=dtype))
            with self.assertRaises(ValueError): e.read_probability(path, e.sha(path), e.GRID)

    def batches(self):
        return [{'stem': ['sample_0', 'sample_1'], 'image': torch.zeros(2, 4, 352, 352)}]

    def test_export_complete_transaction(self):
        model = torch.nn.Conv2d(4, 1, 1)
        output = self.folder / 'out'
        bundle = e.export_batches(model, self.batches(), 'cpu', output, ['sample_0', 'sample_1'],
                                  {'contract_sha256': 'a' * 64}, lambda: None)
        checked = e.verify_receipt(output / 'bundle.json', 'a' * 64, ['sample_0', 'sample_1'])
        self.assertEqual(bundle, checked)

    def test_export_incomplete_preserves_failure(self):
        output = self.folder / 'out'
        with self.assertRaises(ValueError):
            e.export_batches(torch.nn.Conv2d(4, 1, 1), self.batches(), 'cpu', output,
                ['sample_0', 'sample_1', 'missing'], {'contract_sha256': 'a' * 64}, lambda: None)
        self.assertTrue((output / 'failure.json').exists())
        self.assertFalse((output / 'export_complete.json').exists())

    def test_export_masks_and_reorder_refused(self):
        for i, batch in enumerate((dict(self.batches()[0], mask=torch.zeros(2, 1, 352, 352)),
                                   dict(self.batches()[0], stem=['sample_1', 'sample_0']))):
            with self.assertRaises(ValueError):
                e.export_batches(torch.nn.Conv2d(4, 1, 1), [batch], 'cpu', self.folder / str(i),
                    ['sample_0', 'sample_1'], {'contract_sha256': 'a' * 64}, lambda: None)

    def test_final_provenance_failure_no_receipt(self):
        def reject(): raise ValueError('changed')
        with self.assertRaises(ValueError):
            e.export_batches(torch.nn.Conv2d(4, 1, 1), self.batches(), 'cpu', self.folder / 'out',
                ['sample_0', 'sample_1'], {'contract_sha256': 'a' * 64}, reject)
        self.assertFalse((self.folder / 'out/export_complete.json').exists())

    def test_no_overwrite_and_bad_ids(self):
        for ids in ([], ['a', 'a'], ['../a'], ['a/b']):
            with self.assertRaises(ValueError): e.validate_ids(ids)
        with self.assertRaises(FileExistsError):
            e.export_batches(torch.nn.Conv2d(4, 1, 1), [], 'cpu', self.folder, ['x'], {}, lambda: None)

    def test_small_five_models_reload_export(self):
        for method in e.METHODS[:5]:
            model = f.core.backend.ORIGINAL_BUILD(method, base_channels=4, seed=42).eval()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            path = self.folder / (method + '.pt'); torch.save(before, path)
            second = f.core.backend.ORIGINAL_BUILD(method, base_channels=4, seed=42).eval()
            second.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            batch = {'image': torch.rand(1, 4, 352, 352, generator=torch.Generator().manual_seed(53)), 'stem': ['sample_0']}
            with torch.no_grad(): expected = e.probabilities(model(batch['image']), 1)[0]
            folder = self.folder / method
            e.export_batches(second, [batch], 'cpu', folder, ['sample_0'], {'contract_sha256': 'b' * 64}, lambda: None)
            np.testing.assert_array_equal(e.read_probability(folder / 'probabilities/sample_0.npy',
                                          e.sha(folder / 'probabilities/sample_0.npy'), e.GRID), expected)
            for key, value in second.state_dict().items(): self.assertTrue(torch.equal(value, before[key]))
            del model, second, before

    def test_strict_local_best_plain_metadata(self):
        path = self.folder / 'best.pt'
        value = {'identity': {'version': 'plain'}, 'epoch': 3, 'validation_iou': .4,
                 'model_state': {'weight': torch.ones(1)}}
        torch.save(value, path)
        loaded = f.load_selected_best(path, value['identity'], {'best_epoch': 3, 'best_validation_iou': .4}, e.sha(path))
        self.assertTrue(torch.equal(loaded['model_state']['weight'], torch.ones(1)))

    def test_unsupported_local_metadata_no_unsafe_fallback(self):
        from torch.torch_version import TorchVersion
        path = self.folder / 'best.pt'
        torch.save({'identity': {'torch': TorchVersion('2.4.1')}}, path)
        original = torch.load
        with mock.patch.object(torch, 'load', wraps=original) as call:
            with self.assertRaises(pickle.UnpicklingError):
                f.load_selected_best(path, {}, {'best_epoch': 3}, e.sha(path))
            self.assertEqual(call.call_count, 1); self.assertTrue(call.call_args.kwargs['weights_only'])

    def test_formal_unknown_run_refused_before_checkpoint(self):
        formal = {'configs': [{'config': {'run_id': 'valid'}}]}
        with mock.patch.object(torch, 'load', side_effect=AssertionError('No load')):
            with self.assertRaises(ValueError): f.selected(formal, 'development')

    def test_actual_training_gate_refuses_before_checkpoint(self):
        with mock.patch.object(torch, 'load', side_effect=AssertionError('No load')):
            with self.assertRaises(ValueError): f.require_finished_metadata()

    def test_duplicate_json_refused(self):
        path = self.folder / 'bad.json'; path.write_text('{"a":1,"a":2}', encoding='utf-8')
        with self.assertRaises(ValueError): e.read(path)


if __name__ == '__main__':
    raise SystemExit('Use the exclusive CPU-check runner')
