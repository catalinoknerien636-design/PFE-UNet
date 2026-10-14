"""Bounded synthetic checks, with no model construction or real images."""
from collections import OrderedDict
import io
from pathlib import Path
import pickle
import shutil
import unittest
from unittest.mock import patch
import warnings
import zipfile
import numpy as np
import torch
from torch.torch_version import TorchVersion
import metadata54 as m
import tensor54 as s
import convert54 as c

FIXTURES = m.HERE / 'cpu_attempt_v1/fixtures'


def sample():
    state = OrderedDict(weight=torch.arange(12, dtype=torch.float32).reshape(3, 4),
                        counter=torch.tensor(4, dtype=torch.int64))
    state._metadata = OrderedDict([('', {'version': 1})])
    return {'identity': {'versions': {'torch': TorchVersion('2.4.1+cpu')}, 'seed': 42},
            'epoch': 2, 'model_state': state, 'validation_iou': 0.25}


def item():
    return {'identity': {'versions': {'torch': '2.4.1+cpu'}, 'seed': 42},
            'best_epoch': 2, 'best_validation_iou': 0.25}


def fixture(path):
    # Recreate the observed old ZIP member layout from a synthetic CPU object.
    # Removing newer writer hints here is fixture-only, never a production repair.
    buffer = io.BytesIO(); torch.save(sample(), buffer, pickle_protocol=2)
    with zipfile.ZipFile(buffer) as src, zipfile.ZipFile(path, 'x') as dst:
        for name in src.namelist():
            suffix = name.split('/', 1)[1]
            if suffix in ('.format_version', '.storage_alignment'): continue
            dst.writestr(name, src.read(name))
    return path


class MetadataChecks(unittest.TestCase):
    def setUp(self):
        self.folder = FIXTURES / self.id().split('.')[-1]; self.folder.mkdir()
        self.source = fixture(self.folder / 'source.pt')
        self.digest = m.sha(self.source)
        with zipfile.ZipFile(self.source) as archive:
            _, self.name = m.archive_layout(archive); self.payload = archive.read(self.name)

    def test_original_default_reader_rejects(self):
        before = list(torch.serialization.get_safe_globals())
        with self.assertRaises(pickle.UnpicklingError):
            torch.load(self.source, map_location='cpu', weights_only=True)
        self.assertEqual(before, list(torch.serialization.get_safe_globals()))

    def test_normalized_default_reader_exact(self):
        output = self.folder / 'derived.pt'
        proof = m.normalize_archive(self.source, output, self.digest, '2.4.1+cpu')
        before = list(torch.serialization.get_safe_globals())
        result = torch.load(output, map_location='cpu', weights_only=True)
        self.assertIs(type(result['identity']['versions']['torch']), str)
        for key, value in sample()['model_state'].items(): self.assertTrue(torch.equal(result['model_state'][key], value))
        self.assertEqual(result['model_state']._metadata, sample()['model_state']._metadata)
        self.assertEqual(m.sha(self.source), self.digest)
        self.assertEqual(before, list(torch.serialization.get_safe_globals()))
        self.assertTrue(proof['storage_and_tensor_recipe_bytes_unchanged'])

    def test_other_member_bytes_equal(self):
        output = self.folder / 'derived.pt'
        m.normalize_archive(self.source, output, self.digest, '2.4.1+cpu')
        with zipfile.ZipFile(self.source) as a, zipfile.ZipFile(output) as b:
            self.assertEqual(a.namelist(), b.namelist())
            for name in a.namelist():
                if name != self.name: self.assertEqual(a.read(name), b.read(name))

    def test_wrong_version(self):
        with self.assertRaises(ValueError): m.normalize_metadata(self.payload, '2.4.2+cpu')

    def test_wrong_protocol(self):
        with self.assertRaises(ValueError): m.normalize_metadata(b'\x80\x03' + self.payload[2:], '2.4.1+cpu')

    def test_trailing_pickle_bytes(self):
        with self.assertRaises(ValueError): m.normalize_metadata(self.payload + b'.', '2.4.1+cpu')

    def test_missing_constructor(self):
        normalized, _ = m.normalize_metadata(self.payload, '2.4.1+cpu')
        with self.assertRaises(ValueError): m.normalize_metadata(normalized, '2.4.1+cpu')

    def test_additional_global(self):
        with self.assertRaises(ValueError): m.normalize_metadata(self.payload[:-1] + b'cos\nsystem\n.', '2.4.1+cpu')

    def test_additional_constructor(self):
        with self.assertRaises(ValueError): m.normalize_metadata(self.payload[:-1] + b'\x81.', '2.4.1+cpu')

    def test_removed_memo_reference(self):
        _, proof = m.normalize_metadata(self.payload, '2.4.1+cpu')
        for number in proof['removed_unreferenced_memos']:
            with self.subTest(memo=number), self.assertRaises(ValueError):
                m.normalize_metadata(self.payload[:-1] + b'j' + number.to_bytes(4, 'little') + b'.', '2.4.1+cpu')

    def test_wrong_source_hash(self):
        with self.assertRaises(ValueError): m.normalize_archive(self.source, self.folder/'no.pt', '0'*64, '2.4.1+cpu')
        self.assertFalse((self.folder/'no.pt').exists())

    def test_source_overwrite_rejected(self):
        with self.assertRaises(ValueError): m.normalize_archive(self.source, self.source, self.digest, '2.4.1+cpu')
        self.assertEqual(m.sha(self.source), self.digest)

    def test_existing_output_preserved(self):
        target = self.folder/'existing.pt'; target.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError): m.normalize_archive(self.source, target, self.digest, '2.4.1+cpu')
        self.assertEqual(target.read_bytes(), b'preserve')

    def bad_archive(self, change, extra=None, compression=zipfile.ZIP_STORED):
        target = self.folder/'bad.pt'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.source) as src, zipfile.ZipFile(target, 'x', compression=compression) as dst:
                for name in src.namelist():
                    new_name, value = change(name, src.read(name)); dst.writestr(new_name, value)
                if extra is not None: dst.writestr(*extra)
        with zipfile.ZipFile(target) as archive:
            with self.assertRaises(ValueError): m.archive_layout(archive)

    def test_duplicate_member(self): self.bad_archive(lambda n,b:(n,b), (self.name, self.payload))
    def test_foreign_member(self): self.bad_archive(lambda n,b:(n,b), ('archive/foreign', b'bad'))
    def test_path_traversal(self): self.bad_archive(lambda n,b:('archive/../data.pkl' if n==self.name else n,b))
    def test_compressed_archive(self): self.bad_archive(lambda n,b:(n,b), compression=zipfile.ZIP_DEFLATED)
    def test_wrong_byteorder(self): self.bad_archive(lambda n,b:(n,b'big' if n.endswith('/byteorder') else b))
    def test_wrong_archive_version(self): self.bad_archive(lambda n,b:(n,b'4\n' if n.endswith('/version') else b))


class StateChecks(unittest.TestCase):
    def setUp(self):
        self.folder = FIXTURES / self.id().split('.')[-1]; self.folder.mkdir()
        self.best = sample(); self.best['identity'] = item()['identity']
        self.metadata = c.validate_best(torch, self.best, item())
        self.binding = {'run_id': 'synthetic_only', 'sha256': 'a'*64}
        self.receipt = s.write_state(self.folder/'state', self.best['model_state'], self.metadata, self.binding)

    def test_npy_load_without_torch_load(self):
        with patch.object(torch, 'load', side_effect=AssertionError('No pickle provider')):
            result = s.load_state(self.folder/'state', self.receipt, self.binding)
        self.assertEqual(result['identity'], self.best['identity'])
        self.assertEqual(result['model_state']._metadata, self.best['model_state']._metadata)
        for key, value in result['model_state'].items(): self.assertTrue(torch.equal(value, self.best['model_state'][key]))

    def test_bad_receipt(self):
        with self.assertRaises(ValueError): s.verify_state(self.folder/'state', '0'*64, self.binding)

    def test_wrong_binding(self):
        with self.assertRaises(ValueError): s.verify_state(self.folder/'state', self.receipt, {'other': True})

    def test_foreign_state_file(self):
        (self.folder/'state/foreign').write_bytes(b'x')
        with self.assertRaises(ValueError): s.verify_state(self.folder/'state', self.receipt, self.binding)

    def test_changed_array(self):
        path = self.folder/'state/00000.npy'; value = bytearray(path.read_bytes()); value[-1] ^= 1; path.write_bytes(value)
        with self.assertRaises(ValueError): s.verify_state(self.folder/'state', self.receipt, self.binding)

    def test_trailing_array_bytes(self):
        path = self.folder/'state/00000.npy'
        with path.open('ab') as stream: stream.write(b'x')
        with self.assertRaises(ValueError): s.verify_state(self.folder/'state', self.receipt, self.binding)

    def test_metadata_mutation(self):
        path = self.folder/'state/manifest.json'; value = m.read(path); value['metadata']['epoch'] += 1
        path.write_bytes(m.store.json_bytes(value))
        with self.assertRaises(ValueError): s.verify_state(self.folder/'state', self.receipt, self.binding)

    def test_incomplete_state(self):
        (self.folder/'partial').mkdir()
        with self.assertRaises(FileNotFoundError): s.load_state(self.folder/'partial', self.receipt, self.binding)

    def test_no_overwrite_state(self):
        with self.assertRaises(FileExistsError): s.write_state(self.folder/'state', self.best['model_state'], self.metadata, self.binding)

    def test_outside_stage_rejected(self):
        with self.assertRaises(ValueError): s.owned_path(m.ROOT/'not_stage54')

    def test_wrong_identity_rejected(self):
        wrong = dict(item()); wrong['identity'] = {'wrong': True}
        with self.assertRaises(ValueError): c.validate_best(torch, self.best, wrong)

    def test_wrong_best_epoch_rejected(self):
        wrong = dict(item()); wrong['best_epoch'] = 1
        with self.assertRaises(ValueError): c.validate_best(torch, self.best, wrong)

    def test_nonfinite_rejected(self):
        self.best['model_state']['weight'][0,0] = float('nan')
        with self.assertRaises(ValueError): c.validate_best(torch, self.best, item())

    def test_unexpected_dtype_rejected(self):
        self.best['model_state']['weight'] = self.best['model_state']['weight'].double()
        with self.assertRaises(ValueError): c.validate_best(torch, self.best, item())

    def test_parameter_wrapper_rejected(self):
        self.best['model_state']['weight'] = torch.nn.Parameter(self.best['model_state']['weight'])
        with self.assertRaises(ValueError): c.validate_best(torch, self.best, item())

    def test_noncontiguous_rejected(self):
        self.best['model_state']['weight'] = self.best['model_state']['weight'].t()
        with self.assertRaises(ValueError): c.validate_best(torch, self.best, item())

    def test_source_version_subclass_rejected(self):
        self.best['identity']['versions']['torch'] = TorchVersion('2.4.1+cpu')
        with self.assertRaises(ValueError): c.validate_best(torch, self.best, item())

    def test_injected_write_failure_retained(self):
        with patch.object(np.lib.format, 'write_array', side_effect=OSError('injected')):
            with self.assertRaises(OSError): s.write_state(self.folder/'failed', self.best['model_state'], self.metadata, self.binding)
        self.assertTrue((self.folder/'failed/failure.json').exists())
        self.assertFalse((self.folder/'failed/complete.json').exists())

